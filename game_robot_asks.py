"""
Game mode 1: the user thinks of a story and Furhat guesses it by asking yes/no questions.

The questions come from a decision tree over sample_data.xlsx; the LLM only rewrites
each split into a natural-sounding question.
"""

import asyncio
import json

from sklearn.tree import DecisionTreeClassifier

from furhat_robot import FurhatRobot
from furhat_llm_demo import (
    ask,
    ask_yes_no,
    extract_json_object,
    is_quit,
    llm_respond,
    mentions,
    parse_yes_no,
    speak,
)

import question_generator as qg


async def play_round(robot: FurhatRobot, llm, model: str) -> bool:
    """Play one guessing game. Returns False when the user asked to quit."""
    # A new tree every round, so the questions vary between games
    dtree, features, target = qg.generate_decision_tree()
    print(f"Decision tree generated with features: {features} and target: {target}")

    # track where in the decision tree we are; start at the root node
    node_tracker = 0

    # Preload questions by tree node; each bundle request generates both child questions.
    question_cache: dict[int, str] = await request_question_bundle(
        llm,
        model,
        dtree,
        features,
        [0, int(dtree.tree_.children_left[0]), int(dtree.tree_.children_right[0])],
    )
    prefetched_bundles: dict[int, asyncio.Task[dict[int, str]]] = {}

    try:
        while dtree.tree_.feature[node_tracker] >= 0:
            # Start generating the next questions before speaking, so the LLM works while the robot talks. This can be optimized further by prefetching more than one level of questions,
            # but that would require more complex caching and tracking of which nodes have been prefetched.
            prefetch_child_questions(
                llm,
                model,
                dtree,
                features,
                node_tracker,
                question_cache,
                prefetched_bundles,
            )

            # Get the question for the current node, either from the preloaded ones or by generating a fallback question
            split = qg.describe_split(dtree, features, node_tracker)
            question = question_cache.get(node_tracker) or qg.fallback_question(split)
            print(f"Node {node_tracker} question: {question}")


            # Ask the user the question and interpret their answer, repeating if necessary
            user_text = await ask(robot, question)
            if user_text is None:
                await speak(robot, "I didn't hear anything. Let's try again.")
                continue
            if not user_text:
                await speak(robot, "Sorry, I didn't catch that. Please try again.")
                continue
            print(f"You: {user_text}")
            if is_quit(user_text):
                await speak(robot, "Goodbye!")
                return False


            # Interpret the answer if it was a binary yes/no answer or a category name, and move down the tree accordingly
            answer_is_yes = interpret_answer(user_text, split, features)
            if answer_is_yes is None:
                await speak(robot, "Please answer yes or no.")
                continue

            previous_node = node_tracker
            if answer_is_yes == split["yes_goes_left"]:
                node_tracker = int(dtree.tree_.children_left[node_tracker])
            else:
                node_tracker = int(dtree.tree_.children_right[node_tracker])

            bundle_task = prefetched_bundles.pop(previous_node, None)
            if bundle_task is not None:
                question_cache.update(await bundle_task)
    finally:
        for task in prefetched_bundles.values():
            if not task.done():
                task.cancel()
        if prefetched_bundles:
            await asyncio.gather(*prefetched_bundles.values(), return_exceptions=True)

    # If we reach a leaf node we make a prediction and check if we are correct, but we get stuck if the asnwer is wrong
    # We also only guess onnce we fully traverse the tree, which is a bit boring
    class_counts = dtree.tree_.value[node_tracker][0]
    predicted_class = dtree.classes_[max(range(len(class_counts)), key=class_counts.__getitem__)]
    correct = await ask_yes_no(robot, f"I think you're thinking of {predicted_class}. Am I right?")
    if correct:
        await speak(robot, "Great, I got it!")
    elif correct is False:
        await speak(robot, "Oh no! Maybe one of your answers didn't match my notes. I'll do better next time.")
    return True


async def request_question_bundle(
    llm,
    model: str,
    dtree: DecisionTreeClassifier,
    features: list[str],
    node_ids: list[int],
) -> dict[int, str]:
    """Generate questions for the given nodes. Never raises: unusable LLM output falls back to template questions."""
    prompts: dict[str, str] = {}
    for node_id in dict.fromkeys(node_ids):
        split_data = qg.split_prompt_for_node(dtree, features, node_id)
        if split_data is None:
            continue
        prompts[str(node_id)] = split_data[0].replace(
            "Return only the question.",
            "",
        ).strip()

    if not prompts:
        return {}

    bundle_prompt = (
        "Generate one short, natural yes/no question for each tree node below. "
        "Each node description specifies the exact condition the question must test. "
        "Return only a valid JSON object mapping each node ID to its question. "
        "Do not include analysis, markdown, or extra keys.\n\n"
        f"Node descriptions:\n{json.dumps(prompts, ensure_ascii=False)}"
    )
    decoded: dict = {}
    try:
        response = await llm_respond(
            llm,
            model,
            bundle_prompt,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        decoded = json.loads(extract_json_object(response))
        if not isinstance(decoded, dict):
            raise ValueError("LLM question bundle must be a JSON object.")
    except Exception as error:
        print(f"Question generation failed, using template questions: {error}")
        decoded = {}

    questions: dict[int, str] = {}
    for node_id in prompts:
        split = qg.describe_split(dtree, features, int(node_id))
        question = decoded.get(node_id)
        if isinstance(question, str) and question.strip() and question_matches_split(question, split, features):
            questions[int(node_id)] = question.strip()
        else:
            print(f"No usable LLM question for node {node_id}, using template question.")
            questions[int(node_id)] = qg.fallback_question(split)
    return questions


def question_matches_split(question: str, split: dict, features: list[str]) -> bool:
    """
    Reject questions that ask about the wrong category of the tested column,
    e.g. "Is the author female?" for a split on Author Gender = Male. Such a
    question would flip the meaning of yes and no and send the game down the wrong branch.
    We assume that yes goes left.
    """
    if split["category"] is None:
        return True
    other_categories = [
        category
        for category in qg.column_categories(features, split["column"])
        if category != split["category"] and category.lower() not in {"yes", "no"}
    ]
    return not any(mentions(question, category) for category in other_categories)


def prefetch_child_questions(
    llm,
    model: str,
    dtree: DecisionTreeClassifier,
    features: list[str],
    node_tracker: int,
    question_cache: dict[int, str],
    prefetched_bundles: dict[int, asyncio.Task[dict[int, str]]],
) -> None:
    child_nodes = [
        int(dtree.tree_.children_left[node_tracker]),
        int(dtree.tree_.children_right[node_tracker]),
    ]
    missing_children = [
        child_node
        for child_node in child_nodes
        if dtree.tree_.feature[child_node] >= 0
        and child_node not in question_cache
    ]
    if not missing_children or node_tracker in prefetched_bundles:
        return

    prefetched_bundles[node_tracker] = asyncio.create_task(
        request_question_bundle(
            llm,
            model,
            dtree,
            features,
            missing_children,
        )
    )


def interpret_answer(user_text: str, split: dict, features: list[str]) -> bool | None:
    """Parse a yes/no answer, also accepting a category name, e.g. "Male" to "Is the author male?"."""
    answer = parse_yes_no(user_text)
    if answer is not None or split["category"] is None:
        return answer
    if split["category"].lower() in {"yes", "no"}:
        return None

    says_category = mentions(user_text, split["category"])
    says_other = any(
        mentions(user_text, category)
        for category in qg.column_categories(features, split["column"])
        if category != split["category"]
    )
    if says_category == says_other:
        return None
    return says_category
