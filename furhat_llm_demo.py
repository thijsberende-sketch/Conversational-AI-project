import asyncio
import json
import os
import re

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from sklearn.tree import DecisionTreeClassifier

from furhat_robot import FurhatRobot

import question_generator as qg
import robot_settings

QUIT_WORDS = {"quit", "exit", "goodbye", "stop"}

LLM_PROVIDER = "openai"


def create_llm() -> tuple[OpenAI, str]:
    """Create the LLM client for the provider chosen with LLM_PROVIDER in .env."""
    global LLM_PROVIDER
    
    # Choosing whether we use the class openai API or the free openrouter API, based on the .env file
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your .env file, or set LLM_PROVIDER=openrouter.")
        llm = OpenAI(api_key=api_key, max_retries=0, timeout=30.0)
        return llm, os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    if LLM_PROVIDER == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY in your .env file, or set LLM_PROVIDER=openai.")
        llm = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            timeout=30.0,
        )
        model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        if model != "openrouter/free" and not model.endswith(":free"):
            raise ValueError(
                "To keep requests free, OPENROUTER_MODEL must be 'openrouter/free' "
                "or a model ID ending with ':free'."
            )
        return llm, model

    raise ValueError(f"LLM_PROVIDER must be 'openai' or 'openrouter', not '{LLM_PROVIDER}'.")


async def main() -> None:
    load_dotenv()

    furhat_host = os.getenv("FURHAT_HOST")
    if not furhat_host:
        raise ValueError("Set FURHAT_HOST to the IP address of your Furhat.")

    llm, model = create_llm()
    print(f"Using {LLM_PROVIDER} model {model}")
    robot = FurhatRobot()

    robot.connect()
    
    # basic logic to run the game in a loop, asking if the user wants to play again after each round
    try:
        await robot.connect_realtime()
        robot.attend_user()

        while await play_round(robot, llm, model):
            if await ask_yes_no(robot, "Do you want to play again?") is not True:
                await speak(robot, "Thanks for playing. Goodbye!")
                break
    finally:
        await robot.stop_listen()
        await robot.disconnect_realtime()
        robot.disconnect()


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


async def speak(robot: FurhatRobot, text: str) -> None:
    # robot.speak blocks until speech is done; run it in a thread so the realtime event loop keeps running
    await asyncio.to_thread(robot.speak, text)


async def ask(
    robot: FurhatRobot,
    question: str,
    time_to_wait: float = robot_settings.END_SPEECH_TIMEOUT,
) -> str | None:
    """Speak a question and return the user's answer, or None if nothing was heard."""
    # Create a future to hold the user's answer
    answer_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    # callback function that handles eventual response from the user and sets the result of the future
    def on_final_text(text: str) -> None:
        if not answer_future.done():
            answer_future.set_result(text.strip())

    robot.on_final_text = on_final_text
    await speak(robot, question)

    # custom timing can be added here to wait for the user to answer, would be cool to have a more natural timing,
    # but this is not a big problem for this demo
    await robot.start_listen(languages=robot_settings.LISTEN_LANGUAGES, end_speech_timeout=time_to_wait)
    try:
        return await asyncio.wait_for(answer_future, timeout=60)
    except asyncio.TimeoutError:
        return None
    finally:
        await robot.stop_listen()


async def ask_yes_no(robot: FurhatRobot, question: str, attempts: int = 3) -> bool | None:
    """
        Ask a yes/no question, re-asking on unclear answers. Quitting counts as no." 
        Currently forces the user to answer yes or no, but could be changed to accept more natural answers 
        like "maybe" or "I don't know".
    """
    for _ in range(attempts):
        user_text = await ask(robot, question)
        if user_text:
            print(f"You: {user_text}")
            if is_quit(user_text):
                return False
            answer = parse_yes_no(user_text)
            if answer is not None:
                return answer
        await speak(robot, "Sorry, please answer yes or no.")
    return None

 # Send a message to the LLM and return its reply.
async def llm_respond(
    llm,
    model,
    message,
    *,
    max_tokens: int = 128,
    response_format: dict[str, str] | None = None,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "Follow the user's instructions exactly, do not repeat the prompt. Keep the response concise.",
        },
        {"role": "user", "content": message},
    ]

    for attempt in range(5):
        try:
            request_options = {"max_tokens": max_tokens}
            if response_format is not None:
                request_options["response_format"] = response_format
            # OpenRouter-only option; OpenAI rejects unknown request fields
            if LLM_PROVIDER == "openrouter" and model != "openrouter/free":
                request_options["extra_body"] = {
                    "provider": {"sort": "throughput"}
                }

            completion = await asyncio.to_thread(
                llm.chat.completions.create,
                model=model,
                messages=messages,
                **request_options,
            )
            if LLM_PROVIDER == "openrouter" and completion.model != model:
                print(f"OpenRouter selected model: {completion.model}")
            break
        except APIStatusError as error:
            if error.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 4:
                raise
            delay = 0.8 * (2**attempt)
            print(
                f"{LLM_PROVIDER} returned HTTP {error.status_code}; "
                f"retrying in {delay:.1f}s (attempt {attempt + 2}/5)."
            )
            await asyncio.sleep(delay)
        except APIConnectionError:
            if attempt == 4:
                raise
            delay = 0.8 * (2**attempt)
            print(
                f"{LLM_PROVIDER} connection failed; retrying in {delay:.1f}s "
                f"(attempt {attempt + 2}/5)."
            )
            await asyncio.sleep(delay)

    if not completion.choices:
        raise RuntimeError("The LLM returned no choices.")
    reply = completion.choices[0].message.content
    if not reply:
        raise RuntimeError("The LLM returned an empty reply.")

    print(f"Furhat: {reply}")
    return reply


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


def extract_json_object(text: str) -> str:
    """Strip markdown fences or surrounding text that some models add despite instructions."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return text
    return text[start:end + 1]


def mentions(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase.lower())}\b", text.lower()) is not None


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


def normalize(user_text: str) -> str:
    normalized_text = user_text.lower().replace("’", "'").replace("-", " ")
    answer = re.sub(r"[^\w\s']", "", normalized_text).strip()
    return re.sub(r"\s+", " ", answer)


def is_quit(user_text: str) -> bool:
    return normalize(user_text) in QUIT_WORDS


def parse_yes_no(user_text: str) -> bool | None:
    answer = normalize(user_text)

    unclear_prefixes = (
        "i don't know",
        "i do not know",
        "don't know",
        "dunno",
        "no idea",
        "not sure",
        "i'm not sure",
        "maybe",
        "unclear",
        "can't tell",
        "cannot tell",
        "no clue",
        "who knows",
    )
    if any(answer == phrase or answer.startswith(phrase + " ") for phrase in unclear_prefixes):
        return None

    yes_prefixes = (
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "certainly",
        "absolutely",
        "definitely",
        "of course",
        "exactly",
        "true",
        "correct",
        "right",
        "that's right",
        "that is right",
        "it is",
        "it was",
        "they are",
        "i do",
        "i have",
        "i am",
        "i think so",
        "i believe so",
        "uh huh",
        "mm hmm",
        "also yes"
    )
    no_prefixes = (
        "no",
        "nope",
        "nah",
        "not really",
        "not at all",
        "definitely not",
        "certainly not",
        "absolutely not",
        "of course not",
        "incorrect",
        "wrong",
        "false",
        "negative",
        "that's not right",
        "that is not right",
        "it isn't",
        "it's not",
        "it is not",
        "it wasn't",
        "it was not",
        "they aren't",
        "they are not",
        "i don't",
        "i do not",
        "i haven't",
        "i have not",
        "i am not",
        "i'm not",
        "i think not",
        "i don't think so",
        "i do not think so",
        "uh uh",
        "mm mm",
        "also no"
    )

    def longest_match(prefixes: tuple[str, ...]) -> int:
        return max(
            (
                len(phrase)
                for phrase in prefixes
                if answer == phrase or answer.startswith(phrase + " ")
            ),
            default=0,
        )

    # The longest matching phrase wins, so "it is not" counts as no rather than matching "it is"
    yes_length = longest_match(yes_prefixes)
    no_length = longest_match(no_prefixes)
    if yes_length == no_length:
        return None
    return yes_length > no_length


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


if __name__ == "__main__":
    """
        Run the Furhat LLM demo, connecting to the robot and the LLM provider specified in .env.
        It runs asynchronously for parallel execution, so the main function is wrapped in asyncio.run() to start the event loop.
    """
    asyncio.run(main())
