"""
Game mode 1: the user thinks of a literary work and Furhat guesses it by asking yes/no questions.

The questions come from a decision tree over the literary works data; the LLM only rewrites
each split into a natural-sounding question. Furhat guesses as soon as only a few works fit the
answers. If none fit any more (a wrong answer, or a work that is not in the data), the LLM asks
a few extra questions and makes one final guess.
"""

import asyncio
import json
import random

from sklearn.tree import DecisionTreeClassifier

import furhat_llm_demo
from furhat_robot import FurhatRobot
from furhat_llm_demo import (
    ask,
    extract_json_object,
    is_quit,
    llm_respond,
    mentions,
    parse_yes_no,
    speak,
)

import question_generator as qg

# Start guessing once this many works or fewer still fit the answers
GUESS_THRESHOLD = 3
# How many extra questions the LLM asks once the decision tree has no works left, before its final guess
EXTRA_LLM_QUESTIONS = 3

# Waiting for the LLM, in seconds: after THINKING_FILLER_AFTER the robot says "let me think", so it
# never just goes silent. After the wait limit it goes on without the LLM.
THINKING_FILLER_AFTER = 2.5
QUESTION_WAIT = 8  # tree questions; afterwards the template question is used
LLM_FALLBACK_WAIT = 15  # extra questions, final guess and looking up the player's work

# Stop the round when nothing is heard this many times in a row (the player probably left), or when the
# answer to one question stays unclear this many times
MAX_SILENT_TRIES = 3
MAX_UNCLEAR_TRIES = 5


async def wait_for_llm(robot: FurhatRobot, awaitable, timeout: float, fallback):
    """Wait at most `timeout` seconds for an LLM request, saying "let me think" if it takes a moment."""
    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    start = loop.time()
    done, _ = await asyncio.wait({task}, timeout=THINKING_FILLER_AFTER)
    if not done:
        await speak(robot, random.choice(["Hmm, let me think.", "Let me think for a moment.", "Hmm."]))
    try:
        return await asyncio.wait_for(task, max(0.0, timeout - (loop.time() - start)))
    except asyncio.TimeoutError:
        print(f"The LLM took longer than {timeout:g}s; going on without it.")
    except Exception as error:
        print(f"The LLM request failed: {error}")
    return fallback


class QuitGame(Exception):
    """Raised when the user says quit/stop/goodbye in the middle of a round."""


async def welcome(robot: FurhatRobot) -> None:
    """Said once, before the first round."""
    await speak(
        robot,
        "Hello, and welcome to the guessing game! "
        "You think of a literary work, like a poem, a short story or an essay, "
        "and I will try to guess it by asking you yes or no questions. "
        "If you are not sure about an answer, just give your best guess. "
        "You can say stop at any time to end the game.",
    )


async def farewell(robot: FurhatRobot) -> None:
    """Said once, when the user is done playing."""
    await speak(
        robot,
        "Thank you for playing with me! I hope you enjoyed it, "
        "and maybe learned something about these works along the way. Goodbye!",
    )


async def play_round(robot: FurhatRobot, llm, model: str) -> bool:
    """Play one guessing game. Returns False when the user asked to quit."""
    # A new tree every round, so the questions vary between games
    dtree, features, target = qg.generate_decision_tree()
    print(f"Decision tree generated with features: {features} and target: {target}")

    question_cache: dict[int, str] = {}
    # Node -> the running LLM request that generates its question. Finished requests add their
    # questions to the cache by themselves, so the game never has to wait for one it doesn't need yet.
    pending: dict[int, asyncio.Task[dict[int, str]]] = {}

    def start_questions(nodes: list[int]) -> None:
        nodes = [
            node for node in nodes
            if not is_leaf(dtree, node) and node not in question_cache and node not in pending
        ]
        if not nodes:
            return
        task = asyncio.create_task(request_question_bundle(llm, model, dtree, features, nodes))
        task.add_done_callback(
            lambda done: question_cache.update(done.result())
            if not done.cancelled() and done.exception() is None
            else None
        )
        for node in nodes:
            pending[node] = task

    try:
        # Generate the first questions while the user is picking a work
        start_questions([0, *children(dtree, 0)])
        ready = await ask(robot, "Think of a literary work, and tell me when you're ready.")
        if ready:
            print(f"You: {ready}")
            if is_quit(ready):
                raise QuitGame
        await speak(robot, "Great, let's begin!")

        # (question, answered yes, tested tree feature or None), for the LLM fallback and the answer check
        history: list[tuple[str, bool, str | None]] = []
        rejected: set[str] = set()  # titles guessed wrong
        questions_asked = 0
        may_guess = True  # one guess per answered question, unless only one work is left
        node = 0

        while candidates := remaining_titles(dtree, node, rejected):
            leaf = is_leaf(dtree, node)

            # Skip questions whose answer we already know, e.g. after a wrong guess ruled out one side
            if not leaf:
                left, right = children(dtree, node)
                left_open = bool(remaining_titles(dtree, left, rejected))
                right_open = bool(remaining_titles(dtree, right, rejected))
                if not (left_open and right_open):
                    node = left if left_open else right
                    continue

            # Guess once only a few works fit, instead of walking the whole tree
            if leaf or len(candidates) == 1 or (len(candidates) <= GUESS_THRESHOLD and may_guess):
                title = random.choice(candidates)
                guess = guess_phrase(qg.describe_work(title), len(candidates), bool(rejected))
                print(f"Guessing {title} out of {candidates}")
                if await ask_until_answered(robot, guess):
                    plural = "" if questions_asked == 1 else "s"
                    await speak(robot, f"Yes! I got it in {questions_asked} question{plural}.")
                    return True
                rejected.add(title)
                may_guess = False
                if len(candidates) > 2:
                    await speak(robot, "Hmm, then let me ask a bit more.")
                continue

            # Ask the question for this node; normally prefetched. If the LLM is slow, the robot says
            # "let me think", and after QUESTION_WAIT seconds it uses the template question instead.
            if node not in question_cache:
                start_questions([node])
                question_cache.update(
                    await wait_for_llm(robot, asyncio.shield(pending[node]), QUESTION_WAIT, {})
                )
            # Start generating the next questions before speaking, so the LLM works while the robot talks
            start_questions(list(children(dtree, node)))
            split = qg.describe_split(dtree, features, node)
            question = question_cache.get(node) or qg.fallback_question(split)
            print(f"Node {node} question: {question} ({len(candidates)} works left)")

            answer_is_yes = await ask_until_answered(robot, question, split, features)
            questions_asked += 1
            history.append((question, answer_is_yes, split["feature"]))
            may_guess = True
            node = left if answer_is_yes == split["yes_goes_left"] else right

        # No work in the data fits the answers: an answer was wrong, or the work is not in the data
        await guess_with_llm(robot, llm, model, history, rejected)
        return True
    except QuitGame as stop:
        await speak(robot, str(stop) or "Okay, let's stop here.")
        return False
    finally:
        tasks = set(pending.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def guess_phrase(description: str, candidates: int, guessed_wrong_before: bool) -> str:
    """
    How to word a guess, depending on how sure the robot is: only sound confident when a single work
    still fits the answers; with two or three left it is really picking one of them.
    """
    if candidates == 1:
        if guessed_wrong_before:
            return f"Then it must be {description}. Am I right?"
        return f"I think I know it! Is it {description}?"
    if guessed_wrong_before:
        return random.choice([
            f"Hmm, then is it {description}?",
            f"Okay, then maybe... is it {description}?",
        ])
    if candidates == 2:
        return random.choice([
            f"I'm not completely sure yet, but is it {description}?",
            f"I think it's one of two works. Is it {description}?",
        ])
    return random.choice([
        f"Let me take a guess. Is it {description}?",
        f"I have a few ideas. Could it be {description}?",
    ])


async def ask_until_answered(
    robot: FurhatRobot,
    question: str,
    split: dict | None = None,
    features: list[str] | None = None,
) -> bool:
    """
    Ask until the user answers yes or no (or names the category, when a split is given).
    Raises QuitGame when the user wants to stop, or when nobody seems to be there any more.
    """
    silent = unclear = 0
    while True:
        user_text = await ask(robot, question)
        if not user_text:
            silent += 1
            if silent >= MAX_SILENT_TRIES:
                # Without this the robot would keep repeating the question to an empty room
                raise QuitGame("I can't hear you any more, so let's stop for now.")
            if user_text is None:
                await speak(robot, "I didn't hear anything. Let me ask again.")
            else:
                await speak(robot, "Sorry, I didn't catch that. Please try again.")
            continue
        silent = 0
        print(f"You: {user_text}")
        if is_quit(user_text):
            raise QuitGame

        if split is not None and features is not None:
            answer = interpret_answer(user_text, split, features)
        else:
            answer = parse_yes_no(user_text)
        if answer is not None:
            return answer
        unclear += 1
        if unclear >= MAX_UNCLEAR_TRIES:
            raise QuitGame("We seem to be stuck on this one, so let's stop for now.")
        if unclear == 1:
            await speak(robot, "Please answer yes or no.")
        else:
            await speak(robot, "If you're not sure, just give your best guess. Is it yes or no?")


async def guess_with_llm(
    robot: FurhatRobot,
    llm,
    model: str,
    history: list[tuple[str, bool, str | None]],
    rejected: set[str],
) -> bool:
    """
    The decision tree has no works left: the LLM asks a few extra questions and then makes one
    final guess. Returns whether it guessed right.
    """
    if furhat_llm_demo.llm_unavailable:
        # Without the LLM there are no extra questions or guesses, so don't promise them
        await speak(
            robot,
            "Hmm, none of the works I know fit your answers. Maybe one of the answers didn't match "
            "my notes, or it's a work I don't know yet.",
        )
        return await give_up(robot, llm, model, history, rejected)

    # Think of the first extra question while the robot is talking
    first_question = asyncio.create_task(request_llm_question(llm, model, history, rejected))
    try:
        await speak(
            robot,
            "Hmm, none of the works I know fit your answers. Maybe one of the answers didn't match "
            "my notes, or it's a work I don't know yet. Let me ask a few more questions.",
        )
        for number in range(EXTRA_LLM_QUESTIONS):
            if number == 0:
                question = await wait_for_llm(robot, first_question, LLM_FALLBACK_WAIT, None)
            else:
                question = await wait_for_llm(
                    robot, request_llm_question(llm, model, history, rejected), LLM_FALLBACK_WAIT, None
                )
            if question is None:
                # The LLM still failed after retrying; say so instead of silently skipping the questions
                await speak(robot, "Sorry, I'm having trouble thinking of a good question.")
                break
            print(f"LLM question: {question}")
            history.append((question, await ask_until_answered(robot, question), None))
    finally:
        if not first_question.done():
            first_question.cancel()
            await asyncio.gather(first_question, return_exceptions=True)

    # One final guess; ask the LLM again once if it repeats a work that was already guessed wrong
    guess = None
    for _ in range(2):
        guess = await wait_for_llm(
            robot, request_llm_guess(llm, model, history, rejected), LLM_FALLBACK_WAIT, None
        )
        if guess is None or guess[0].lower() not in {name.lower() for name in rejected}:
            break
        print(f"LLM repeated a wrong guess: {guess[0]}")
        guess = None

    if guess is not None:
        title, author = guess
        description = f"{title} by {author}" if author else title
        if await ask_until_answered(robot, f"Okay, my final guess. Is it {description}?"):
            await speak(robot, "Yes! I got there in the end.")
            await review_work(robot, llm, model, description, history, rejected, confirmed=True)
            return True
        rejected.add(title)

    return await give_up(robot, llm, model, history, rejected)


async def give_up(
    robot: FurhatRobot,
    llm,
    model: str,
    history: list[tuple[str, bool, str | None]],
    rejected: set[str],
) -> bool:
    """Ask the player which work it was, then check the answers and learn the work. Always returns False."""
    user_text = await ask(robot, "I give up! What were you thinking of?")
    if not user_text:
        await speak(robot, "That's okay, that one was a real challenge.")
        return False
    print(f"You: {user_text}")
    if is_quit(user_text):
        raise QuitGame
    await review_work(robot, llm, model, user_text, history, rejected, confirmed=False)
    return False


async def review_work(
    robot: FurhatRobot,
    llm,
    model: str,
    said: str,
    history: list[tuple[str, bool, str | None]],
    rejected: set[str],
    confirmed: bool,
) -> None:
    """
    Now that the player told (or confirmed) the work: point out the answers that don't match it,
    and save it for next time if the robot didn't know it yet.
    `said` is what the player said, or the title and author of a guess the player confirmed.
    """
    title = qg.find_title(said)
    details = None
    if title is None:
        details = await wait_for_llm(robot, request_work_details(llm, model, said), LLM_FALLBACK_WAIT, None)
        if details is None:
            await speak(robot, "Thanks for telling me! I don't know that one, so I'll have to read up on it.")
            return
        # The LLM may recognise a known work that speech recognition misheard
        title = qg.find_title(details["Title"])

    is_new = title is None
    if is_new:
        title = details["Title"]
        work = qg.prepare_work(details)
        author = details.get("Author")
        description = f"{title} by {author}" if author else title
    else:
        work = qg.work_row(title)
        description = qg.describe_work(title)

    if not confirmed and not await ask_until_answered(robot, f"Do you mean {description}?"):
        await speak(robot, "Hmm, then I don't know that one. Thanks for telling me anyway!")
        return

    guessed_earlier = title.lower() in {name.lower() for name in rejected}
    message = mismatch_message(title, find_mismatches(work, history), guessed_earlier, is_new)
    if message:
        await speak(robot, message)
    elif not is_new:
        await speak(robot, f"Ah, {title}! Your answers match my notes, so I'm not sure where I went wrong.")

    if is_new:
        qg.save_work(details)
        print(f"Saved new work to {qg.LEARNED_FILE.name}: {details}")
        await speak(robot, f"I didn't know {title} yet, so I've saved it. Next time, I might guess it!")


def find_mismatches(
    work,
    history: list[tuple[str, bool, str | None]],
) -> list[tuple[str, bool, bool | None]]:
    """
    The tree questions where the player's answer and the data disagree, as (question, answer, noted).
    noted is None when the data has no value, which the tree reads as "no". Neither side is assumed
    to be right: the data can contain mistakes too. The extra LLM questions have no data to compare with.
    """
    mismatches = []
    for question, answer, feature in history:
        if feature is None:
            continue
        noted = qg.work_has_feature(work, feature)
        # An unknown value only sent the tree the wrong way if the player said yes
        if (noted is None and answer) or (noted is not None and noted != answer):
            mismatches.append((question, answer, noted))
    return mismatches


def mismatch_message(
    title: str,
    mismatches: list[tuple[str, bool, bool | None]],
    guessed_earlier: bool,
    is_new: bool,
    limit: int = 3,
) -> str | None:
    """
    Tell the player where their answers and the robot's notes disagree, mentioning at most `limit`
    of them. Worded as a disagreement to look up, not as the player being wrong. None if nothing to say.
    """
    notes = "what I know about it" if is_new else "my notes"
    notes_say = "I thought" if is_new else "my notes say"
    notes_unknown = "I wasn't sure about that" if is_new else "my notes don't mention that"
    parts = []
    for question, answer, noted in mismatches[:limit]:
        said = "yes" if answer else "no"
        if noted is None:
            parts.append(f'when I asked "{question}", you said {said}, but {notes_unknown}')
        else:
            parts.append(f'when I asked "{question}", you said {said}, while {notes_say} {"yes" if noted else "no"}')
    if len(mismatches) > limit:
        parts.append(f"and {len(mismatches) - limit} more")

    sentences = []
    if parts:
        count = {1: "one question", 2: "two questions", 3: "three questions"}
        which = count.get(len(mismatches), "a few questions")
        sentences.append(f"On {which}, your answer and {notes} don't agree: {'; '.join(parts)}.")
        if is_new:
            sentences.append("I could easily be wrong there, so it might be worth looking it up.")
        else:
            sentences.append("My notes aren't always right, so it might be worth looking up which of us is right.")
    if guessed_earlier:
        sentences.append("I did guess it earlier, so maybe one of us misheard the other.")
    if not sentences:
        return None
    return f"Ah, {title}! " + " ".join(sentences)


async def request_work_details(llm, model: str, said: str) -> dict | None:
    """
    Ask the LLM which work the player meant and fill in its data columns, so it can be checked and saved.
    Returns a dict with the file columns (Title, Author, Year, ...), or None if the LLM doesn't know it.
    """
    column_rules = []
    for column in qg.RAW_COLUMNS:
        if column == "Title":
            column_rules.append('- "Title": the usual title of the work')
        elif column == "Author":
            column_rules.append('- "Author": the full name of the author')
        elif column == "Year":
            column_rules.append('- "Year": the year it was first published, as a number')
        elif column in qg.MULTI_VALUE_COLUMNS:
            column_rules.append(
                f'- "{column}": 2 to 4 items separated by "; ". Reuse these where they fit: '
                f"{', '.join(qg.column_values(column))}"
            )
        else:
            column_rules.append(
                f'- "{column}": one of: {", ".join(qg.column_values(column))}. '
                "Use a new short value only if none of these fit, or null if it does not apply."
            )
    known_titles = "\n".join(f"- {title}" for title in qg.data["Title"])
    prompt = (
        "In a guessing game, the player was asked which literary work they were thinking of. "
        f'They said (through speech recognition, so it may contain errors): "{said}"\n\n'
        "If they mean one of these known works, use its exact title:\n"
        f"{known_titles}\n\n"
        "Return only a JSON object describing the work, with exactly these keys:\n"
        + "\n".join(column_rules)
        + '\n\nIf you cannot tell which work they mean, return {"Title": null}.'
    )
    try:
        response = await llm_respond(
            llm, model, prompt, max_tokens=600, response_format={"type": "json_object"}
        )
        decoded = json.loads(extract_json_object(response))
    except Exception as error:
        print(f"Looking up the work failed: {error}")
        return None
    if not isinstance(decoded, dict):
        return None
    title = decoded.get("Title")
    if not isinstance(title, str) or not title.strip():
        return None

    details = {}
    for column in qg.RAW_COLUMNS:
        value = decoded.get(column)
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value)
        if isinstance(value, str):
            value = value.strip() or None
        details[column] = value
    return details


def llm_game_context(history: list[tuple[str, bool, str | None]], rejected: set[str]) -> str:
    """What the LLM needs to know about the round so far, shared by the question and guess prompts."""
    answers = "\n".join(
        f"- {question} {'Yes' if answer else 'No'}" for question, answer, _ in history
    ) or "- (no questions asked)"
    known_works = "\n".join(f"- {qg.describe_work(title)}" for title in qg.data["Title"])
    wrong_guesses = ", ".join(sorted(rejected)) or "none"
    return (
        "We are playing a guessing game. The player is thinking of a literary work "
        "(a poem, short story, essay, speech, novel or similar). "
        f"These are the yes/no questions I asked and the player's answers:\n{answers}\n\n"
        f"It is NOT one of these, I already guessed them wrong: {wrong_guesses}.\n\n"
        "It may be one of the works below if the player gave a wrong answer to one or two questions, "
        "or it may be a different well-known work that is not on this list:\n"
        f"{known_works}\n\n"
    )


async def request_llm_question(
    llm,
    model: str,
    history: list[tuple[str, bool, str | None]],
    rejected: set[str],
) -> str | None:
    """Ask the LLM for one extra yes/no question. Returns None if the LLM fails."""
    prompt = llm_game_context(history, rejected) + (
        "Ask one short, natural yes/no question that best helps to find out which work it is. "
        "Do not repeat a question that was already asked and do not name a specific title. "
        "Return only the question."
    )
    try:
        response = await llm_respond(llm, model, prompt)
    except Exception as error:
        print(f"LLM question failed: {error}")
        return None
    # Some models add a preamble; keep the first line that is a question
    for line in response.splitlines():
        line = line.strip().strip('"').strip()
        if "?" in line:
            return line[: line.index("?") + 1]
    return None


async def request_llm_guess(
    llm,
    model: str,
    history: list[tuple[str, bool, str | None]],
    rejected: set[str],
) -> tuple[str, str | None] | None:
    """Ask the LLM for a title and author. Returns None if the LLM fails."""
    prompt = llm_game_context(history, rejected) + (
        'Return only a JSON object with your single best guess: {"title": "...", "author": "..."}'
    )
    try:
        response = await llm_respond(
            llm, model, prompt, max_tokens=256, response_format={"type": "json_object"}
        )
        decoded = json.loads(extract_json_object(response))
    except Exception as error:
        print(f"LLM guess failed: {error}")
        return None

    title = decoded.get("title") if isinstance(decoded, dict) else None
    if not isinstance(title, str) or not title.strip():
        return None
    title = title.strip()
    author = decoded.get("author")
    return title, author.strip() if isinstance(author, str) and author.strip() else None


def children(dtree: DecisionTreeClassifier, node: int) -> tuple[int, int]:
    return int(dtree.tree_.children_left[node]), int(dtree.tree_.children_right[node])


def is_leaf(dtree: DecisionTreeClassifier, node: int) -> bool:
    return dtree.tree_.feature[node] < 0


def remaining_titles(dtree: DecisionTreeClassifier, node: int, rejected: set[str]) -> list[str]:
    """The titles under a tree node that have not been guessed wrong yet."""
    weights = dtree.tree_.value[node][0]
    return [
        str(title)
        for title, weight in zip(dtree.classes_, weights)
        if weight > 0 and title not in rejected
    ]


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


def interpret_answer(user_text: str, split: dict, features: list[str]) -> bool | None:
    """Parse a yes/no answer, also accepting a category name, e.g. "Male" to "Is the author male?"."""
    answer = parse_yes_no(user_text)
    # A work has several symbols/themes, so naming another one does not mean "no": only accept yes/no there
    if answer is not None or split["category"] is None or split["multi"]:
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
