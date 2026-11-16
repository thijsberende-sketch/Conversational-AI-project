import asyncio
import os
import re
import sys

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from furhat_robot import FurhatRobot

import game_2 as g2
import robot_settings

QUIT_WORDS = {"quit", "exit", "goodbye", "stop"}
GIVE_UP_PHRASES = ("i give up", "give up", "tell me the answer", "i surrender")
MAX_QUESTIONS = 20 # We need to limit the number of questions, let's discuss this. 20 is a lot, but it allows for some mistakes and clarifications.

LLM_PROVIDER = "openai"

ANSWER_REPLIES = {
    "yes": "Yes.",
    "no": "No.",
    "unknown": "Sorry, I'm not sure. Could you ask that another way?",
}


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


class ConsoleRobot:
    """Stand-in for FurhatRobot when playing in the terminal: speaking prints, listening reads a typed line."""

    on_final_text = None

    def connect(self): pass
    def disconnect(self): pass
    def attend_user(self): pass
    async def connect_realtime(self): pass
    async def disconnect_realtime(self): pass
    async def stop_listen(self): pass

    def speak(self, text):
        print(f"Furhat: {text}")

    async def start_listen(self, **_):
        try:
            text = await asyncio.to_thread(input, "> ")
        except EOFError:  # Ctrl+Z / Ctrl+D ends the game
            text = "quit"
        self.on_final_text(text)


async def main(text_mode: bool = False) -> None:
    load_dotenv()

    if not text_mode and not os.getenv("FURHAT_HOST"):
        raise ValueError("Set FURHAT_HOST to the IP address of your Furhat (or localhost for the virtual one).")

    llm, model = create_llm()
    print(f"Using {LLM_PROVIDER} model {model}")
    robot = ConsoleRobot() if text_mode else FurhatRobot()

    robot.connect()

    # run the game in a loop, asking if the user wants to play again after each round
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
    """Play one round where Furhat has a secret book. Returns False when the user asked to quit."""
    # A new random book every round (this replaces the decision tree of game 1)
    book = g2.select_secret_book()
    title = book[g2.TARGET]
    questions_asked = 0

    next_prompt = (
        "I'm thinking of a book. Ask me yes or no questions about it, and when you think you know it, "
        f"say the title. You have {MAX_QUESTIONS} questions. Go ahead!"
    )

    while True:
        # Speak what we have to say (intro or the answer to the previous question) and listen for the next question
        user_text = await ask(robot, next_prompt)
        if user_text is None:
            next_prompt = "I didn't hear anything. Ask me a question about the book."
            continue
        if not user_text:
            next_prompt = "Sorry, I didn't catch that. Please ask again."
            continue
        print(f"You: {user_text}")

        if is_quit(user_text):
            await speak(robot, "Goodbye!")
            return False
        if is_give_up(user_text):
            await speak(robot, f"The book was {title}. Better luck next time!")
            return True

        # This is a bit problematic right now: if the user asks a question that is not yes/no, we still count it as a question.
        questions_asked += 1

        # A correct guess ends the round
        if g2.is_correct_guess(user_text, book):
            await speak(robot, f"Yes! It was {title}. You got it in {questions_asked} questions!")
            return True

        # A guess of a different book from the data is answered directly, without asking the LLM
        wrong_title = other_title_mentioned(user_text, title)
        if wrong_title is not None:
            answer_text = f"No, it's not {wrong_title}."
        else:
            answer = await answer_question(llm, model, book, user_text)
            answer_text = ANSWER_REPLIES[answer]

        if questions_asked >= MAX_QUESTIONS:
            await speak(robot, f"{answer_text} That was your last question. The book was {title}.")
            return True

        remaining = MAX_QUESTIONS - questions_asked
        next_prompt = answer_text
        if remaining <= 3:
            next_prompt += f" You have {remaining} question{'s' if remaining != 1 else ''} left."


def other_title_mentioned(user_text: str, secret_title: str) -> str | None:
    """Return a different title from the data that the user said, so a wrong guess gets a clear 'no'."""
    for other_title in g2.data[g2.TARGET].dropna().unique():
        if other_title != secret_title and g2.mentions_title(user_text, other_title):
            return other_title
    return None


async def answer_question(llm, model: str, book, question: str) -> str:
    """Ask the LLM to answer a player's question from the book's facts. Falls back to keyword matching on failure."""
    try:
        reply = await llm_respond(llm, model, g2.answer_prompt(book, question), max_tokens=16)
        return g2.parse_answer(reply)
    except Exception as error:
        print(f"LLM failed, using fallback answer: {error}")
        return g2.fallback_answer(book, question)


async def speak(robot: FurhatRobot, text: str) -> None:
    # robot.speak blocks until speech is done; run it in a thread so the realtime event loop keeps running
    await asyncio.to_thread(robot.speak, text)


async def ask(
    robot: FurhatRobot,
    question: str,
    time_to_wait: float = robot_settings.END_SPEECH_TIMEOUT,
) -> str | None:
    """Speak a line and return the user's reply, or None if nothing was heard."""
    # Create a future to hold the user's answer
    answer_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    # callback function that handles eventual response from the user and sets the result of the future
    def on_final_text(text: str) -> None:
        if not answer_future.done():
            answer_future.set_result(text.strip())

    robot.on_final_text = on_final_text
    await speak(robot, question)

    await robot.start_listen(languages=robot_settings.LISTEN_LANGUAGES, end_speech_timeout=time_to_wait)
    try:
        return await asyncio.wait_for(answer_future, timeout=60)
    except asyncio.TimeoutError:
        return None
    finally:
        await robot.stop_listen()


async def ask_yes_no(robot: FurhatRobot, question: str, attempts: int = 3) -> bool | None:
    """Ask a yes/no question, re-asking on unclear answers. Quitting counts as no."""
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

    print(f"LLM answer: {reply}")
    return reply


def normalize(user_text: str) -> str:
    normalized_text = user_text.lower().replace("’", "'").replace("-", " ")
    answer = re.sub(r"[^\w\s']", "", normalized_text).strip()
    return re.sub(r"\s+", " ", answer)


def is_quit(user_text: str) -> bool:
    return normalize(user_text) in QUIT_WORDS


def is_give_up(user_text: str) -> bool:
    answer = normalize(user_text)
    return any(phrase in answer for phrase in GIVE_UP_PHRASES)


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
        "also yes",
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
        "also no",
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


if __name__ == "__main__":
    """
        Run the Furhat game 2 demo, connecting to the robot and the LLM provider specified in .env.
        It runs asynchronously, so the main function is wrapped in asyncio.run() to start the event loop.
        Run with --text to play in the terminal without a robot: python furhat_game2_demo.py --text
    """
    asyncio.run(main(text_mode="--text" in sys.argv))