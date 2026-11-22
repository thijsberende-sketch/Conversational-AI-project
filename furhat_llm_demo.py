"""
General functions shared by all game modes: connecting to the robot and the LLM,
speaking, listening and parsing answers. The game modes live in their own files
(game_robot_asks.py, ...) and are started from main().
"""

import asyncio
import os
import re

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from furhat_robot import FurhatRobot

import robot_settings

QUIT_WORDS = {"quit", "exit", "goodbye", "stop"}

# OpenRouter models that refused a request with reasoning switched off, so later calls skip that attempt
REASONING_REQUIRED: set[str] = set()

# Set to the reason when the LLM can't be used for the rest of the session (daily free limit reached)
llm_unavailable: str | None = None


def create_llm() -> tuple[OpenAI, str]:
    """Create the LLM client for the provider chosen with LLM_PROVIDER in .env."""
    # Choosing whether we use the class openai API or the free openrouter API, based on the .env file
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your .env file, or set LLM_PROVIDER=openrouter.")
        llm = OpenAI(api_key=api_key, max_retries=0, timeout=30.0)
        return llm, os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY in your .env file, or set LLM_PROVIDER=openai.")
        llm = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            timeout=30.0,
        )
        # A comma-separated list means: use the first model, and fall back to the next ones when it is
        # rate limited or down (free models often are)
        model = ",".join(
            name.strip() for name in os.getenv("OPENROUTER_MODEL", "openrouter/free").split(",") if name.strip()
        )
        if any(name != "openrouter/free" and not name.endswith(":free") for name in model.split(",")):
            raise ValueError(
                "To keep requests free, OPENROUTER_MODEL must be 'openrouter/free' "
                "or model IDs ending with ':free'."
            )
        return llm, model

    raise ValueError(f"LLM_PROVIDER must be 'openai' or 'openrouter', not '{provider}'.")


def llm_provider(llm: OpenAI) -> str:
    """Which provider a client from create_llm() talks to: "openai" or "openrouter"."""
    return "openrouter" if "openrouter.ai" in str(llm.base_url) else "openai"


async def main() -> None:
    # Game modes import the general functions from this file, so they are imported here instead of at the top
    import game_robot_asks as game

    load_dotenv()

    furhat_host = os.getenv("FURHAT_HOST")
    if not furhat_host:
        raise ValueError("Set FURHAT_HOST to the IP address of your Furhat.")

    llm, model = create_llm()
    print(f"Using {llm_provider(llm)} model {model}")
    robot = FurhatRobot()

    robot.connect()
    
    # basic logic to run the game in a loop, asking if the user wants to play again after each round
    try:
        await robot.connect_realtime()
        robot.attend_user()

        await game.welcome(robot)
        while await game.play_round(robot, llm, model):
            if await ask_yes_no(robot, "Do you want to play again?") is not True:
                break
        await game.farewell(robot)
    finally:
        await robot.stop_listen()
        await robot.disconnect_realtime()
        robot.disconnect()


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

    global llm_unavailable
    if llm_unavailable:
        # Fail at once, so the game uses its template questions instead of waiting on requests that will fail
        raise RuntimeError(llm_unavailable)

    provider = llm_provider(llm)
    token_budget = max_tokens
    # Free OpenRouter models are reasoning models: thinking first takes 20+ seconds and can use up the
    # whole token budget, so it is switched off. Some models refuse that; those get low effort instead.
    reasoning = {"effort": "low"} if model in REASONING_REQUIRED else {"enabled": False}
    empty_retried = False
    # Few attempts on purpose: on the free tier every attempt, also a failed one, counts towards the daily limit
    attempts = 3
    for attempt in range(attempts):
        last_attempt = attempt == attempts - 1
        try:
            request_options = {"max_tokens": token_budget}
            if response_format is not None:
                request_options["response_format"] = response_format
            # OpenRouter-only options; OpenAI rejects unknown request fields
            models = model.split(",")
            if provider == "openrouter":
                extra_body = {"reasoning": reasoning}
                if model != "openrouter/free":
                    extra_body["provider"] = {"sort": "throughput"}
                if len(models) > 1:
                    # OpenRouter tries these in order when a model is rate limited or unavailable
                    extra_body["models"] = models
                request_options["extra_body"] = extra_body

            completion = await asyncio.to_thread(
                llm.chat.completions.create,
                model=models[0],
                messages=messages,
                **request_options,
            )
            if provider == "openrouter" and completion.model != models[0]:
                print(f"OpenRouter selected model: {completion.model}")

            # A reasoning model can spend the whole budget thinking and return nothing: retry once with more room
            reply = completion.choices[0].message.content if completion.choices else None
            if not reply:
                if empty_retried or last_attempt:
                    raise RuntimeError("The LLM returned an empty reply.")
                empty_retried = True
                token_budget = min(token_budget * 4, 2048)
                print(f"{provider} returned an empty reply; retrying once with max_tokens={token_budget}.")
                continue
            break
        except APIStatusError as error:
            message_text = str(error).lower()
            if error.status_code == 429 and "per-day" in message_text:
                # The daily free limit is used up; every further request would fail too
                llm_unavailable = (
                    "The daily limit of free OpenRouter requests is used up, so the LLM is off for this session. "
                    "It resets at midnight UTC; adding 10 credits on openrouter.ai raises it to 1000 per day."
                )
                print(llm_unavailable)
                raise RuntimeError(llm_unavailable) from error
            if (
                error.status_code == 400
                and "reasoning" in message_text
                and reasoning.get("enabled") is False
                and not last_attempt
            ):
                print(f"{model} cannot switch reasoning off; using low reasoning effort instead.")
                REASONING_REQUIRED.add(model)
                reasoning = {"effort": "low"}
                continue
            if error.status_code not in {408, 429, 500, 502, 503, 504} or last_attempt:
                raise
            delay = 0.8 * (2**attempt)
            print(
                f"{provider} returned HTTP {error.status_code}; "
                f"retrying in {delay:.1f}s (attempt {attempt + 2}/{attempts})."
            )
            await asyncio.sleep(delay)
        except APIConnectionError:
            if last_attempt:
                raise
            delay = 0.8 * (2**attempt)
            print(
                f"{provider} connection failed; retrying in {delay:.1f}s "
                f"(attempt {attempt + 2}/{attempts})."
            )
            await asyncio.sleep(delay)

    print(f"Furhat: {reply}")
    return reply


def extract_json_object(text: str) -> str:
    """Strip markdown fences or surrounding text that some models add because they ignore the prompt instructions."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return text
    return text[start:end + 1]


def mentions(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase.lower())}\b", text.lower()) is not None


def normalize(user_text: str) -> str:
    normalized_text = user_text.lower().replace("’", "'").replace("-", " ")
    answer = re.sub(r"[^\w\s']", "", normalized_text).strip()
    return re.sub(r"\s+", " ", answer)


def is_quit(user_text: str) -> bool:
    """Check if the user wants to quit the game."""
    return normalize(user_text) in QUIT_WORDS


def parse_yes_no(user_text: str) -> bool | None:
    """To extract a yes/no without having to ask the llm"""
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


if __name__ == "__main__":
    """
        Run the Furhat LLM demo, connecting to the robot and the LLM provider specified in .env.
        It runs asynchronously for parallel execution, so the main function is wrapped in asyncio.run() to start the event loop.
    """
    asyncio.run(main())
