import asyncio
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from furhat_robot import FurhatRobot
from session_recorder import SessionRecorder

import game_2 as g2
import question_matcher as qm
import robot_settings

QUIT_WORDS = {"quit", "exit", "goodbye", "stop"}
GIVE_UP_PHRASES = ("i give up", "give up", "tell me the answer", "i surrender")
# Used by is_question to decide whether the player's turn is over
QUESTION_WORDS = {
    "is", "are", "was", "were", "am", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "should", "shall", "may", "might", "must",
    "isn't", "aren't", "wasn't", "weren't", "doesn't", "didn't", "hasn't", "haven't", "can't", "won't",
    "don't", "couldn't", "wouldn't", "shouldn't",
    "who", "what", "when", "where", "which", "why", "how",
}
FILLER_WORDS = {"um", "uh", "erm", "er", "hmm", "mm", "so", "okay", "ok", "well", "and", "then", "right", "alright", "oh"}
QUESTION_PAIR = re.compile(
    r"\b(is|are|was|were|does|do|did|has|have|had|can|could|will|would)\s+(it|he|she|they|there|the|you)\b"
)
# A question ending on one of these words is probably not finished yet ("is it a", "was it written by")
UNFINISHED_ENDINGS = QUESTION_WORDS | FILLER_WORDS | {
    "a", "an", "the", "of", "by", "in", "on", "at", "to", "for", "with", "about", "from", "or", "but", "than",
    "it", "its", "it's", "this", "that", "there", "he", "she", "they", "his", "her", "their", "any", "some",
}
# A question that so far is only "is the author" / "does the book" / "is the main character" is not finished:
# people often pause there ("is the... author... male"). Hard-coded because it is by far the most common case.
UNFINISHED_QUESTION = re.compile(
    r"\b(is|are|was|were|does|do|did|has|have|had|can|could|will|would) the (main character|\w+)$"
)
MAX_QUESTIONS = 20 # We need to limit the number of questions, let's discuss this. 20 is a lot, but it allows for some mistakes and clarifications.

# Furhat picks one of these at random, so the answers don't sound the same every time.
# Keep them general: they have to fit any yes/no question.
ANSWER_REPLIES = {
    "yes": ["Yes.", "Yep.", "Yes, that's right.", "That's right.", "Correct.", "Yes, indeed."],
    "no": ["No.", "Nope.", "No, sorry.", "I'm afraid not.", "No, I'm afraid not."],
    "unknown": [
        "Sorry, I'm not sure. Could you ask that another way?",
        "Hmm, I can't really answer that. Could you rephrase it?",
    ],
}
# Said while the LLM works on the answer (not for answers from the table, which are instant)
THINKING_FILLERS = [
    "Hmm.",
    "Let me think.",
    "Hmm, let me think.",
    "Let me see.",
    "Good question.",
    "Ooh, interesting.",
    "Let me check.",
    "Hmm, one moment.",
    "Let me think about that.",
    "Interesting question.",
]

_last_picked: dict[tuple[str, ...], str] = {}

# The recording of this session (set in main when RECORD_SESSIONS is on), None when not recording
recorder: SessionRecorder | None = None


def record(event: str, **data) -> None:
    """Add a game event to the session log, if recording."""
    if recorder is not None:
        recorder.log(event, **data)


def pick(options: list[str]) -> str:
    """A random option, never the same one twice in a row."""
    key = tuple(options)
    choices = [option for option in options if option != _last_picked.get(key)] or options
    _last_picked[key] = random.choice(choices)
    return _last_picked[key]


def create_llm() -> tuple[OpenAI, str]:
    """Create the LLM client for the provider chosen with LLM_PROVIDER in .env."""
    global LLM_PROVIDER

    # Choosing whether we use the class openai API or the free openrouter API, based on the .env file
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your .env file, or set LLM_PROVIDER=openrouter.")
        llm = OpenAI(api_key=api_key, max_retries=0, timeout=5.0)
        return llm, os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    if LLM_PROVIDER == "openrouter":
        print("slect operouter")
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
    on_partial_text = None  # never called: typed input has no partial results

    on_event = None                   # no robot events or audio in text mode; the log still records the game
    on_mic_and_speaker_audio = None

    def connect(self): pass
    def disconnect(self): pass
    def attend_user(self): pass
    async def connect_realtime(self): pass
    async def disconnect_realtime(self): pass
    async def stop_listen(self): pass
    async def start_audio(self, **_): pass
    async def stop_audio(self): pass

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

    # Record the whole session (audio + event log); started after connecting, so a failed connection leaves no folder
    global recorder
    if robot_settings.RECORD_SESSIONS:
        recorder = SessionRecorder(
            Path(__file__).parent / robot_settings.RECORDINGS_DIR, robot_settings.RECORDING_SAMPLE_RATE
        )
        recorder.log("settings", provider=LLM_PROVIDER, model=model, text_mode=text_mode,
                     end_speech_timeout=robot_settings.END_SPEECH_TIMEOUT,
                     question_end_silence=robot_settings.QUESTION_END_SILENCE,
                     standard_reply_silence=robot_settings.STANDARD_REPLY_SILENCE)
        robot.on_event = recorder.robot_event
        robot.on_mic_and_speaker_audio = recorder.write_audio
        print(f"Recording this session to {recorder.folder}")

    # run the game in a loop, asking if the user wants to play again after each round
    try:
        await robot.connect_realtime()
        if recorder is not None and not text_mode:
            await robot.start_audio(sample_rate=robot_settings.RECORDING_SAMPLE_RATE, speaker=True)
        robot.attend_user()

        while await play_round(robot, llm, model):
            if await ask_yes_no(robot, "Do you want to play again?") is not True:
                await speak(robot, "Thanks for playing. Goodbye!")
                break
    finally:
        await robot.stop_audio()
        await robot.stop_listen()
        await robot.disconnect_realtime()
        robot.disconnect()
        # Saves the recording: also runs when the game crashes or is stopped with Ctrl+C
        if recorder is not None:
            recorder.close()
            print(f"Recording saved to {recorder.folder}")


async def play_round(robot: FurhatRobot, llm, model: str) -> bool:
    """Play one round where Furhat has a secret book. Returns False when the user asked to quit."""
    # A new random book every round (this replaces the decision tree of game 1)
    book = g2.select_secret_book()
    title = book[g2.TARGET]
    questions_asked = 0
    record("round_start", secret_book=title)

    next_prompt = (
        "I'm thinking of a book. Ask me yes or no questions about it, and when you think you know it, "
        f"say the title. You have {MAX_QUESTIONS} questions. Go ahead!"
    )

    while True:
        # Speak what we have to say (intro or the answer to the previous question) and listen for the next question
        # Turn-taking is based on content: Furhat keeps listening until the player asks a question,
        # guesses a title, quits or gives up. Anything else (thinking aloud, "hmm", noise) is ignored.
        # A finished question is answered after QUESTION_END_SILENCE, a standard reply (quit, give up, a title)
        # after STANDARD_REPLY_SILENCE; anything else gets the full END_SPEECH_TIMEOUT so the player can think.
        user_text = await ask(
            robot,
            next_prompt,
            ends_turn=ends_player_turn,
            timeout=robot_settings.NO_QUESTION_TIMEOUT,
            quick_end_after=game_quick_end_after,
        )
        # Fail-safe: no question for a long time (or the robot stopped sending results), so prompt again
        if user_text is None:
            record("no_question_timeout", seconds=robot_settings.NO_QUESTION_TIMEOUT)
            next_prompt = "I didn't hear a question. Ask me a yes or no question about the book."
            continue
        print(f"You: {user_text}")

        if is_quit(user_text):
            record("round_end", outcome="quit", questions=questions_asked, secret_book=title)
            await speak(robot, "Goodbye!")
            return False
        if is_give_up(user_text):
            record("round_end", outcome="gave_up", questions=questions_asked, secret_book=title)
            await speak(robot, f"The book was {title}. Better luck next time!")
            return True

        # This is a bit problematic right now: if the user asks a question that is not yes/no, we still count it as a question.
        questions_asked += 1

        # A correct guess ends the round
        if g2.is_correct_guess(user_text, book):
            record("round_end", outcome="guessed", questions=questions_asked, secret_book=title)
            await speak(robot, f"Yes! It was {title}. You got it in {questions_asked} questions!")
            return True

        # A guess of a different book from the data is answered directly, without asking the LLM
        wrong_title = other_title_mentioned(user_text, title)
        if wrong_title is not None:
            answer_text = f"No, it's not {wrong_title}."
            record("answer", question_number=questions_asked, source="wrong_title", answer="no")
        else:
            # Phase 1: answer from the table when possible (instant); phase 2: ask the LLM
            answer = qm.table_answer(book, user_text)
            if answer is not None:
                print(f"Answered from the table: {answer}")
                record("answer", question_number=questions_asked, source="table", answer=answer)
            else:
                # Start the LLM first and say a filler ("Hmm, let me think") while it works, so the
                # filler covers the waiting time instead of adding to it
                llm_task = asyncio.create_task(answer_question(llm, model, book, user_text))
                await speak(robot, pick(THINKING_FILLERS))
                # A short pause after the filler so "Hmm... Yes." doesn't come out as one phrase. It runs
                # alongside the LLM, so a slow LLM doesn't get this pause added on top.
                pause = random.uniform(*robot_settings.FILLER_PAUSE)
                answer, _ = await asyncio.gather(llm_task, asyncio.sleep(pause))
                record("answer", question_number=questions_asked, source="llm", answer=answer)
            answer_text = pick(ANSWER_REPLIES[answer])

        if questions_asked >= MAX_QUESTIONS:
            record("round_end", outcome="out_of_questions", questions=questions_asked, secret_book=title)
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
        # Only one word is needed, but reasoning models (which openrouter/free sometimes picks) spend tokens
        # thinking first; with 200 they could run out before answering and return nothing
        reply = await llm_respond(llm, model, g2.answer_prompt(book, question), max_tokens=500)
        return g2.parse_answer(reply)
    except Exception as error:
        print(f"LLM failed, using fallback answer: {error}")
        record("llm_failed", error=str(error))
        return g2.fallback_answer(book, question)


async def speak(robot: FurhatRobot, text: str) -> None:
    # robot.speak blocks until speech is done; run it in a thread so the realtime event loop keeps running
    if recorder is not None:
        recorder.furhat_says(text)
    await asyncio.to_thread(robot.speak, text)
    record("furhat_done_speaking")


async def ask(
    robot: FurhatRobot,
    question: str,
    time_to_wait: float = robot_settings.END_SPEECH_TIMEOUT,
    ends_turn: Callable[[str], bool] | None = None,
    timeout: float | None = 60,
    quick_end_after: Callable[[str], float | None] | None = None,
) -> str | None:
    """
    Speak a line and return the user's reply, or None if nothing was heard within `timeout` seconds
    (None = wait forever).

    With `ends_turn`, replies it rejects are ignored and Furhat keeps listening, so the turn only goes back
    to the robot once the user says something that ends their turn (in the game: a question or a guess).

    With `quick_end_after`, the partial text is checked while the user is still talking: when it returns a
    number of seconds, the reply counts as finished after that long without new words, instead of waiting
    the full end-of-speech silence. None means: not finished yet, keep waiting.
    """
    loop = asyncio.get_running_loop()
    # Everything the robot hears goes into this queue; a new queue per call, so old replies can't leak in
    utterances: asyncio.Queue[str] = asyncio.Queue()
    robot.on_final_text = lambda text: utterances.put_nowait(text.strip())

    # Partial results arrive every time the recognised text grows. Each one restarts a short timer; if the
    # text is a finished question and the timer runs out (no new words), we take the text as the reply.
    quick_timer: asyncio.TimerHandle | None = None

    def on_partial_text(text: str) -> None:
        nonlocal quick_timer
        if quick_timer is not None:
            quick_timer.cancel()
            quick_timer = None
        silence = quick_end_after(text) if quick_end_after is not None else None
        if silence is not None:
            quick_timer = loop.call_later(silence, end_turn_quickly, text.strip(), silence)

    def end_turn_quickly(text: str, silence: float) -> None:
        record("quick_end", text=text, silence=silence)
        utterances.put_nowait(text)

    robot.on_partial_text = on_partial_text
    await speak(robot, question)

    try:
        return await asyncio.wait_for(listen_until(robot, utterances, time_to_wait, ends_turn), timeout)
    except asyncio.TimeoutError:
        return None
    finally:
        robot.on_partial_text = None
        if quick_timer is not None:
            quick_timer.cancel()
        await robot.stop_listen()


async def listen_until(
    robot: FurhatRobot,
    utterances: asyncio.Queue[str],
    time_to_wait: float,
    ends_turn: Callable[[str], bool] | None,
) -> str:
    """Listen until the user says something that ends their turn, and return it."""
    while True:
        # Drop anything left over from an earlier listen, e.g. the final result the robot still sends after
        # we cut a question short, so the same question isn't answered twice
        while not utterances.empty():
            utterances.get_nowait()
        await robot.start_listen(
            languages=robot_settings.LISTEN_LANGUAGES,
            end_speech_timeout=time_to_wait,
            stop_no_speech=robot_settings.STOP_NO_SPEECH,
        )
        text = await utterances.get()
        if ends_turn is None or ends_turn(text):
            if recorder is not None:
                recorder.player_says(text, counted=True)
            return text
        if text:
            print(f"(Not a question, still listening: {text})")
            if recorder is not None:
                recorder.player_says(text, counted=False)
        # The robot stops listening after each utterance, so stop and start again for the next one
        await robot.stop_listen()


async def ask_yes_no(robot: FurhatRobot, question: str, attempts: int = 3) -> bool | None:
    """Ask a yes/no question, re-asking on unclear answers. Quitting counts as no."""
    for _ in range(attempts):
        # A clear yes, no or quit gets the quick reply; anything else waits the full END_SPEECH_TIMEOUT
        user_text = await ask(
            robot,
            question,
            quick_end_after=lambda text: robot_settings.STANDARD_REPLY_SILENCE if is_clear_yes_no(text) else None,
        )
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
    print(message)
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
            print(completion)
          
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


def is_question(user_text: str) -> bool:
    """
    Rough check whether the player asked a question. Speech recognition often leaves out the "?", so we
    also look at the words: a question word at the start ("Is it...", "Does the..."), possibly after
    fillers like "um" or "okay so", or an "is it" / "does the" pair anywhere ("let me think, is it a novel").
    """
    if "?" in user_text:
        return True
    words = words_without_leading_fillers(user_text)
    # "what's" / "who's" / "how's" count as "what" / "who" / "how"
    if words and (words[0] in QUESTION_WORDS or words[0].split("'")[0] in QUESTION_WORDS):
        return True
    return QUESTION_PAIR.search(" ".join(words)) is not None


def words_without_leading_fillers(user_text: str) -> list[str]:
    """The normalized words, without fillers at the start ("um okay so is it..." -> ["is", "it", ...])."""
    words = normalize(user_text).split()
    while words and words[0] in FILLER_WORDS:
        words.pop(0)
    return words


def mentions_any_title(user_text: str) -> bool:
    return any(g2.mentions_title(user_text, title) for title in g2.data[g2.TARGET].dropna().unique())


def ends_player_turn(user_text: str) -> bool:
    """The player's turn ends with a question, a title guess, quitting or giving up."""
    if not user_text:
        return False
    return is_question(user_text) or is_quit(user_text) or is_give_up(user_text) or mentions_any_title(user_text)


def is_clear_yes_no(user_text: str) -> bool:
    """True for a clear yes or no ("yes", "no thanks", "sure") or a quit word, used for the quick reply."""
    return is_quit(user_text) or parse_yes_no(user_text) is not None


def question_sounds_finished(user_text: str) -> bool:
    """
    True if the player asked a question that sounds complete. Used for the quick reply, so it must not fire
    on a question that stops on a word that needs more after it ("is it a", "was it written by"), or that is
    only "is the author" so far (UNFINISHED_QUESTION).
    """
    words = normalize(user_text).split()
    return (
        bool(words)
        and is_question(user_text)
        and words[-1] not in UNFINISHED_ENDINGS
        and UNFINISHED_QUESTION.search(" ".join(words)) is None
    )


def game_quick_end_after(user_text: str) -> float | None:
    """
    How long a silence ends the player's turn early, based on what they have said so far; None = keep waiting
    (the robot's END_SPEECH_TIMEOUT still applies). Standard replies are short and can't be cut off halfway,
    so they get the shorter pause.
    """
    if is_quit(user_text) or is_give_up(user_text) or mentions_any_title(user_text):
        return robot_settings.STANDARD_REPLY_SILENCE
    if question_sounds_finished(user_text):
        return robot_settings.QUESTION_END_SILENCE
    return None


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