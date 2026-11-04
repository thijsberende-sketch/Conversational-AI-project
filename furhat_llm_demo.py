import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI

from furhat_robot import FurhatRobot


async def main() -> None:
    load_dotenv()

    furhat_host = os.getenv("FURHAT_HOST")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not furhat_host:
        raise ValueError("Set FURHAT_HOST to the IP address of your Furhat.")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in your .env file.")

    llm = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    robot = FurhatRobot()

    robot.connect()
    try:
        await robot.connect_realtime()
        robot.attend_user()

        history = [
            {
                "role": "system",
                "content": "You are a friendly, concise conversational assistant speaking with someone through a robot.",
            }
        ]

        while True:
            answer_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

            def on_final_text(text: str) -> None:
                if not answer_future.done():
                    answer_future.set_result(text.strip())

            robot.on_final_text = on_final_text
            robot.speak("What would you like to talk about?")
            await robot.start_listen(languages=["en-US"], end_speech_timeout=2.0)
            try:
                user_text = await asyncio.wait_for(answer_future, timeout=60)
            except asyncio.TimeoutError:
                await robot.stop_listen()
                robot.speak("I didn't hear anything. Let's try again.")
                continue
            await robot.stop_listen()

            if not user_text:
                robot.speak("Sorry, I didn't catch that. Please try again.")
                continue
            if user_text.lower() in {"quit", "exit", "goodbye"}:
                robot.speak("Goodbye!")
                break

            print(f"You: {user_text}")
            history.append({"role": "user", "content": user_text})
            completion = await asyncio.to_thread(
                llm.chat.completions.create,
                model=model,
                messages=history,
            )
            reply = completion.choices[0].message.content
            if not reply:
                raise RuntimeError("The LLM returned an empty reply.")

            print(f"Furhat: {reply}")
            history.append({"role": "assistant", "content": reply})
            robot.speak(reply)
    finally:
        await robot.stop_listen()
        await robot.disconnect_realtime()
        robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
