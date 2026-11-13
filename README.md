# Furhat Title-Guessing Game

The Furhat robot guesses which story you are thinking of by asking yes/no questions.

A decision tree is trained on the stories in `sample_data.xlsx`. Each split in the tree becomes a question, and an LLM (OpenAI's gpt-3.5-turbo by default, or a free OpenRouter model) turns that split into a natural-sounding binary question for that split. The tree is rebuilt every round, so the questions change from game to game, but every story can always be reached.

## Requirements

- Python 3.10 or newer
- A Furhat robot (or the Furhat virtual robot) with the Realtime API enabled, on the same network as your computer
- An OpenAI API key (the class key), or an [OpenRouter](https://openrouter.ai/keys) API key with `LLM_PROVIDER=openrouter` (free models are enough). See `.env.example` for all settings.


The robot sets its voice and LED, then asks questions until it can guess the story. Answer with "yes" or "no" (or the value itself, such as "male"). Say "quit", "stop" or "goodbye" to end the game.

## Configuration

Initial robot behavior is currently set in [robot_settings.py](robot_settings.py), grouped like the tabs in the Furhat web interface:

Face is currently not changed, it has the default mask and expressions
Speaking is set to British English with the Amy voice
Listening settings are also left as default
LED is set to forest green
`None` means the robot keeps its own setting.

To change the stories, edit `sample_data.xlsx`. The `Title` column is what the robot guesses; every other column except `Author` is used for questions. Maybe later we add author again, but I removed it as it was splitting on that which felt useless as it is a almost unique value.

## Project structure
`furhat_llm_demo.py` Main file, run this: general functions shared by the game modes (robot and LLM connection, asking, listening, answer parsing)
`game_robot_asks.py` Game mode 1: Furhat asks yes/no questions to guess your story (decision tree walk, LLM question generation)
`question_generator.py` Loads the data, builds the decision tree and describes its splits 
`furhat_robot.py` Wrapper around the Furhat client and the Realtime API (speech, listening, camera, audio), also has the functions to change things about the robot
`robot_settings.py` Face, voice, listening and LED settings
`sample_data.xlsx` The stories and their properties 
`retrieve_data.py` and `Saving_answers.py` Currently not used, was meant for when Furhat is the one answering questions
