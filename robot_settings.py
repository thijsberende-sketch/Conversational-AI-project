"""
Initial robot settings, grouped like the tabs in the Furhat web interface
(Face, Speaking, Listening, LED).

None means: leave the robot's own setting as it was.
"""

# Face
# Not currntly in use; the robot keeps its current face.
FACE_ID = None                # mask/character, as listed by request_face_status()
FACE_VISIBLE = None           # True / False
FACE_MICROEXPRESSIONS = None  # True / False

# Speaking 
# Applied on connect, so the robot always uses this voice instead of whatever was last selected on it.
VOICE_NAME = None
VOICE_LANGUAGE = "en-GB"
VOICE_GENDER = None
VOICE_PROVIDER = None

# Listening
LISTEN_LANGUAGES = ["en-US"] # currently input is American, but output is British
END_SPEECH_TIMEOUT = 4.0      # seconds of silence before the user's answer counts as finished; long so the player can pause mid-question ("is the author a... woman"). In game 2 a finished question ends sooner, see QUESTION_END_SILENCE
QUESTION_END_SILENCE = 1.0    # game 2: once what the user has said so far is a finished question, answer after this many seconds without new words (instead of END_SPEECH_TIMEOUT); 0.5 cut people off mid-sentence
FILLER_PAUSE = (0.3, 0.5)     # game 2: random pause (min, max seconds) between a thinking filler ("Hmm.") and the LLM's answer
NO_QUESTION_TIMEOUT = 120.0   # game 2 fail-safe: if no question is heard for this many seconds, Furhat prompts the player again
STOP_NO_SPEECH = False       # False: keep listening until the user speaks. True (Furhat default): the robot silently stops listening after ~8 s without speech

# LED
# Currently just set to green, sets this at default on connect, but the robot can change it during the conversation.
LED_COLOR = "#228B22"              # e.g. "#00FF00"
