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
VOICE_NAME = "Amy-generative"
VOICE_LANGUAGE = "en-GB"
VOICE_GENDER = None
VOICE_PROVIDER = None

# Listening
LISTEN_LANGUAGES = ["en-US"] # currently input is American, but output is British
END_SPEECH_TIMEOUT = 2.0      # seconds of silence before the user's answer counts as finished, this could be changed to make it more natural, but it is not a big problem for this demo

# LED
# Currently just set to green, sets this at default on connect, but the robot can change it during the conversation.
LED_COLOR = "#228B22"              # e.g. "#00FF00"
