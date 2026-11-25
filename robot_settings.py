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
# Name and language must belong to the same voice, otherwise the robot ignores the request
# (Danielle is en-US, Amy is en-GB).
# Use a Neural voice, not a "-generative" one: Amazon Polly's generative voices give no lip-sync data
# (visemes), so the mouth doesn't move.
VOICE_NAME = "Danielle-Neural"
VOICE_LANGUAGE = "en-US"
VOICE_GENDER = None
VOICE_PROVIDER = None

# Listening
LISTEN_LANGUAGES = ["en-US"] # American English, like the Danielle voice
NO_SPEECH_TIMEOUT = 8.0       # seconds of silence after which the robot stops listening and asks again
LISTEN_TIMEOUT = 20.0         # safety net: never wait longer than this for an answer, even if the robot sends nothing
END_SPEECH_TIMEOUT = 2.0      # seconds of silence before the user's answer counts as finished, this could be changed to make it more natural, but it is not a big problem for this demo

# LED
# Currently just set to green, sets this at default on connect, but the robot can change it during the conversation.
LED_COLOR = "#228B22"              # e.g. "#00FF00"
