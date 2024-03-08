# Speech to Text using Deepgram API
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
import os
from dotenv import load_dotenv
load_dotenv('../AI_phone_agent_v1/.env.py')


AUDIO_FILE = "output.mp3"
deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

with open(AUDIO_FILE, "rb") as file:
    buffer_data = file.read()

payload: FileSource = {
    "buffer": buffer_data,
}

options = PrerecordedOptions(
    model="nova-2",
    smart_format=True,
)

response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

print(response['results']['channels'][0]['alternatives'][0]['transcript'])
