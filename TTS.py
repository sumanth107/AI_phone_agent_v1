# Text To Speech (TTS) Implementation using ElevenLabs API

from dotenv import load_dotenv
import os
import requests
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
from elevenlabs.client import ElevenLabs
from elevenlabs import generate, play, voices, Voice, VoiceSettings, generate, stream


load_dotenv('../AI_phone_agent_v1/.env.py')

audio_stream = generate(
    api_key=os.getenv("ELEVEN_API_KEY"),
    text="Hello!!", stream=True, voice="Dorothy", model="eleven_monolingual_v1"
)

stream(audio_stream)

#
# eleven_labs_url = "https://api.elevenlabs.io/v1/text-to-speech/ThT5KcBeYPX3keUQqHPh"
#
# eleven_labs_headers = {
#     "Accept": "audio/mpeg",
#     "Content-Type": "application/json",
#     "xi-api-key": os.getenv("ELEVEN_API_KEY")
# }
#
#
# def text_to_speech(text):
#     data = {
#         "text": text,
#         "model_id": "eleven_monolingual_v1",
#         "voice_settings": {
#             "stability": 0.5,
#             "similarity_boost": 0.5
#         }
#     }
#     response = requests.post(eleven_labs_url, json=data, headers=eleven_labs_headers)
#     with open('temp.mp3', 'wb') as f:
#         for chunk in response.iter_content(chunk_size=1024):
#             if chunk:
#                 f.write(chunk)
#     return "temp.mp3"


text = "Hi"
text_to_speech(text)


