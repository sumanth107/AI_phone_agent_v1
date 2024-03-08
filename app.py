import os
from flask import Flask, request, jsonify
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from openai import OpenAI
from dotenv import load_dotenv
import requests
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv('../AI_phone_agent_v1/.env.py')

app = Flask(__name__)

# Twilio client setup
twilio_client = Client(os.environ['TWILIO_ACCOUNT_SID'], os.environ['TWILIO_AUTH_TOKEN'])

# Deepgram client setup
deepgram_client = DeepgramClient(os.environ['DEEPGRAM_API_KEY'])

# Eleven Labs API setup
eleven_labs_url = "https://api.elevenlabs.io/v1/text-to-speech/ThT5KcBeYPX3keUQqHPh"
eleven_labs_headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": os.getenv("ELEVEN_API_KEY")
}


@app.route("/make_call", methods=['GET'])
def make_call():
    """Initiate a phone call to the specified number."""
    call = twilio_client.calls.create(
        url='https://modest-pretty-squirrel.ngrok-free.app',  # Update this URL to your ngrok URL
        to='+16507894895',  # Update this number to the number you want to call
        from_='+18559540265'  # Update this number to your Twilio phone number
    )
    return jsonify({"message": "Call initiated", "call_sid": call.sid})


@app.route("/answer", methods=['GET', 'POST'])
def answer():
    """Respond to an incoming phone call and start the conversation."""
    response = VoiceResponse()
    response.say("Hey there Delilah! What's it like in New York City?")
    response.record(action='/handle_recording', method='POST', timeout=3)
    return str(response)


@app.route("/handle_recording", methods=['POST'])
def handle_recording():
    """Handle the recorded message, generate a response, and convert it to speech."""
    recording_url = request.form.get("RecordingUrl")+'.mp3'
    logging.info(str(recording_url))
    transcript = transcribe_recording(recording_url)
    logging.info(transcript)
    response_text = generate_response(transcript)
    logging.info(response_text)
    speech_url = text_to_speech(response_text)
    response = VoiceResponse()
    response.play(speech_url)
    return str(response)


def transcribe_recording(url):
    """Transcribe the recorded message using Deepgram."""
    print('-----------'+ url)
    options = PrerecordedOptions(model="nova-2", smart_format=True)
    audio_url = {"url": url}
    response = deepgram_client.listen.prerecorded.v("1").transcribe_url(audio_url, options)
    transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
    return transcript


def generate_response(transcript):
    """Generate a response to the transcript using OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a friendly AI assistant. Ask me anything!"},
            {"role": "user", "content": transcript}
        ]
    )
    return response.choices[0].message.content


def text_to_speech(text):
    """Convert the response text to speech using Eleven Labs."""
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(eleven_labs_url, json=data, headers=eleven_labs_headers)
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return "output.mp3"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
