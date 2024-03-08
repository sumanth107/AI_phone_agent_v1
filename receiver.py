from flask import Flask
import os
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse
load_dotenv('../AI_phone_agent_v1/.env.py')

app = Flask(__name__)

@app.route("/answer", methods=['GET', 'POST'])
def answer():
    """Respond to an incoming phone call and start the conversation."""
    response = VoiceResponse()
    response.say("Hey there Delilah! What's it like in New York City? I'm a thousand miles")
    # response.record(action='/handle_recording', maxLength=30)
    return str(response)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)