from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
import pyaudio
import threading
import os
from dotenv import load_dotenv

load_dotenv('../AI_phone_agent_v1/.env.py')

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# PyAudio setup
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
audio = pyaudio.PyAudio()

def main():
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        dg_connection = deepgram.listen.live.v('1')

        # Listen for any transcripts received from Deepgram and write them to the console
        def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            print(f'transcript: {sentence}')
            if len(sentence) == 0:
                return
            print(f'transcript: {sentence}')

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # Create a websocket connection to Deepgram
        options = LiveOptions(
            smart_format=True, model="nova-2", language="en-US"
        )
        dg_connection.start(options)

        lock_exit = threading.Lock()
        exit = False

        # Function to capture audio from the microphone and stream it to Deepgram
        def stream_audio():
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            print("Streaming audio from microphone...")

            while True:
                lock_exit.acquire()
                if exit:
                    lock_exit.release()
                    break
                lock_exit.release()

                data = stream.read(CHUNK)
                dg_connection.send(data)

            stream.stop_stream()
            stream.close()

        audio_thread = threading.Thread(target=stream_audio)
        audio_thread.start()

        input('Press Enter to stop transcription...\n')
        lock_exit.acquire()
        exit = True
        lock_exit.release()

        audio_thread.join()

        # Indicate that we've finished by sending the close stream message
        dg_connection.finish()
        print('Finished')

    except Exception as e:
        print(f'Could not open socket: {e}')
        return

if __name__ == '__main__':
    main()
    audio.terminate()  # Terminate the PyAudio object
