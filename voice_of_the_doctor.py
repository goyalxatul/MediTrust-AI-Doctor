# if you don't use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import os
from gtts import gTTS
import platform
import subprocess
from playsound import playsound
from elevenlabs import ElevenLabs

# Load ElevenLabs API key
ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")

# Create ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

def play_audio(file_path):
    """Cross-platform audio playback"""
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', file_path])
        elif os_name == "Windows":
            playsound(file_path)
        elif os_name == "Linux":
            subprocess.run(['aplay', file_path])
        else:
            raise OSError("Unsupported OS for audio playback")
    except Exception as e:
        print(f"Error playing audio: {e}")

def text_to_speech_with_gtts(input_text, output_filepath):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=input_text, lang="en", slow=False)
        tts.save(output_filepath)
        play_audio(output_filepath)
        return output_filepath
    except Exception as e:
        print(f"gTTS error: {e}")

def text_to_speech_with_elevenlabs(input_text, output_filepath, voice="21m00Tcm4TlvDq8ikWAM"):  # Rachel
    if not client:
        raise ValueError("ELEVEN_API_KEY not set in environment variables")
    
    try:
        audio = client.text_to_speech.convert(
            voice_id=voice,
            model_id="eleven_multilingual_v2",
            text=input_text
        )
        with open(output_filepath, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        play_audio(output_filepath)
        return output_filepath
    except Exception as e:
        print(f"ElevenLabs error: {e}")


# Example usage:
if __name__ == "__main__":
    text = "Hi, this is AI with MediTrust!"

    print("Testing gTTS...")
    text_to_speech_with_gtts(text, "gtts_test.mp3")

    print("Testing ElevenLabs...")
    text_to_speech_with_elevenlabs(text, "eleven_test.mp3")
