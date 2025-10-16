# if you don't use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import os
import gradio as gr
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purpose. 
What's in this image? Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies. 
Do not add any numbers or special characters in your response. 
Your response should mimic an actual doctor, concise (max 2 sentences), and no preamble."""

def process_inputs(audio_filepath, image_filepath):
    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3"
    )

    # Handle the image input
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encode_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for me to analyze"

    # Generate TTS audio and return file path
    output_filepath = "final.mp3"
    text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=output_filepath)

    return speech_to_text_output, doctor_response, output_filepath

iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice")   # <--- better label, and it will show
    ],
    title="AI Doctor with Vision and Voice"
)
iface.launch(debug=True)
