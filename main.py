# main.py
import os
import uuid
import traceback
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Try to import your modules (fail gracefully if missing)
missing_imports = {}
try:
    import brain_of_the_doctor as brain
except Exception as e:
    brain = None
    missing_imports["brain_of_the_doctor"] = str(e)

try:
    import voice_of_the_doctor as vod
except Exception as e:
    vod = None
    missing_imports["voice_of_the_doctor"] = str(e)

try:
    import voice_of_the_patient as vop
except Exception as e:
    vop = None
    missing_imports["voice_of_the_patient"] = str(e)


app = FastAPI(title="MediTrust API")

# Allow CORS so your React frontend can call this API during development:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audio / uploads folder (project/static/audio)
BASE_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.get("/")
def root():
    info = {"message": "MediTrust API running 🚀"}
    if missing_imports:
        info["missing_imports"] = missing_imports
    return info


@app.post("/ask")
async def ask(request: Request):
    """
    POST JSON: { "query": "patient says ..." }
    Returns: { "reply": "...", "audio_url": "/audio/..." }
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_text = data.get("query")
    if not user_text:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    # Get a textual response: use brain_of_the_doctor.diagnose if available, otherwise echo
    try:
        if brain and hasattr(brain, "diagnose"):
            response_text = brain.diagnose(user_text)
        else:
            response_text = f"You said: {user_text}"
    except Exception as e:
        response_text = f"Error in diagnosis logic: {e}"

    # Prepare filename
    filename = f"reply_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    # Generate TTS (prefer ElevenLabs if available, else fallback to gTTS)
    tts_error = None
    if vod:
        try:
            # If voice_of_the_doctor module created a client, assume ElevenLabs available
            if getattr(vod, "client", None):
                vod.text_to_speech_with_elevenlabs(response_text, filepath)
            else:
                vod.text_to_speech_with_gtts(response_text, filepath)
        except Exception as e:
            tts_error = str(e)
    else:
        tts_error = missing_imports.get("voice_of_the_doctor", "voice module not available")

    result = {"reply": response_text}
    if os.path.exists(filepath):
        result["audio_url"] = f"/audio/{filename}"
    else:
        result["audio_url"] = None
        result["tts_error"] = tts_error

    return JSONResponse(result)


@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    """
    Upload an audio file -> get transcription (via Groq / whisper)
    Returns: { "transcription": "..." }
    """
    if not vop:
        raise HTTPException(status_code=500, detail=f"voice_of_the_patient not available: {missing_imports.get('voice_of_the_patient')}")

    ext = os.path.splitext(file.filename)[1] or ".wav"
    filename = f"patient_{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(AUDIO_DIR, filename)

    # Save uploaded file
    with open(filepath, "wb") as f:
        f.write(await file.read())

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment variables")

    try:
        # vop.transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY)
        transcription = vop.transcribe_with_groq("whisper-large-v3", filepath, groq_key)
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.post("/talk")
async def talk(
    audio: UploadFile = File(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Full pipeline:
    - upload audio (patient) [required]
    - optional image (patient)
    Returns: { patient_text, doctor_response, audio_url }
    """
    if not vop:
        raise HTTPException(status_code=500, detail=f"voice_of_the_patient not available: {missing_imports.get('voice_of_the_patient')}")

    # Save audio
    audio_ext = os.path.splitext(audio.filename)[1] or ".wav"
    audio_filename = f"patient_{uuid.uuid4().hex}{audio_ext}"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment variables")

    # Transcribe audio
    try:
        transcription = vop.transcribe_with_groq("whisper-large-v3", audio_path, groq_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # If image provided and brain module supports multimodal analysis, call it.
    doctor_response = None
    if image and brain and hasattr(brain, "encode_image") and hasattr(brain, "analyze_image_with_query"):
        img_ext = os.path.splitext(image.filename)[1] or ".jpg"
        img_filename = f"img_{uuid.uuid4().hex}{img_ext}"
        img_path = os.path.join(AUDIO_DIR, img_filename)
        with open(img_path, "wb") as f:
            f.write(await image.read())

        try:
            encoded = brain.encode_image(img_path)
            system_prompt = os.environ.get("SYSTEM_PROMPT") or ""
            query = f"{system_prompt}\nPatient: {transcription}"
            # model selection (tweak if you want)
            model = "meta-llama/llama-4-scout-17b-16e-instruct"
            doctor_response = brain.analyze_image_with_query(query, model, encoded)
        except Exception as e:
            doctor_response = f"Image analysis failed: {e}"
    else:
        # fallback: use diagnose() if available
        if brain and hasattr(brain, "diagnose"):
            try:
                doctor_response = brain.diagnose(transcription)
            except Exception as e:
                doctor_response = f"diagnose() failed: {e}"
        else:
            doctor_response = f"You said: {transcription}"

    # Generate TTS for doctor_response
    filename = f"reply_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    tts_error = None
    if vod:
        try:
            if getattr(vod, "client", None):
                vod.text_to_speech_with_elevenlabs(doctor_response, filepath)
            else:
                vod.text_to_speech_with_gtts(doctor_response, filepath)
        except Exception as e:
            tts_error = str(e)
    else:
        tts_error = missing_imports.get("voice_of_the_doctor", "voice_of_the_doctor missing")

    result = {
        "patient_text": transcription,
        "doctor_response": doctor_response
    }
    if os.path.exists(filepath):
        result["audio_url"] = f"/audio/{filename}"
    else:
        result["audio_url"] = None
        result["tts_error"] = tts_error

    return JSONResponse(result)


@app.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...), query: Optional[str] = None):
    """
    Upload image + optional textual query; uses brain_of_the_doctor.analyze_image_with_query
    """
    if not brain or not hasattr(brain, "analyze_image_with_query") or not hasattr(brain, "encode_image"):
        raise HTTPException(status_code=500, detail="brain_of_the_doctor multimodal functions not available")

    img_ext = os.path.splitext(image.filename)[1] or ".jpg"
    img_filename = f"img_{uuid.uuid4().hex}{img_ext}"
    img_path = os.path.join(AUDIO_DIR, img_filename)
    with open(img_path, "wb") as f:
        f.write(await image.read())

    encoded = brain.encode_image(img_path)
    if not query:
        query = "Please inspect the image and answer concisely."

    try:
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        response = brain.analyze_image_with_query(query, model, encoded)
        return {"analysis": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {e}")


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="audio/mpeg")
