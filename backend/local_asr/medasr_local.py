"""
Self-hosted google/medasr — Google's Health AI Developer Foundations model,
purpose-trained on ~5,000 hours of real physician dictation (radiology,
internal medicine, family medicine). This is the model whose training data
most closely matches your actual use case (solo doctor dictating a
prescription), not conversational Hindi/Hinglish speech.
"""

import tempfile
import os
import json
from dotenv import load_dotenv
from groq import Groq

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# This will automatically read your .env file and load variables into os.environ
load_dotenv() 

MODEL_ID = "google/medasr"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipe = None

def _load_pipeline():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        
        # Replace "hf_YOUR_TOKEN_HERE" with your actual Hugging Face token in your .env file
        hf_token = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")
        
        _pipe = pipeline(
            "automatic-speech-recognition", 
            model=MODEL_ID,
            token=hf_token
        )
    return _pipe

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    pipe = _load_pipeline()

    suffix = os.path.splitext(audio.filename or "chunk.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # chunk_length_s / stride_length_s per Google's own quickstart notebook —
        # handles audio longer than the model's native window via overlapping windows.
        result = pipe(tmp_path, chunk_length_s=20, stride_length_s=2)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
    finally:
        os.remove(tmp_path)

    return {"text": text.strip(), "model": MODEL_ID}

@app.post("/suggest")
async def suggest(payload: dict):
    """Accepts the accumulated dictation transcript, returns a structured prescription."""
    transcript = payload.get("transcript", "")

    prompt = f"""You are parsing a doctor's spoken prescription dictation (NOT a doctor-patient
conversation — this is the doctor speaking prescription details out loud to themselves)
into structured data for a prescription pad.

Transcript:
{transcript}

Return ONLY valid JSON with this exact shape (omit fields you can't find, don't invent data):
{{
  "patient_name": string or null,
  "age": string or null,
  "gender": string or null,
  "diagnosis": [string],
  "medications": [
    {{"name": string, "dosage": string, "frequency": string, "duration": string}}
  ],
  "investigations": [string],
  "follow_up": string or null,
  "advice": [string]
}}
"""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = completion.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        rx = json.loads(raw)
    except json.JSONDecodeError:
        rx = {"patient_name": None, "diagnosis": [], "medications": [],
              "investigations": [], "follow_up": None, "advice": [raw]}

    return rx

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_ID}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)