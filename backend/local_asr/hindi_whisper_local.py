"""
Self-hosted Hindi/Hinglish Whisper fine-tune. Two options, toggled below:

  1. Oriserve/Whisper-Hindi2Hinglish-Apex  (recommended — see note)
     Oriserve's newest, ~800M params. Built specifically for conversational
     Hindi, Hinglish, and Indian-accented English. Their own numbers claim 8x
     faster inference and 42% average improvement over base Whisper. This is
     the closest fit on paper to your actual code-switching problem.
     NOTE: you originally found "Whisper-Hindi2Hinglish-Prime" — Oriserve's own
     model card says Prime is superseded by Apex, so Apex is what's below.

  2. akanshSirohi/whisper-Large-v3-hindi-ct2
     An individual's CTranslate2 conversion of a Hindi Whisper large-v3
     fine-tune. Confirmed to exist, but I could not find a model card with
     disclosed training data or WER benchmarks — test with appropriately lower
     confidence than Apex. Its upside is speed: CTranslate2 (faster-whisper)
     runs fast even on CPU.

Set MODEL_TYPE below to switch between them — the API this serves stays the
same either way (port 8001, /transcribe), so the frontend dropdown doesn't
need to know or care which one is actually running.

    # for Apex (standard transformers path):
    pip install -U transformers torch accelerate

    # for the ct2 option (faster-whisper path):
    pip install faster-whisper

    python hindi_whisper_local.py
"""

import tempfile
import os
import json
from dotenv import load_dotenv
from groq import Groq

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Load variables from .env file
load_dotenv()

# ---- flip this to switch models ----
MODEL_TYPE = "apex"  # "apex" or "ct2"
# -------------------------------------

APEX_MODEL_ID = "Oriserve/Whisper-Hindi2Hinglish-Apex"
CT2_MODEL_ID = "akanshSirohi/whisper-Large-v3-hindi-ct2"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipe = None
_ct2_model = None

def _load_apex():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            APEX_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(APEX_MODEL_ID)
        _pipe = pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor, device=device,
        )
    return _pipe

def _load_ct2():
    global _ct2_model
    if _ct2_model is None:
        from faster_whisper import WhisperModel
        _ct2_model = WhisperModel(CT2_MODEL_ID)
    return _ct2_model

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "chunk.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        if MODEL_TYPE == "ct2":
            model = _load_ct2()
            segments, _info = model.transcribe(tmp_path)
            text = " ".join(seg.text for seg in segments).strip()
            model_id = CT2_MODEL_ID
        else:
            pipe = _load_apex()
            result = pipe(tmp_path)
            text = (result.get("text", "") if isinstance(result, dict) else str(result)).strip()
            model_id = APEX_MODEL_ID
    finally:
        os.remove(tmp_path)

    return {"text": text, "model": model_id}

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
    return {"status": "ok", "model_type": MODEL_TYPE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)