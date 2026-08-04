import os
import tempfile
import json
import librosa
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

load_dotenv()
MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_processor = None

def _load_model():
    global _model, _processor
    if _model is None:
        print(f"Loading {MODEL_ID} locally...")
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")
    return _model, _processor

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    print("Received audio chunk...")
    model, processor = _load_model()

    suffix = os.path.splitext(audio.filename or "chunk.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # 1. Load audio with librosa
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_audio, sampling_rate = librosa.load(tmp_path, sr=16000)
        
        # 2. Generate required prompt string containing <|audio_pad|> tags
        messages = [
            {"role": "user", "content": [{"type": "audio", "audio_url": "dummy.wav"}]}
        ]
        text_prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 3. Create tensors from audio and text prompt
        inputs = processor(
            text=text_prompt, 
            audio=raw_audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        )
        
        # 4. Move tensors to device and cast precision to match model.dtype (bfloat16 fix)
        inputs = {
            k: v.to(model.device, dtype=model.dtype) if torch.is_floating_point(v) else v.to(model.device)
            for k, v in inputs.items()
        }
        
        print("Generating text...")
        output_ids = model.generate(**inputs, max_new_tokens=256)
        
        # 5. Slice off prompt tokens to extract decoded output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Success: '{text.strip()}'")
        
    except Exception as e:
        print(f"Error: {e}")
        text = ""
    finally:
        os.remove(tmp_path)

    return {"text": text.strip(), "model": MODEL_ID}

@app.post("/suggest")
async def suggest(payload: dict):
    """Accepts the accumulated dictation transcript, returns a structured prescription."""
    transcript = payload.get("transcript", "")
    print("Generating structured prescription via Groq...")

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
        print("Prescription generated successfully.")
    except json.JSONDecodeError:
        print("Failed to parse JSON from LLM.")
        rx = {"patient_name": None, "diagnosis": [], "medications": [],
              "investigations": [], "follow_up": None, "advice": [raw]}

    return rx

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_ID}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)