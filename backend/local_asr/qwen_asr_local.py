"""
Self-hosted Qwen3-ASR endpoint — the "in-house, not Whisper" option.

Qwen3-ASR (Alibaba, Apache 2.0) covers 30+ languages including Hindi/English
with mid-sentence code-switching. Uses Qwen/Qwen3-ASR-0.6B-hf for now — swap
to 1.7B-hf if 0.6B's accuracy on drug names doesn't hold up under the extended
stress test (it's the smaller/faster of the two, which is likely part of why
formulary drug names are landing rougher than expected — 1.7B is the next
thing to try if the correction layer below still isn't enough).
"""

import os
import re
import tempfile
import json
import difflib
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

# Extend this with your actual formulary — the more specific to what your
# doctors actually prescribe, the harder Qwen3-ASR biases toward it.
MEDICAL_CONTEXT = (
    "Vocabulary: Metformin, Amlodipine, Amoxicillin, Azithromycin, Metronidazole, "
    "Pantoprazole, Atorvastatin, Losartan, Paracetamol, Ibuprofen, Cetirizine, "
    "HbA1c, BP, RBS, FBS, mg, ml, OD, BD, TDS, QID, SOS, stat. "
    "Numbers and dosages should be transcribed as digits (e.g. 500 mg, twice daily)."
)

# Same list as MEDICAL_CONTEXT, but as discrete entries for fuzzy-matching —
# keep these two in sync when you extend the formulary.
FORMULARY = [
    "Metformin", "Amlodipine", "Amoxicillin", "Azithromycin", "Metronidazole",
    "Pantoprazole", "Atorvastatin", "Losartan", "Paracetamol", "Ibuprofen",
    "Cetirizine", "HbA1c", "BP", "RBS", "FBS",
]
_FORMULARY_LOWER = {w.lower(): w for w in FORMULARY}

# Deliberately lower than app.py's Whisper cutoff (0.78). Qwen's drug-name
# misses in testing were bigger distortions than Whisper's (e.g. "Amlodipine"
# -> "MLOD pine", "Metformin" -> "मेटाफॉर्न") — a tighter cutoff would let
# those slip through uncorrected. Lowering it risks more false-positive
# corrections on words that were never drug names; that tradeoff is exactly
# what the extended stress-test run should validate either way.
FUZZY_CUTOFF = 0.62


def correct_drug_names(text: str, cutoff: float = FUZZY_CUTOFF) -> str:
    """Fuzzy-matches each word/short phrase against the known formulary and swaps
    in the correct spelling on a close-enough match. Runs on whatever script the
    ASR produced (Devanagari, Roman-script Hinglish, or English) — matching is
    done against Roman-script formulary entries, so it only fixes formulary words
    that came out in Roman characters. Devanagari renderings of drug names
    (e.g. "मेटाफॉर्न") won't be caught by this pass; that's a known gap, not
    a silent failure — flag it if it shows up a lot in the extended test."""
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        matched = False
        for span in (2, 1):
            if i + span > len(words):
                continue
            candidate = " ".join(words[i:i + span]).strip(",.।").lower()
            if not candidate:
                continue
            match = difflib.get_close_matches(candidate, _FORMULARY_LOWER.keys(), n=1, cutoff=cutoff)
            if match:
                out.append(_FORMULARY_LOWER[match[0]])
                i += span
                matched = True
                break
        if not matched:
            out.append(words[i])
            i += 1
    return " ".join(out)


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

    text = ""
    detected_lang = None

    try:
        # 1. Load audio with librosa
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_audio, sampling_rate = librosa.load(tmp_path, sr=16000)

        # 2. Build the prompt: system message carries the medical vocabulary
        # (context biasing), user turn carries the audio. Language is left on
        # auto-detect on purpose — Indian clinical dictation code-switches
        # between Hindi and English mid-sentence, and forcing a single
        # language would hurt accuracy on whichever one you didn't pick.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": MEDICAL_CONTEXT}]},
            {"role": "user", "content": [{"type": "audio", "audio_url": "dummy.wav"}]},
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
        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 6. Strip the model's own leaked language tag ("language Hindi<text>",
        # "language English<text>") instead of feeding it downstream. We keep
        # the detected value for logging/debugging rather than discarding it —
        # a run of unexpected tags on audio you know is en/hi is a useful
        # signal something upstream (mic gain, chunk length) is off, even
        # though it doesn't corrupt the transcript itself.
        match = re.match(r'^\s*language\s*([A-Za-z]+)', raw_text, flags=re.IGNORECASE)
        detected_lang = match.group(1) if match else None
        text = re.sub(r'^\s*language\s*[A-Za-z]+', '', raw_text, flags=re.IGNORECASE).strip()

        # 7. Formulary correction — new. Catches Roman-script near-misses
        # ("MLOD pine" -> "Amlodipine") after the language tag is gone, so the
        # regex above isn't tripped up by corrected text and this isn't tripped
        # up by the leaked tag.
        text = correct_drug_names(text)

        if detected_lang and detected_lang.lower() not in ("english", "hindi"):
            print(f"Note: detected_lang={detected_lang} on expected en/hi audio")

        print(f"Success ({detected_lang}): '{text}'")

    except Exception as e:
        print(f"Error: {e}")
        text = ""
    finally:
        os.remove(tmp_path)

    return {"text": text.strip(), "model": MODEL_ID, "detected_language": detected_lang}


@app.post("/suggest")
async def suggest(payload: dict):
    """Accepts the accumulated dictation transcript, returns a structured prescription."""
    transcript = payload.get("transcript", "")
    print("Generating structured prescription via Groq...")

    prompt = f"""You are parsing a doctor's spoken prescription dictation (NOT a doctor-patient
conversation — this is the doctor speaking prescription details out loud to themselves)
into structured data for a prescription pad.

Known formulary (prefer these spellings if the transcript contains a close variant,
but do not invent a drug that isn't plausibly supported by the transcript):
{MEDICAL_CONTEXT}

IMPORTANT — follow_up vs. investigation timing: doctors often mention two different
timeframes in one dictation — when to REPEAT A TEST (e.g. "HbA1c repeat after 3
months") and when the PATIENT SHOULD COME BACK for a visit (e.g. "Follow up in 6
weeks"). These are not the same thing. Only use an explicit "follow up" / "review"
/ "come back" phrase for the follow_up field. A test-recheck interval belongs in
investigations (e.g. "HbA1c (repeat in 3 months)"), never in follow_up, even if it's
the only timeframe mentioned.

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