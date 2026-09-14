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
from pathlib import Path
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

# ── Formulary loading ────────────────────────────────────────────────────────
# Loads from backend/formulary/nlem_2022.json (India's National List of
# Essential Medicines 2022 — the government-published source of truth for drugs
# that can appear in an Indian prescription). Falls back to a minimal hardcoded
# list if the file is missing so the server still starts.
#
# To extend: add drug names to nlem_2022.json and restart. No code changes
# needed. If a new drug isn't in the NLEM but your doctors prescribe it, add it
# to nlem_2022.json — the file is the single source of truth now, not this code.

_FORMULARY_PATH = Path(__file__).parent.parent / "formulary" / "nlem_2022.json"
_ALIASES_PATH = Path(__file__).parent.parent / "formulary" / "phonetic_aliases.json"

_FALLBACK_FORMULARY = [
    "Metformin", "Amlodipine", "Amoxicillin", "Azithromycin",
    "Pantoprazole", "Atorvastatin", "Losartan", "Paracetamol",
]

def _load_formulary() -> list[str]:
    if _FORMULARY_PATH.exists():
        with open(_FORMULARY_PATH, "r", encoding="utf-8") as f:
            drugs = json.load(f)
        print(f"Formulary loaded: {len(drugs)} drugs from {_FORMULARY_PATH.name}")
        return drugs
    print(f"WARNING: {_FORMULARY_PATH} not found — using fallback formulary ({len(_FALLBACK_FORMULARY)} drugs)")
    return _FALLBACK_FORMULARY

def _load_aliases() -> dict:
    if _ALIASES_PATH.exists():
        with open(_ALIASES_PATH, "r", encoding="utf-8") as f:
            aliases = json.load(f)
        print(f"Phonetic aliases loaded: {len(aliases)} entries from {_ALIASES_PATH.name}")
        return aliases
    print(f"WARNING: {_ALIASES_PATH} not found — run backend/formulary/generate_aliases.py to generate it")
    return {}

FORMULARY = _load_formulary()
_FORMULARY_LOWER = {w.lower(): w for w in FORMULARY}
_PHONETIC_ALIASES = _load_aliases()

# Clinical abbreviations added separately — these don't live in the NLEM drug
# list but are just as important for ASR correction in prescription dictation.
_CLINICAL_ABBREVS = {
    "od": "OD", "bd": "BD", "tds": "TDS", "qid": "QID",
    "sos": "SOS", "ac": "AC", "pc": "PC", "hs": "HS", "stat": "STAT",
    "hba1c": "HbA1c", "bp": "BP", "rbs": "RBS", "fbs": "FBS",
    "egfr": "eGFR", "ldl": "LDL", "bmi": "BMI",
}

# Build MEDICAL_CONTEXT dynamically from the loaded formulary so the ASR
# vocabulary-bias prompt always reflects the current formulary file —
# no more keeping two lists in sync manually.
def _build_medical_context(formulary: list[str]) -> str:
    # Top 30 most commonly prescribed — enough for the prompt bias without
    # hitting token limits. The full list is used for fuzzy correction below.
    top_drugs = ", ".join(formulary[:30])
    abbrevs = ", ".join(_CLINICAL_ABBREVS.values())
    return (
        f"Prescription dictation vocabulary. Common drugs: {top_drugs}. "
        f"Abbreviations: {abbrevs}, mg, ml, units. "
        "Dosages should be transcribed as digits (e.g. 500 mg, twice daily, OD, BD)."
    )

MEDICAL_CONTEXT = _build_medical_context(FORMULARY)

# Fuzzy cutoff — raised back to 0.72 now that phonetic aliases handle the
# hard cases. A lower cutoff was causing dangerous false positives
# (e.g. "Dapoxetine" matching for "Domperidone"). The alias file covers
# 2900+ variants so fuzzy only needs to catch minor spelling differences now.
FUZZY_CUTOFF = 0.72


def correct_drug_names(text: str, cutoff: float = FUZZY_CUTOFF) -> str:
    """Three-pass correction:
    Pass 1 — phonetic alias map (multi-word and single-word known mishearings)
    Pass 2 — exact match on clinical abbreviations (OD, BD, HbA1c etc.)
    Pass 3 — fuzzy match on drug names from the full NLEM formulary

    Phonetic aliases run first because they handle multi-word splits
    (e.g. 'onden cetron' → 'Ondansetron') that neither abbreviation
    nor single-token fuzzy matching can catch."""
    import re as _re

    # Pass 1 — phonetic aliases on full text string
    # Sort by length descending so longer phrases match before their substrings
    text_lower = text.lower()
    for alias in sorted(_PHONETIC_ALIASES.keys(), key=len, reverse=True):
        if alias in text_lower:
            pattern = _re.compile(_re.escape(alias), _re.IGNORECASE)
            text = pattern.sub(_PHONETIC_ALIASES[alias], text)
            text_lower = text.lower()

    # Pass 2 & 3 — token-level abbreviation + fuzzy correction
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        raw_word = words[i]
        clean = raw_word.strip(",.।").lower()

        # Pass 2: exact abbreviation match
        if clean in _CLINICAL_ABBREVS:
            out.append(_CLINICAL_ABBREVS[clean])
            i += 1
            continue

        # Pass 3: fuzzy drug-name match — try 2-word spans first
        matched = False
        for span in (2, 1):
            if i + span > len(words):
                continue
            candidate = " ".join(words[i:i + span]).strip(",.।").lower()
            if not candidate:
                continue
            match = difflib.get_close_matches(
                candidate, _FORMULARY_LOWER.keys(), n=1, cutoff=cutoff
            )
            if match:
                out.append(_FORMULARY_LOWER[match[0]])
                i += span
                matched = True
                break

        if not matched:
            out.append(raw_word)
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