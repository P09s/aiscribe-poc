import os
import json
import tempfile
import difflib

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq's Whisper endpoint accepts an optional `prompt` used only to bias decoding
# (spelling/vocabulary), not as an instruction. Feeding it a dense list of the
# drug names / dosage units / abbreviations we expect nudges the model toward
# correct spellings for terms it would otherwise mangle (e.g. "met for min" -> "Metformin").
MEDICAL_VOCAB_PROMPT = (
    "Prescription dictation. Drugs: Metformin, Amlodipine, Telmisartan, Atorvastatin, "
    "Glipizide, Insulin Glargine, Pantoprazole, Azithromycin. "
    "Dosages: 500 mg, 250 mg, 100 mg, 50 mg, 40 mg, 25 mg, 10 mg, 5 mg, 10 units, 0.5 ml. "
    "Frequencies: OD, BD, TDS, QID, SOS, AC, PC, HS. "
    "Terms: HbA1c, eGFR, LDL, BMI, creatinine, albuminuria, T2DM, hypertension, "
    "dyslipidemia, CAD, CKD, systolic, diastolic."
)

ALLOWED_MODELS = {"whisper-large-v3", "whisper-large-v3-turbo"}

# Fixed formulary for fuzzy-correction. This is the actual practical fix for a
# closed drug list: don't try to make the ASR perfect, catch its near-misses
# after the fact against terms you know are valid. Extend this from your real
# formulary before relying on it for anything beyond today's test.
FORMULARY = [
    "Metformin", "Amlodipine", "Telmisartan", "Atorvastatin", "Glipizide",
    "Insulin Glargine", "Pantoprazole", "Azithromycin", "Paracetamol",
    "HbA1c", "eGFR", "LDL", "BMI", "Creatinine", "Albuminuria",
    "T2DM", "Hypertension", "Dyslipidemia", "CAD", "CKD",
]
_FORMULARY_LOWER = {w.lower(): w for w in FORMULARY}

# Groq flags hallucinated / non-speech segments via these two fields on each
# verbose_json segment. These thresholds are Groq's own documented guidance,
# not something we invented — see console.groq.com/docs/speech-to-text.
NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0


def filter_hallucinated_segments(segments: list) -> str:
    """Drops segments Whisper itself is signaling as likely silence/hallucination
    and joins what's left. This is what stops 'Drugs. Drugs. Drugs.' from a quiet
    mic from ever reaching the transcript."""
    kept = []
    for seg in segments:
        no_speech = seg.get("no_speech_prob", 0.0)
        avg_logprob = seg.get("avg_logprob", 0.0)
        if no_speech > NO_SPEECH_PROB_THRESHOLD and avg_logprob < AVG_LOGPROB_THRESHOLD:
            continue
        kept.append(seg.get("text", ""))
    return " ".join(t.strip() for t in kept if t.strip())


def correct_drug_names(text: str, cutoff: float = 0.78) -> str:
    """Fuzzy-matches each word/short phrase against the known formulary and swaps
    in the correct spelling on a close-enough match. Deliberately conservative
    (high cutoff) — better to leave an unrecognized word alone than to silently
    rewrite something that wasn't actually a drug name."""
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        # try 2-word phrases first (e.g. "Insulin Glargine") before single words
        matched = False
        for span in (2, 1):
            if i + span > len(words):
                continue
            candidate = " ".join(words[i:i + span]).strip(",.").lower()
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


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Form("whisper-large-v3"),
):
    """Accepts a short audio chunk from the doctor's mic, returns transcript text.
    Filters Whisper's own hallucinated-silence segments and fuzzy-corrects drug
    names against the formulary before returning."""
    if model not in ALLOWED_MODELS:
        model = "whisper-large-v3"

    suffix = os.path.splitext(audio.filename or "chunk.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=f,
                model=model,
                prompt=MEDICAL_VOCAB_PROMPT,
                temperature=0.0,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        segments = [s if isinstance(s, dict) else s.model_dump() for s in (result.segments or [])]
        text = filter_hallucinated_segments(segments) if segments else (result.text or "").strip()
        text = correct_drug_names(text)
    finally:
        os.remove(tmp_path)

    return {"text": text, "model": model}


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
        model="openai/gpt-oss-20b",
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
    return {"status": "ok"}