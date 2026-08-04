"""
Word Error Rate evaluation for AI Scribe transcripts.

Usage:
    pip install jiwer
    python measure_wer.py

Computes:
  1. Overall WER (reference vs hypothesis)
  2. Medical-term-only WER — the metric that actually matters for this use case,
     since a doctor doesn't care if "twice daily" became "two times a day" but
     absolutely cares if "Metformin" became "met for min".

Edit REFERENCE / HYPOTHESIS below (or wire this up to read the live transcript
from the /transcribe responses) and run.
"""

import re
from jiwer import wer

# Terms considered "medical" for the purpose of the narrow WER slice.
# Keep this in sync with MEDICAL_VOCAB_PROMPT in backend/app.py.
MEDICAL_TERMS = {
    "metformin", "amlodipine", "telmisartan", "atorvastatin", "glipizide",
    "insulin", "glargine", "pantoprazole", "azithromycin",
    "od", "bd", "tds", "qid", "sos", "ac", "pc", "hs",
    "hba1c", "egfr", "ldl", "bmi", "creatinine", "albuminuria",
    "t2dm", "hypertension", "dyslipidemia", "cad", "ckd",
    "systolic", "diastolic",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def medical_term_wer(reference: str, hypothesis: str) -> float | None:
    """Extracts only the medical-term tokens from both strings (in order) and
    computes WER over just those tokens. Returns None if the reference has no
    medical terms to compare."""
    ref_tokens = [w for w in normalize(reference).split() if w in MEDICAL_TERMS]
    hyp_tokens = [w for w in normalize(hypothesis).split() if w in MEDICAL_TERMS]
    if not ref_tokens:
        return None
    return wer(" ".join(ref_tokens), " ".join(hyp_tokens))


def evaluate(reference: str, hypothesis: str, label: str = ""):
    overall = wer(normalize(reference), normalize(hypothesis))
    med = medical_term_wer(reference, hypothesis)

    print(f"--- {label or 'result'} ---")
    print(f"Overall WER:      {overall:.2%}")
    print(f"Medical-term WER: {med:.2%}" if med is not None else "Medical-term WER: n/a (no medical terms in reference)")
    print()
    return {"overall_wer": overall, "medical_wer": med}


if __name__ == "__main__":
    reference = (
        "Tab Metformin 500 mg twice daily after meals 30 days "
        "Tab Amlodipine 5 mg once daily morning 30 days "
        "Tab Telmisartan 40 mg once daily morning 30 days"
    )

    # Paste what the ASR actually returned here (from the live transcript panel,
    # or from GET-ing the transcript off the backend) and re-run.
    hypothesis = (
        "Tab met for min 500 mg twice daily after meals 30 days "
        "Tab amlodipine 5 mg once daily morning 30 days "
        "Tab telmisartan 40 mg once daily morning 30 days"
    )

    evaluate(reference, hypothesis, label="whisper-large-v3")