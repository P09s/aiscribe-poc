"""
Side-by-side WER comparison across models on the same reference text — this is
what turns "Qwen felt better than medasr" into a number you can put in front
of Shantanu.

Usage:
    pip install jiwer
    python compare_models.py

Fill in REFERENCE (paste the exact patient block from
test_scripts/prescription_script_extended.txt you dictated — copy it verbatim,
including the hesitations/self-corrections, since that's what was actually
said) and each model's TRANSCRIPTS entry with what its live transcript panel
showed, then run.

CAVEAT — read before trusting the numbers: WER here is plain word-level edit
distance. If a model outputs Devanagari for a word your reference has in Roman
script (e.g. reference "Metformin" vs a model outputting "मेटाफॉर्न"), jiwer
scores that as a totally different word even when a human would recognize it
as the same drug, correctly transcribed in a different script. That's not a
bug in the script, it's a real limitation of word-level WER on code-switched,
multi-script text — treat the numbers below as directional, not absolute, and
still eyeball the transcripts yourself before writing the group summary.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from measure_wer import evaluate  # noqa: E402


# Paste the exact patient block you dictated, verbatim (hesitations included —
# WER measures faithfulness to what was actually said, not to the "clean"
# intended meaning; see the note in the module docstring about self-corrections).
REFERENCE = """
Patient 1 — Suresh Yadav, umar pachaas saal, male.
Diagnosis: Type 2 Diabetes Mellitus, uncontrolled — HbA1c ekdum high aaya hai,
9.2 percent. Hypertension bhi hai, saath mein dyslipidemia.
Prescription — Tab Metformin, wait, Metformin 500 mg — nahi nahi, 1000 mg BD,
after meals, 30 days ke liye. Tab Amlodipine 5 mg OD, morning mein, 30 days.
Tab Atorvastatin 20 mg, once daily at night, 30 days. Tab Losartan 50 mg,
subah lena hai, once daily, 30 days.
"""

# Fill each of these in with what that model's live transcript panel actually
# showed for the same patient block. Leave a model out entirely (don't add an
# empty string) if you didn't test it on this block.
TRANSCRIPTS = {
    "whisper-large-v3": "",
    "whisper-large-v3-turbo": "",
    "medasr": "",
    "qwen3-asr (0.6B, with correction)": "",
    "oriserve-apex": "",
}


def main():
    results = {}
    for label, hyp in TRANSCRIPTS.items():
        if not hyp.strip():
            continue
        results[label] = evaluate(REFERENCE, hyp, label=label)

    if not results:
        print("Nothing to compare — fill in TRANSCRIPTS with at least one model's output.")
        return

    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["medical_wer"] if kv[1]["medical_wer"] is not None else 1.0),
    )

    print("=" * 60)
    print("RANKED BY MEDICAL-TERM WER (lower is better — this is the number that matters)")
    print("=" * 60)
    for label, r in ranked:
        med = f"{r['medical_wer']:.1%}" if r["medical_wer"] is not None else "n/a"
        print(f"{label:38s}  medical WER: {med:>6s}   overall WER: {r['overall_wer']:.1%}")


if __name__ == "__main__":
    main()