# AI Scribe PoC

A small end-to-end demo: WebRTC video call → live audio capture → transcription
(Groq-hosted Whisper) → AI suggestions (Groq LLM) → mock care plan.

Note: This uses Groq's hosted Whisper API for transcription (not a locally-run
model like Qwen3-ASR) to keep setup simple — no GPU, no model downloads. The
model choice for real production use is a separate decision already discussed
separately; this demo is about proving the pipeline shape end to end.

## Setup

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and paste your Groq API key (the same one used for Daivam):

```
GROQ_API_KEY=gsk_...
```

Run the server:

```bash
uvicorn app:app --reload --port 8000
```

Leave this running in one terminal.

### 2. Frontend

No build step needed — it's a single HTML file. In a second terminal:

```bash
cd frontend
python3 -m http.server 5500
```

### 3. Try it

Open two browser windows (or one normal + one incognito, so they don't share
camera/mic state):

- Window 1: `http://localhost:5500` → Room ID: `test1` → Role: Doctor → Join call
- Window 2: `http://localhost:5500` → Room ID: `test1` (same room) → Role: Patient → Join call

Both video feeds should connect within a couple seconds.

Click **Start recording** in either window, talk for a bit (simulate a
consult — mention a symptom, a possible medication), click **Stop recording**,
then **Generate suggestions**. Suggestions appear as cards — click Accept to
move one into the Care Plan panel.

## Notes / known limitations (intentional, this is a PoC)

- Speaker label comes from which browser tab/role joined the call, not from
  audio-based diarization — the real production pipeline (or a closer PoC)
  would need actual diarization for single-mic room audio, e.g. in-clinic use.
- No persistence — refreshing the page loses everything. This is meant to be
  screen-recorded live, not used as a real app.
- Groq's hosted Whisper is used here purely for setup simplicity. It is not
  the model recommended for production (see prior discussion on Qwen3-ASR for
  Hindi-heavy audio).
