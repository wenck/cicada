# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "python-multipart",
#     "sherpa-onnx>=1.10.0",
#     "soundfile",
#     "numpy",
# ]
# ///

import io
import os
import re
import sys
import threading
import numpy as np
import soundfile as sf
import sherpa_onnx
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="Local SenseVoice ASR Server for Cicada")

PORT = int(os.environ.get("CICADA_ASR_PORT", "8000"))

MODEL_DIR = os.path.expanduser("~/.cache/asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
MODEL_PATH = os.path.join(MODEL_DIR, "model.int8.onnx")
TOKENS_PATH = os.path.join(MODEL_DIR, "tokens.txt")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

# Owner PIDs (Cicada apps that want this service up). The watchdog exits the process when every
# owner is gone, however it went away: a quit, SIGTERM, kill -9 or a crash all end the owner PID.
# Started without an owner (run by hand) it never exits on its own.
owners = set()
had_owner = False
owners_lock = threading.Lock()


def add_owner(pid):
    global had_owner
    with owners_lock:
        owners.add(pid)
        had_owner = True


def del_owner(pid):
    with owners_lock:
        owners.discard(pid)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def watch_owners():
    while True:
        threading.Event().wait(2)
        with owners_lock:
            dead = [p for p in owners if not pid_alive(p)]
            for p in dead:
                owners.discard(p)
                print(f"owner {p} is gone", file=sys.stderr)
            empty_after_owner = had_owner and not owners
        if empty_after_owner:
            print("last owner gone, exiting", file=sys.stderr)
            os._exit(0)


initial_owner = os.environ.get("CICADA_OWNER_PID")
if initial_owner:
    add_owner(int(initial_owner))

threading.Thread(target=watch_owners, daemon=True).start()

print(f"Loading SenseVoice model from {MODEL_DIR}...")
recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=MODEL_PATH,
    tokens=TOKENS_PATH,
    num_threads=4,
    use_itn=True,
)
print("SenseVoice ASR model loaded successfully!")

TAG_PATTERN = re.compile(r"<\|.*?\|>")

def clean_sense_voice_output(text: str) -> str:
    # Remove special tags like <|NEUTRAL|>, <|HAPPY|>, <|zh|>, etc.
    cleaned = TAG_PATTERN.sub("", text).strip()
    return cleaned

@app.post("/audio/transcriptions")
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form(None),
    temperature: float = Form(0.0),
):
    try:
        content = await file.read()
        if not content:
            return JSONResponse({"text": ""})

        with io.BytesIO(content) as bio:
            audio_data, sample_rate = sf.read(bio, dtype="float32")

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Silence check: nothing heard is an empty transcript, not a guessed language.
        energy = np.mean(np.abs(audio_data)) if len(audio_data) > 0 else 0
        if energy < 1e-4:
            return JSONResponse({"text": ""})

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_data)
        recognizer.decode_stream(stream)

        raw_text = stream.result.text or ""
        text = clean_sense_voice_output(raw_text)

        # Clean language code: <|zh|> -> zh
        raw_lang = getattr(stream.result, "lang", None) or "zh"
        lang = raw_lang.replace("<|", "").replace("|>", "").strip() or "zh"

        if text:
            print(f"[ASR ({lang})]: {text}")

        return JSONResponse({"text": text, "language": lang})
    except Exception as e:
        # An error is a 500: answering 200 with empty text would keep the caller's backup
        # model from ever taking over.
        print(f"Transcription error: {e}", file=sys.stderr)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/owner")
async def owner_add(pid: int):
    add_owner(pid)
    return {"owners": sorted(owners)}

@app.delete("/owner")
async def owner_del(pid: int):
    del_owner(pid)
    return {"owners": sorted(owners)}

@app.get("/health")
@app.get("/")
async def health():
    return {"status": "ok", "service": "cicada-local-asr", "model": "sense-voice-int8", "owners": sorted(owners)}

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Local ASR Server on http://127.0.0.1:{PORT} ...")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
