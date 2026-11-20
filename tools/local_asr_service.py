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
import numpy as np
import soundfile as sf
import sherpa_onnx
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="Local SenseVoice ASR Server for Cicada")

MODEL_DIR = os.path.expanduser("~/.cache/asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
MODEL_PATH = os.path.join(MODEL_DIR, "model.int8.onnx")
TOKENS_PATH = os.path.join(MODEL_DIR, "tokens.txt")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

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
            return JSONResponse({"text": "", "language": "zh"})

        with io.BytesIO(content) as bio:
            audio_data, sample_rate = sf.read(bio, dtype="float32")

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Silence check
        energy = np.mean(np.abs(audio_data)) if len(audio_data) > 0 else 0
        if energy < 1e-4:
            return JSONResponse({"text": "", "language": "zh"})

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
        print(f"Transcription error: {e}", file=sys.stderr)
        return JSONResponse({"text": "", "error": str(e), "language": "zh"}, status_code=200)

@app.get("/health")
@app.get("/")
async def health():
    return {"status": "ok", "model": "sense-voice-int8"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Local ASR Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
