# api/endpoints/transcribe.py

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from api.services.whisper_service import load_whisper_model
import tempfile
import shutil
import time

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    asr_pipeline = load_whisper_model()

    if not file.filename.endswith((".mp3", ".wav", ".flac", ".m4a")):
        return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    start_time = time.time()
    result = asr_pipeline(tmp_path)
    elapsed_time = time.time() - start_time

    return {
        "text": result["text"],
        "time_elapsed": elapsed_time
    }
