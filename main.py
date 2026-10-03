"""Visual question answering API: upload an image, ask a question in plain English, get answers.

    GET  /          web page to try it
    POST /ask       multipart form: image (file), question (text) -> answers with confidence
    GET  /health    readiness check (the model is loaded at start-up)
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

import model

MAX_BYTES = int(os.getenv("MAX_IMAGE_MB", "8")) * 1024 * 1024
PAGE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("VQA_PRELOAD", "1") == "1":
        model.load()  # load once at start-up so the first question is fast
    yield


app = FastAPI(title="Visual Question Answering API", version="2.0.0", lifespan=lifespan,
              description="Ask questions about an image in plain English. ViLT transformer, FastAPI, Docker.")


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE)


@app.get("/health")
def health():
    return {"status": "ok", "model": model.MODEL_ID}


@app.post("/ask")
def ask(image: UploadFile = File(...), question: str = Form(..., min_length=2, max_length=200), top_k: int = Form(5, ge=1, le=10)):
    question = question.strip()
    if len(question) < 2:
        raise HTTPException(422, "Ask a question of at least two characters.")
    data = image.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Images are limited to {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "That file is not an image this API can read (JPEG, PNG, WebP, GIF, BMP).") from exc
    started = time.perf_counter()
    answers = model.answer(img, question, top_k)
    return {"question": question, "answer": answers[0]["answer"], "confidence": answers[0]["score"], "answers": answers,
            "latency_ms": round((time.perf_counter() - started) * 1000), "model": model.MODEL_ID}
