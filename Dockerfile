# Visual question answering API: FastAPI + ViLT, CPU-only image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU build of PyTorch keeps the image far smaller than the default CUDA wheels.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the model weights into the image so a container starts without a download.
RUN python -c "from transformers import ViltProcessor, ViltForQuestionAnswering as M; m='dandelin/vilt-b32-finetuned-vqa'; ViltProcessor.from_pretrained(m); M.from_pretrained(m)"

COPY main.py model.py ./
COPY static ./static

RUN useradd --create-home app && chown -R app /app /opt/hf
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
