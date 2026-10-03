# Visual Question Answering API (ViLT + FastAPI + Docker)

![CI](https://github.com/samiulhuda360/visual-question-answering-api/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Model: ViLT](https://img.shields.io/badge/model-ViLT%20(Hugging%20Face)-yellow)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)

**Upload a photo, ask a question about it in plain English, get an answer with a confidence score.** A
multimodal AI service: a vision-and-language transformer (ViLT) served through a FastAPI REST API, with a
small web page to try it and a Docker image that runs anywhere.

![Asking "How many dogs are there?" about a photo of two puppies: the answer is 2 at 92% confidence, with the top five answers and their scores](docs/screenshot.png)

## What it does

- **Visual question answering (VQA):** "How many people are there?", "What colour is the car?", "Is it raining?",
  "What is the man holding?" The model looks at the image and the question together.
- **Top answers with confidence:** returns the five most likely answers with their probabilities, not just one
  guess, so a client can decide when to trust it.
- **REST API + web UI:** `POST /ask` with a multipart image and question; interactive OpenAPI docs at `/docs`;
  a drag-and-drop page at `/`.
- **Production basics:** input validation (file type, size cap, question length), model loaded once at
  start-up, health check endpoint, non-root Docker user, model weights baked into the image, CI on every push.

## Run it

**Docker** (recommended):

```bash
docker compose up --build        # then open http://localhost:8000
```

**Locally**, Python 3.11:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --port 8000     # first start downloads the model (about 470 MB)
```

**API:**

```bash
curl -F "image=@puppies.jpg" -F "question=How many dogs are there?" http://localhost:8000/ask
```

```json
{
  "question": "How many dogs are there?",
  "answer": "2",
  "confidence": 0.9238,
  "answers": [{"answer": "2", "score": 0.9238}, {"answer": "1", "score": 0.0661}, {"answer": "3", "score": 0.0013}, "..."],
  "latency_ms": 299,
  "model": "dandelin/vilt-b32-finetuned-vqa"
}
```

Real answers on the photo above (CPU, about 0.3 s each):

| Question | Answer | Confidence |
|---|---|---|
| How many dogs are there? | 2 | 92% |
| What animal is this? | dog | 99% |
| What color are the dogs? | brown | 67% (then "tan", 24%) |
| Are they outside? | yes | 99.9% |
| What are the dogs lying on? | grass | 99% |

## How it works

[ViLT](https://arxiv.org/abs/2102.03334) (Kim et al., 2021) splits the image into patches and feeds them, together
with the question's tokens, through a single transformer: no separate object detector, which makes it fast on a
CPU. The checkpoint used here is fine-tuned on VQA v2 and scores 3,129 candidate answers; because it was trained
with binary cross-entropy over soft answer scores, each answer's sigmoid is read as its confidence.

```
image ──► 32x32 patches ─┐
                         ├─► ViLT transformer ─► 3,129 answer scores ─► top 5 + confidence
question ─► word tokens ─┘
```

**Limits:** answers come from a fixed vocabulary of short answers (VQA v2), so the model is best at short
factual questions (counting, colours, yes/no, objects) and cannot write long descriptions or read long text.
Swapping in a generative vision-language model (for example Qwen2-VL or LLaVA) behind the same API is the
natural next step.

## Project layout

```
main.py            FastAPI app: /ask, /health, the web page; validation and timing
model.py           ViLT loading (once) and top-k answering
static/index.html  drag-and-drop web UI (no build step)
tests/             API tests with the model mocked (no download in CI)
Dockerfile         CPU-only PyTorch, model baked in, non-root, health check
compose.yaml       one-command start
```

## Tech stack

Python · PyTorch · Hugging Face Transformers · ViLT · FastAPI · Uvicorn · Pillow · Docker · pytest · GitHub Actions

**Keywords:** multimodal AI, visual question answering, VQA, vision-language model, computer vision, NLP,
transformers, Hugging Face, model serving, ML deployment, REST API, FastAPI, Docker, MLOps.

Screenshot photo: [Two German Shepherd puppies](https://commons.wikimedia.org/wiki/File:Two_German_Shepherd_puppies.jpg), Wikimedia Commons, public domain. Model: `dandelin/vilt-b32-finetuned-vqa` (Apache 2.0).

## Author

[Samiul Huda](https://github.com/samiulhuda360) · MSc Data Science · Auckland, New Zealand.
