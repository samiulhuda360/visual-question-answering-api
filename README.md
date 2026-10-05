# Visual Question Answering API (ViLT + FastAPI + Docker)

![CI](https://github.com/samiulhuda360/visual-question-answering-api/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Model: ViLT](https://img.shields.io/badge/model-ViLT%20(Hugging%20Face)-yellow)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)

**Upload a photo, ask a question about it in plain English, and get the top answers with a confidence score for
each.** A multimodal AI service: the ViLT vision-and-language transformer served through a FastAPI REST API, with a
drag-and-drop web page and a Docker image that has the model weights built in. It is for developers who want a small,
production-shaped example of serving a vision-language model, and for anyone who wants to try visual question
answering on their own photos.

![The web page asking "How many dogs are there?" about a photo of two puppies: the answer is 2 at 92% confidence, with the top five answers and their scores](docs/screenshot.png)

## Key features

- **Visual question answering:** "How many people are there?", "What colour is the car?", "Is it raining?" The model
  reads the image and the question together.
- **Top answers with confidence:** five answers by default (`top_k` from 1 to 10), each with its score, so a client can
  decide when to trust the result.
- **REST API and web UI:** `POST /ask` takes a multipart image and question, `GET /health` is a readiness check,
  interactive OpenAPI docs are at `/docs`, and a drag-and-drop page with example questions is at `/`.
- **Input validation:** questions of 2 to 200 characters, an image size cap (8 MB by default) and a clear error for
  files that aren't readable images.
- **Model loaded once at start-up,** so the first question doesn't wait for it. Every response reports how long the
  model took.
- **Docker image:** CPU-only PyTorch, model weights downloaded at build time, a non-root user and a health check.
  `docker compose up` starts it.
- **CI on every push:** API tests with the model mocked, plus a Docker job that builds the image, waits for `/health`
  and asks a real question.

## Architecture

```mermaid
flowchart TD
    BR["Browser<br/>drag-and-drop page"] -->|"GET / and POST /ask"| API
    CL["API client<br/>curl or another service"] -->|"POST /ask"| API
    DK["Docker health check<br/>every 30 s"] -.->|"GET /health"| API
    API["FastAPI app, main.py<br/>routes, input validation,<br/>timing of the model call"] --> ANS["answer() in model.py"]
    HUB["Hugging Face checkpoint<br/>dandelin/vilt-b32-finetuned-vqa"] -.->|"load() once at start-up"| ANS
    ANS --> PROC["ViltProcessor<br/>image to pixel values<br/>question to tokens"]
    PROC --> VILT["ViltForQuestionAnswering<br/>image patches and tokens<br/>in one transformer"]
    VILT --> TOPK["sigmoid over 3,129 answers<br/>top_k answers with scores"]
```

`main.py` holds the web layer: routes, validation and timing. `model.py` is the only module that touches the model:
`load()` builds the processor and model once per process (cached with `lru_cache`, in evaluation mode) and `answer()`
runs one question. The Docker image downloads the checkpoint at build time into `HF_HOME=/opt/hf`, so a container
starts without a download.

## How it works

```mermaid
sequenceDiagram
    participant C as Browser or API client
    participant A as FastAPI POST /ask
    participant M as answer() in model.py
    participant P as ViltProcessor
    participant V as ViLT model
    C->>A: multipart form with image, question, top_k
    A->>A: question must be 2 to 200 characters, else 422
    A->>A: upload over MAX_IMAGE_MB gives 413
    A->>A: open with Pillow, unreadable file gives 400
    A->>M: answer(image, question, top_k)
    M->>P: RGB image and question text
    P-->>M: pixel values and token ids
    M->>V: forward pass without gradients
    V-->>M: 3,129 answer logits
    M->>M: sigmoid, then the top_k highest scores
    M-->>A: answers with scores, best first
    A-->>C: JSON with answer, confidence, answers, latency_ms, model
```

1. **Start-up.** FastAPI's lifespan hook calls `model.load()`, which downloads (or reads from the cache) the processor
   and the model and keeps them in memory. Setting `VQA_PRELOAD=0` skips this, and the model then loads on the first
   question.
2. **Validation.** `POST /ask` strips the question and rejects anything under 2 or over 200 characters (422). It reads
   at most `MAX_IMAGE_MB` plus one byte of the upload and rejects larger files (413), then opens the bytes with Pillow
   and rejects anything it can't decode (400).
3. **Encoding.** The processor resizes the image so its shorter side is 384 pixels and normalises it, and turns the
   question into lower-case BERT word-piece tokens, truncated to the model's 40-token limit.
4. **Inference.** [ViLT](https://arxiv.org/abs/2102.03334) (Kim et al., 2021) cuts the image into 32 x 32 patches and
   feeds them, together with the question tokens, through a single transformer, with no convolutional backbone or
   object detector. Its question-answering head scores 3,129 candidate answers.
5. **Scoring.** The checkpoint was trained with binary cross-entropy over soft answer scores, so each answer's sigmoid
   is read as its confidence. The `top_k` highest are returned, best first.
6. **Response.** `answer` and `confidence` are the top answer and its score, `answers` lists all `top_k`,
   `latency_ms` is the time spent in the model, and `model` names the checkpoint.

### Model and scope

The default checkpoint, `dandelin/vilt-b32-finetuned-vqa`, is ViLT fine-tuned on VQA v2: it chooses each answer from
3,129 short answers. It is best at short factual answers: counting, colours, yes/no and naming objects. Long
descriptions and reading longer text from an image fall outside that answer set. Scores are independent sigmoid
outputs, so the scores of the top answers need not add up to 1. `VQA_MODEL` selects another ViLT
question-answering checkpoint from the Hugging Face Hub.

## Screenshots

| Web page | API docs |
|---|---|
| ![The web page: a photo of two puppies, the question "How many dogs are there?", the answer 2 at 92% confidence in 272 ms, and bars for the top five answers](docs/screenshot.png) | ![The interactive OpenAPI docs at /docs: GET /health, and POST /ask with a multipart body of image, question (2 to 200 characters) and top_k (1 to 10)](docs/api-docs.png) |
| Drop or choose a photo, type a question or click an example, and read the answer, its confidence, the latency and a bar for each of the top five answers. | `/docs` documents the API endpoints and lets you send a request from the browser with **Try it out**. |

## Tech stack

Python 3.11 · PyTorch (CPU build) · Hugging Face Transformers · ViLT · FastAPI · Uvicorn · Pillow · Docker and
Docker Compose · pytest · GitHub Actions

**Keywords:** multimodal AI, visual question answering, VQA, vision-language model, computer vision, NLP,
transformers, Hugging Face, model serving, ML deployment, REST API, FastAPI, Docker, MLOps.

## Getting started

### Prerequisites

- Docker with Docker Compose, or Python 3.11 (the version CI and the Docker image use).
- About 470 MB of disk space for the model weights, which are downloaded on the first build or start.

### Run with Docker

```bash
git clone https://github.com/samiulhuda360/visual-question-answering-api
cd visual-question-answering-api
docker compose up --build        # then open http://localhost:8000
```

The build installs the CPU build of PyTorch and downloads the model weights into the image. Compose publishes port
8000 and restarts the container unless you stop it.

### Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --port 8000     # the first start downloads the model, about 470 MB
```

Then open http://localhost:8000.

### Configuration

All settings are optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `VQA_MODEL` | `dandelin/vilt-b32-finetuned-vqa` | Hugging Face ID of the ViLT question-answering checkpoint |
| `MAX_IMAGE_MB` | `8` | largest accepted upload, in MB |
| `VQA_PRELOAD` | `1` | `1` loads the model at start-up; `0` loads it on the first question (the tests use `0`) |
| `HF_HOME` | Hugging Face's default cache | where the model weights are stored; the Docker image sets `/opt/hf` |

The Docker image contains the default checkpoint only, so a different `VQA_MODEL` is downloaded when the container
starts.

## Usage

### Web page

1. Open http://localhost:8000.
2. Drop an image on the left panel, or click it to choose a file.
3. Type a question, or click one of the example questions.
4. Read the answer, its confidence and the latency, with a bar for each of the top five answers.

### API

```bash
curl -F "image=@puppies.jpg" -F "question=How many dogs are there?" http://localhost:8000/ask
```

The response for the photo in the screenshot:

```json
{
  "question": "How many dogs are there?",
  "answer": "2",
  "confidence": 0.9238,
  "answers": [{"answer": "2", "score": 0.9238}, {"answer": "1", "score": 0.0661}, {"answer": "3", "score": 0.0013}, "..."],
  "latency_ms": 272,
  "model": "dandelin/vilt-b32-finetuned-vqa"
}
```

| Endpoint | Input | Output |
|---|---|---|
| `GET /` | | the web page |
| `GET /health` | | `{"status": "ok", "model": "dandelin/vilt-b32-finetuned-vqa"}` |
| `POST /ask` | multipart form: `image` (file, required), `question` (2 to 200 characters, required), `top_k` (1 to 10, default 5) | `question`, `answer`, `confidence`, `answers` (each with `answer` and `score`), `latency_ms`, `model` |
| `GET /docs` | | interactive OpenAPI docs |

| Status | When |
|---|---|
| `400` | the file isn't an image Pillow can read (JPEG, PNG, WebP, GIF, BMP) |
| `413` | the image is larger than `MAX_IMAGE_MB` |
| `422` | the question is missing, shorter than 2 or longer than 200 characters, or `top_k` is outside 1 to 10 |

## Project structure

```
.
├── main.py                    FastAPI app: GET /, GET /health and POST /ask, with validation and timing
├── model.py                   loads ViLT once and returns the top-k answers with scores
├── static/index.html          drag-and-drop web page (no build step)
├── tests/test_api.py          API tests with the model mocked
├── docs/                      README screenshots
├── Dockerfile                 CPU-only PyTorch, model weights built in, non-root user, health check
├── compose.yaml               one-command start on port 8000
├── requirements.txt           runtime dependencies (PyTorch is installed separately, CPU build)
├── requirements-dev.txt       test dependencies: pytest and httpx
├── pytest.ini                 test settings
└── .github/workflows/ci.yml   tests and a Docker smoke test on every push and pull request
```

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

The tests replace `model.answer` with a stub and set `VQA_PRELOAD=0`, so they run in seconds without PyTorch or the
model download. They cover:

- a question about an image returns the answer, its confidence and the list of answers (200)
- a file that isn't an image is rejected (400)
- a blank question is rejected (422)
- `/health` reports `ok` and `/` serves the web page

**CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request:

- **test:** Python 3.11, the CPU build of PyTorch, the requirements, then `pytest -q`.
- **docker:** builds the image, starts a container, polls `/health` (up to 60 tries, 3 seconds apart), then posts a
  generated 224 x 224 red PNG with the question "What color is this?" and checks that the JSON reply has an answer.

## Licence and credits

- Model: [`dandelin/vilt-b32-finetuned-vqa`](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa), Apache 2.0.
- Method: [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334),
  Kim et al., 2021.
- Screenshot photo: [Two German Shepherd puppies](https://commons.wikimedia.org/wiki/File:Two_German_Shepherd_puppies.jpg),
  Wikimedia Commons, public domain.

## Author

[Samiul Huda](https://github.com/samiulhuda360) · MSc Data Science · Auckland, New Zealand
