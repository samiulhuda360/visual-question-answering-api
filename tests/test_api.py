import io
import os

os.environ["VQA_PRELOAD"] = "0"  # tests never download the model

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import main  # noqa: E402


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="PNG")
    return buf.getvalue()


def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(main.model, "answer", lambda img, q, k=5: [{"answer": "red", "score": 0.93}, {"answer": "orange", "score": 0.04}][:k])
    return TestClient(main.app)


def test_answers_a_question_about_an_image(monkeypatch):
    r = client(monkeypatch).post("/ask", files={"image": ("a.png", png(), "image/png")}, data={"question": "What color is it?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "red" and body["confidence"] == 0.93 and len(body["answers"]) == 2


def test_rejects_files_that_are_not_images(monkeypatch):
    r = client(monkeypatch).post("/ask", files={"image": ("a.txt", b"not an image", "text/plain")}, data={"question": "What is it?"})
    assert r.status_code == 400


def test_rejects_empty_questions(monkeypatch):
    r = client(monkeypatch).post("/ask", files={"image": ("a.png", png(), "image/png")}, data={"question": " "})
    assert r.status_code == 422


def test_health_and_page(monkeypatch):
    c = client(monkeypatch)
    assert c.get("/health").json()["status"] == "ok"
    assert "Ask the image" in c.get("/").text
