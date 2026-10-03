"""Visual question answering with ViLT (Vision-and-Language Transformer).

ViLT reads the image patches and the question tokens in one transformer, then scores
3,129 candidate answers learned from the VQA v2 dataset. It was trained with a binary
cross-entropy loss over soft answer scores, so each answer gets an independent sigmoid
probability; the top few are returned with their confidence.
"""

from __future__ import annotations

import os
from functools import lru_cache

from PIL import Image

MODEL_ID = os.getenv("VQA_MODEL", "dandelin/vilt-b32-finetuned-vqa")


@lru_cache(maxsize=1)
def load():
    """Load the processor and model once per process (about 470 MB, cached by Hugging Face)."""
    from transformers import ViltForQuestionAnswering, ViltProcessor

    processor = ViltProcessor.from_pretrained(MODEL_ID)
    model = ViltForQuestionAnswering.from_pretrained(MODEL_ID).eval()
    return processor, model


def answer(image: Image.Image, question: str, top_k: int = 5) -> list[dict]:
    """The model's top answers to a question about an image, best first."""
    import torch

    processor, model = load()
    encoding = processor(image.convert("RGB"), question, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**encoding).logits
    scores, indices = logits.sigmoid()[0].topk(top_k)
    return [{"answer": model.config.id2label[i.item()], "score": round(s.item(), 4)} for s, i in zip(scores, indices)]
