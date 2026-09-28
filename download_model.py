"""Скачать intfloat/multilingual-e5-small в cache/ для офлайн-инференса."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer

from src.config import MODEL_DIR, MODEL_NAME
from src.retriever import HybridRetriever


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = HybridRetriever._torch_device()
    print(f"Скачиваю {MODEL_NAME} -> {MODEL_DIR} (device={device})")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.save(str(MODEL_DIR))
    print("Готово.")


if __name__ == "__main__":
    main()
