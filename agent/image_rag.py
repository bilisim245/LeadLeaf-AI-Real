"""
Görsel RAG sorgulama: bir embedding vektörüne en benzer referans görselleri bulur (kosinüs
benzerliği). rag/build_image_index.py çalıştırılmamışsa boş liste döner; "benzer_gorseller"
alanı boş kalır.
"""
from __future__ import annotations

import os

import numpy as np

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "rag", "image_embeddings.npz")

_veri = None
_hazir = False


def _yukle() -> bool:
    global _veri, _hazir
    if _hazir:
        return True
    if not os.path.exists(INDEX_PATH):
        return False
    try:
        _veri = np.load(INDEX_PATH, allow_pickle=True)
        _hazir = True
        return True
    except Exception as e:
        print(f"[Görsel RAG] index yüklenemedi ({e})")
        return False


def _kosinus_benzerligi(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_similar(query_embedding: np.ndarray, k: int = 3) -> list[dict]:
    """query_embedding: inference/app.py'de aynı modelin GAP katmanından çıkan vektör."""
    if not _yukle():
        return []

    benzerlikler = [
        {
            "sinif": str(_veri["labels"][i]),
            "dosya": str(_veri["dosyalar"][i]),
            "benzerlik": round(_kosinus_benzerligi(query_embedding, _veri["embeddings"][i]) * 100, 1),
        }
        for i in range(len(_veri["labels"]))
    ]
    benzerlikler.sort(key=lambda x: x["benzerlik"], reverse=True)
    return benzerlikler[:k]


def index_var_mi() -> bool:
    return os.path.exists(INDEX_PATH)
