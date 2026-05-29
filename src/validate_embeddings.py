"""Validate embedding files before building Qdrant index."""

import sys
import numpy as np
from config import CORPUS_EMBEDDINGS_FILE, CORPUS_AIDS_FILE, EMBEDDING_DIM, ARTICLES_FILE
from data_io import load_articles_jsonl


def validate() -> bool:
    errors = []

    if not CORPUS_EMBEDDINGS_FILE.exists():
        errors.append(f"Missing: {CORPUS_EMBEDDINGS_FILE}")
    if not CORPUS_AIDS_FILE.exists():
        errors.append(f"Missing: {CORPUS_AIDS_FILE}")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return False

    embeddings = np.load(CORPUS_EMBEDDINGS_FILE)
    aids = np.load(CORPUS_AIDS_FILE)
    articles = load_articles_jsonl(ARTICLES_FILE)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Aids count:       {len(aids)}")
    print(f"Articles count:   {len(articles)}")
    print(f"File size:        {CORPUS_EMBEDDINGS_FILE.stat().st_size / 1024**2:.1f} MB")

    if embeddings.shape[0] != len(aids):
        errors.append(f"Row mismatch: embeddings={embeddings.shape[0]}, aids={len(aids)}")
    if embeddings.shape[0] != len(articles):
        errors.append(f"Row mismatch: embeddings={embeddings.shape[0]}, articles={len(articles)}")
    if embeddings.shape[1] != EMBEDDING_DIM:
        errors.append(f"Dim mismatch: got {embeddings.shape[1]}, expected {EMBEDDING_DIM}")
    if np.isnan(embeddings).any():
        errors.append("Embeddings contain NaN values")
    if np.isinf(embeddings).any():
        errors.append("Embeddings contain Inf values")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return False

    print("[OK] Embeddings valid — ready to build Qdrant index")
    return True


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
