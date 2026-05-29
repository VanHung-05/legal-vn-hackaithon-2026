"""Embed legal corpus with BGE-M3 (dense vectors).

Backends:
  - sentence-transformers (default)
  - FlagEmbedding (--flag)
  - Google Colab (see embed_colab.ipynb)
"""

import sys
import time
import numpy as np
from tqdm import tqdm

from config import (
    ARTICLES_FILE,
    EMBEDDINGS_DIR,
    CORPUS_EMBEDDINGS_FILE,
    CORPUS_AIDS_FILE,
    BGE_M3_MODEL,
    EMBEDDING_BATCH_SIZE,
    MAX_SEQ_LENGTH,
)
from data_io import load_articles_jsonl
from device import get_device, use_fp16
CHECKPOINT_EVERY = 5000


def embed_sentence_transformers(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = get_device()
    model = SentenceTransformer(BGE_M3_MODEL, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    chunks = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        embs = model.encode(batch, batch_size=EMBEDDING_BATCH_SIZE,
                            show_progress_bar=False, normalize_embeddings=True)
        chunks.append(embs)
        if i > 0 and (i + EMBEDDING_BATCH_SIZE) % CHECKPOINT_EVERY < EMBEDDING_BATCH_SIZE:
            _save_checkpoint(np.vstack(chunks), i + len(batch))
    return np.vstack(chunks)


def embed_flagembedding(texts: list[str]) -> np.ndarray:
    from FlagEmbedding import BGEM3FlagModel

    device = get_device()
    model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=use_fp16(device))

    chunks = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        output = model.encode(
            batch, batch_size=EMBEDDING_BATCH_SIZE, max_length=MAX_SEQ_LENGTH,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        chunks.append(output["dense_vecs"])
        if i > 0 and (i + EMBEDDING_BATCH_SIZE) % CHECKPOINT_EVERY < EMBEDDING_BATCH_SIZE:
            _save_checkpoint(np.vstack(chunks), i + len(batch))
    return np.vstack(chunks)


def _save_checkpoint(embeddings: np.ndarray, n_done: int) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = EMBEDDINGS_DIR / f"checkpoint_{n_done}.npy"
    np.save(path, embeddings)
    print(f"\n  [Checkpoint] {n_done} embeddings → {path}")


def _find_checkpoint() -> tuple[int, np.ndarray | None]:
    if not EMBEDDINGS_DIR.exists():
        return 0, None
    checkpoints = sorted(EMBEDDINGS_DIR.glob("checkpoint_*.npy"))
    if not checkpoints:
        return 0, None
    latest = checkpoints[-1]
    n_done = int(latest.stem.split("_")[1])
    print(f"[Resume] checkpoint at {n_done} ({latest.name})")
    return n_done, np.load(latest)


def main():
    backend = "flagembedding" if "--flag" in sys.argv else "sentence-transformers"
    print(f"=== Embedding Corpus (backend: {backend}) ===")

    articles = load_articles_jsonl(ARTICLES_FILE)
    texts = [a.text_truncated or a.text for a in articles]
    aids = [a.aid for a in articles]

    for i, t in enumerate(texts):
        if not t.strip():
            texts[i] = f"Điều luật số {aids[i]}"

    start_idx, prev_embs = _find_checkpoint()
    if start_idx > 0:
        texts = texts[start_idx:]
        print(f"Resuming from article {start_idx}, {len(texts)} remaining")

    t0 = time.time()
    embed_fn = embed_flagembedding if backend == "flagembedding" else embed_sentence_transformers
    new_embs = embed_fn(texts)
    elapsed = time.time() - t0

    embeddings = np.vstack([prev_embs, new_embs]) if prev_embs is not None else new_embs
    print(f"Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} articles/sec)")
    print(f"Shape: {embeddings.shape}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CORPUS_EMBEDDINGS_FILE, embeddings.astype(np.float32))
    np.save(CORPUS_AIDS_FILE, np.array(aids))
    print(f"[OK] Saved → {CORPUS_EMBEDDINGS_FILE} ({embeddings.nbytes / 1024**2:.1f} MB)")
    print(f"[OK] Saved → {CORPUS_AIDS_FILE}")

    for ckpt in EMBEDDINGS_DIR.glob("checkpoint_*.npy"):
        ckpt.unlink()
    print("=== Done ===")


if __name__ == "__main__":
    main()
