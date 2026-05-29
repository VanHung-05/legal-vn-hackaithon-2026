"""Build Qdrant vector index from pre-computed embeddings."""

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    OptimizersConfigDiff, HnswConfigDiff,
)

from config import (
    CORPUS_EMBEDDINGS_FILE, CORPUS_AIDS_FILE, ARTICLES_FILE,
    QDRANT_DIR, QDRANT_COLLECTION, EMBEDDING_DIM,
    HNSW_M, HNSW_EF_CONSTRUCT,
)
from data_io import load_articles_map


def build_index(
    qdrant_dir=None,
    batch_size: int = 500,
    payload_text_len: int = 1000,
) -> QdrantClient:
    """Build Qdrant index from saved embeddings. Returns open client."""
    qdrant_dir = qdrant_dir or str(QDRANT_DIR)

    print("Loading embeddings...")
    embeddings = np.load(CORPUS_EMBEDDINGS_FILE)
    aids = np.load(CORPUS_AIDS_FILE)
    print(f"  Embeddings: {embeddings.shape}, Aids: {aids.shape}")

    articles = load_articles_map(ARTICLES_FILE)
    print(f"  Articles: {len(articles)}")

    assert embeddings.shape[0] == len(aids), "Embedding/aids count mismatch"
    assert embeddings.shape[1] == EMBEDDING_DIM, f"Expected dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"

    print(f"Initializing Qdrant → {qdrant_dir}")
    client = QdrantClient(path=qdrant_dir)

    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
    )

    total = len(aids)
    print(f"Uploading {total} vectors...")
    for i in tqdm(range(0, total, batch_size), desc="Uploading"):
        batch_aids = aids[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]

        points = []
        for aid, emb in zip(batch_aids, batch_embs):
            aid_int = int(aid)
            article = articles.get(aid_int)
            points.append(PointStruct(
                id=aid_int,
                vector=emb.tolist(),
                payload={
                    "aid": aid_int,
                    "law_id": article.law_id if article else "",
                    "text": (article.text[:payload_text_len] if article else ""),
                },
            ))
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"[OK] Index built: {info.points_count} points, dim={info.config.params.vectors.size}")
    return client


def main():
    print("=== Building Qdrant Index ===")
    client = build_index()
    client.close()
    print("=== Done ===")


if __name__ == "__main__":
    main()
