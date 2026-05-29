"""Dense vector search — BGE-M3 query embedding + Qdrant similarity search."""

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from config import (
    QDRANT_DIR, QDRANT_COLLECTION, BGE_M3_MODEL,
    MAX_SEQ_LENGTH, TOP_K_CANDIDATES, HNSW_EF_SEARCH,
)
from device import get_device
from schemas import SearchResult


class DenseSearcher:
    """Dense retrieval using BGE-M3 embeddings and Qdrant."""

    def __init__(self, qdrant_path: str | None = None, load_model: bool = True):
        self.qdrant_path = qdrant_path or str(QDRANT_DIR)
        self.client = QdrantClient(path=self.qdrant_path)
        self.model = None
        self._device = get_device()
        if load_model:
            self.load_model()

    def load_model(self) -> None:
        """Load BGE-M3 for query embedding."""
        from sentence_transformers import SentenceTransformer
        print(f"Loading BGE-M3 on {self._device}...")
        self.model = SentenceTransformer(BGE_M3_MODEL, device=self._device)
        self.model.max_seq_length = MAX_SEQ_LENGTH
        print("Model ready.")

    def embed_query(self, query: str) -> list[float]:
        vec = self.model.encode(
            [query], batch_size=1, show_progress_bar=False, normalize_embeddings=True,
        )
        return vec[0].tolist()

    def embed_queries(self, queries: list[str], batch_size: int = 32) -> list[list[float]]:
        vecs = self.model.encode(
            queries, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True,
        )
        return vecs.tolist()

    def search(self, query: str, top_k: int = TOP_K_CANDIDATES) -> list[SearchResult]:
        """Search for relevant articles. Returns ranked SearchResult list."""
        return self.search_by_vector(self.embed_query(query), top_k)

    def search_by_vector(self, query_vector: list[float], top_k: int = TOP_K_CANDIDATES) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=HNSW_EF_SEARCH, exact=False),
        )
        return [
            SearchResult(
                aid=hit.id,
                score=hit.score,
                law_id=hit.payload.get("law_id", ""),
                text=hit.payload.get("text", ""),
            )
            for hit in response.points
        ]

    def search_aids(self, query: str, top_k: int = TOP_K_CANDIDATES) -> list[int]:
        """Convenience: return only aid list (for Person B integration)."""
        return [r.aid for r in self.search(query, top_k)]

    def batch_search(self, queries: list[str], top_k: int = TOP_K_CANDIDATES) -> list[list[SearchResult]]:
        vectors = self.embed_queries(queries)
        return [self.search_by_vector(v, top_k) for v in vectors]

    def close(self) -> None:
        self.client.close()


def main():
    """Quick smoke test on 5 train questions."""
    import json
    from config import TRAIN_FILE

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train = json.load(f)

    searcher = DenseSearcher()
    print("\n=== Dense Search Smoke Test ===")
    for item in train[:5]:
        results = searcher.search(item["question"], top_k=10)
        predicted = [r.aid for r in results]
        true_aids = item["relevant_laws"]
        hit = any(a in predicted for a in true_aids)
        scores = [f"{r.score:.4f}" for r in results[:3]]
        print(f"\nQ{item['qid']}: {item['question'][:70]}...")
        print(f"  True: {true_aids}  |  Top-3: {predicted[:3]}  |  Hit@10: {hit}  |  Scores: {scores}")

    searcher.close()


if __name__ == "__main__":
    main()
