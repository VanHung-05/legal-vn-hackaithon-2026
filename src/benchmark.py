"""Benchmark inference speed for dense retrieval."""

import json
import time
from pathlib import Path

from config import TRAIN_FILE, OUTPUT_DIR, TOP_K_CANDIDATES
from data_io import load_questions
from evaluate import split_holdout


def benchmark(
    searcher,
    questions: list,
    top_k: int = TOP_K_CANDIDATES,
    warmup: int = 3,
) -> dict:
    """
    Measure end-to-end retrieval speed.

    Returns dict with avg_ms, qps, p50_ms, p95_ms.
    """
    texts = [q.question if hasattr(q, "question") else q["question"] for q in questions]

    # Warmup
    for text in texts[:warmup]:
        searcher.search(text, top_k=top_k)

    times = []
    for text in texts:
        t0 = time.perf_counter()
        searcher.search(text, top_k=top_k)
        times.append(time.perf_counter() - t0)

    times.sort()
    n = len(times)
    return {
        "total_queries": n,
        "top_k": top_k,
        "avg_ms": sum(times) / n * 1000,
        "p50_ms": times[n // 2] * 1000,
        "p95_ms": times[int(n * 0.95)] * 1000,
        "min_ms": times[0] * 1000,
        "max_ms": times[-1] * 1000,
        "queries_per_sec": n / sum(times),
        "total_sec": sum(times),
    }


def main():
    from dense_search import DenseSearcher

    print("=== Inference Speed Benchmark ===")
    train_data = load_questions(TRAIN_FILE)
    _, holdout = split_holdout(train_data)
    print(f"Benchmarking on {len(holdout)} holdout queries...")

    searcher = DenseSearcher()
    stats = benchmark(searcher, holdout)

    print(f"\n{'='*40}")
    print(f"  Avg:  {stats['avg_ms']:.1f} ms/query")
    print(f"  P50:  {stats['p50_ms']:.1f} ms")
    print(f"  P95:  {stats['p95_ms']:.1f} ms")
    print(f"  QPS:  {stats['queries_per_sec']:.2f} queries/sec")
    print(f"{'='*40}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "benchmark_dense.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Saved → {out_path}")

    searcher.close()


if __name__ == "__main__":
    main()
