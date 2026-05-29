"""Retrieval evaluation metrics — Recall@K, MRR, F1 on train holdout."""

import json
import random
import time
from pathlib import Path

from config import TRAIN_FILE, OUTPUT_DIR, EVAL_HOLDOUT_RATIO, EVAL_SEED, EVAL_K_VALUES, TOP_K_CANDIDATES
from data_io import load_questions


def split_holdout(data: list, ratio: float = EVAL_HOLDOUT_RATIO, seed: int = EVAL_SEED):
    """Split data into (train_part, holdout)."""
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split = int(len(data) * (1 - ratio))
    return [data[i] for i in sorted(indices[:split])], [data[i] for i in sorted(indices[split:])]


def recall_at_k(true_aids: list[int], predicted: list[int], k: int) -> float:
    if not true_aids:
        return 1.0
    top_k = set(predicted[:k])
    return sum(1 for a in true_aids if a in top_k) / len(true_aids)


def hit_at_k(true_aids: list[int], predicted: list[int], k: int) -> bool:
    return any(a in set(predicted[:k]) for a in true_aids)


def mrr(true_aids: list[int], predicted: list[int]) -> float:
    for rank, aid in enumerate(predicted, 1):
        if aid in true_aids:
            return 1.0 / rank
    return 0.0


def f1_set(true_aids: list[int], predicted: list[int]) -> float:
    if not true_aids and not predicted:
        return 1.0
    if not true_aids or not predicted:
        return 0.0
    tp = len(set(true_aids) & set(predicted))
    p = tp / len(set(predicted))
    r = tp / len(set(true_aids))
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate(
    holdout: list,
    search_fn,
    k_values: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate retrieval.

    Args:
        holdout: list of Question or dict with qid, question, relevant_laws
        search_fn: callable(question: str) -> list[int] (ranked aids)
    """
    k_values = k_values or EVAL_K_VALUES
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"hit@{k}": [] for k in k_values})
    metrics["mrr"] = []
    for k in [1, 3, 5]:
        metrics[f"f1@{k}"] = []

    failed = []
    times = []

    for item in holdout:
        qid = item.qid if hasattr(item, "qid") else item["qid"]
        question = item.question if hasattr(item, "question") else item["question"]
        true_aids = item.relevant_laws if hasattr(item, "relevant_laws") else item["relevant_laws"]

        t0 = time.time()
        predicted = search_fn(question)
        times.append(time.time() - t0)

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(true_aids, predicted, k))
            metrics[f"hit@{k}"].append(float(hit_at_k(true_aids, predicted, k)))

        metrics["mrr"].append(mrr(true_aids, predicted))
        for k in [1, 3, 5]:
            metrics[f"f1@{k}"].append(f1_set(true_aids, predicted[:k]))

        if not hit_at_k(true_aids, predicted, 5):
            failed.append({
                "qid": qid,
                "question": question[:120],
                "true_aids": true_aids,
                "top5_predicted": predicted[:5],
            })

    results = {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}
    results["avg_time_ms"] = sum(times) / len(times) * 1000 if times else 0
    results["queries_per_sec"] = len(times) / sum(times) if sum(times) > 0 else 0
    results["total_queries"] = len(holdout)
    results["failed_queries"] = failed

    if verbose:
        print(f"\n{'='*55}")
        print(f"Evaluation ({len(holdout)} queries)")
        print(f"{'='*55}")
        for k in k_values:
            print(f"  Recall@{k:2d}: {results[f'recall@{k}']:.4f}  |  Hit@{k:2d}: {results[f'hit@{k}']:.4f}")
        print(f"  MRR:     {results['mrr']:.4f}")
        print(f"  F1@1:    {results['f1@1']:.4f}  |  F1@3: {results['f1@3']:.4f}  |  F1@5: {results['f1@5']:.4f}")
        print(f"  Speed:   {results['avg_time_ms']:.1f} ms/query ({results['queries_per_sec']:.2f} q/s)")
        print(f"  Failed@5: {len(failed)}/{len(holdout)}")
        if failed:
            print("\n  Sample failures:")
            for fq in failed[:3]:
                print(f"    Q{fq['qid']}: {fq['question']}")
                print(f"      True: {fq['true_aids']}  Got: {fq['top5_predicted']}")

    return results


def main():
    from dense_search import DenseSearcher

    print("=== Retrieval Evaluation ===")
    train_data = load_questions(TRAIN_FILE)
    _, holdout = split_holdout(train_data)
    print(f"Holdout: {len(holdout)} queries")

    searcher = DenseSearcher()
    results = evaluate(
        holdout,
        search_fn=lambda q: searcher.search_aids(q, top_k=TOP_K_CANDIDATES),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in results.items() if k != "failed_queries"}
    with open(OUTPUT_DIR / "eval_dense_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUTPUT_DIR / "eval_dense_failures.json", "w", encoding="utf-8") as f:
        json.dump(results["failed_queries"], f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Results → {OUTPUT_DIR / 'eval_dense_results.json'}")

    searcher.close()


if __name__ == "__main__":
    main()
