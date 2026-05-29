"""Run inference on test set and write submission output."""

import time
from tqdm import tqdm

from config import DATA_DIR, OUTPUT_DIR, TOP_K_FINAL
from data_io import load_test_data, save_predictions_json, save_predictions_csv
from schemas import Prediction
from dense_search import DenseSearcher


def run_inference(searcher: DenseSearcher, test_data, top_k: int = TOP_K_FINAL) -> list[Prediction]:
    predictions = []
    for item in tqdm(test_data, desc="Inference"):
        results = searcher.search(item.question, top_k=top_k)
        predictions.append(Prediction(
            qid=item.qid,
            relevant_laws=[r.aid for r in results],
        ))
    return predictions


def main():
    print("=== Running Inference ===")

    test_data = load_test_data(DATA_DIR)
    print(f"Test questions: {len(test_data)}")

    searcher = DenseSearcher()
    t0 = time.time()
    predictions = run_inference(searcher, test_data)
    elapsed = time.time() - t0

    print(f"Inference: {elapsed:.2f}s ({len(test_data)/elapsed:.2f} q/s)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions_json(predictions, OUTPUT_DIR / "pred.json")
    save_predictions_csv(predictions, OUTPUT_DIR / "pred.csv")

    searcher.close()
    print("=== Done ===")


if __name__ == "__main__":
    main()
