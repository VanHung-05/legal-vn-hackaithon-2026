"""Shared I/O utilities for loading and saving pipeline data."""

import csv
import json
from pathlib import Path

from schemas import Article, Question, Prediction


def load_articles_jsonl(path: Path) -> list[Article]:
    """Load flattened articles from JSONL."""
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            articles.append(Article(
                aid=row["aid"],
                law_id=row["law_id"],
                text=row["text"],
                text_truncated=row.get("text_truncated", row["text"]),
                char_len=row.get("char_len", len(row["text"])),
            ))
    return articles


def load_articles_map(path: Path) -> dict[int, Article]:
    """Load articles as aid -> Article mapping."""
    return {a.aid: a for a in load_articles_jsonl(path)}


def load_questions(path: Path) -> list[Question]:
    """Load questions from JSON (train or test)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Question(
            qid=item["qid"],
            question=item["question"],
            relevant_laws=item.get("relevant_laws", []),
        )
        for item in data
    ]


def load_questions_csv(path: Path) -> list[Question]:
    """Load questions from CSV."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(Question(
                qid=int(row["qid"]),
                question=row["question"],
            ))
    return questions


def find_test_file(data_dir: Path) -> Path:
    """Find test file — supports JSON/CSV and various naming conventions."""
    candidates = [
        data_dir / "private_test.json",
        data_dir / "public_test.json",
        data_dir / "test.json",
        data_dir / "private_test.csv",
        data_dir / "public_test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    for f in data_dir.iterdir():
        if "test" in f.name.lower() and f.suffix in (".json", ".csv"):
            return f

    raise FileNotFoundError(f"No test file found in {data_dir}")


def load_test_data(data_dir: Path) -> list[Question]:
    """Auto-detect and load test data."""
    path = find_test_file(data_dir)
    if path.suffix == ".json":
        return load_questions(path)
    return load_questions_csv(path)


def save_predictions_json(predictions: list[Prediction], path: Path) -> None:
    """Save predictions as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in predictions], f, ensure_ascii=False, indent=2)


def save_predictions_csv(predictions: list[Prediction], path: Path) -> None:
    """Save predictions as CSV for submission."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "relevant_laws"])
        for pred in predictions:
            aids_str = ",".join(str(a) for a in pred.relevant_laws)
            writer.writerow([pred.qid, aids_str])
