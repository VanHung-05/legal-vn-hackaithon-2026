"""Process legal corpus — flatten articles, normalize text, export metadata."""

import json
import re
import unicodedata

from config import CORPUS_FILE, PROCESSED_DIR, ARTICLES_FILE, AID_LAW_MAP_FILE, TRAIN_FILE, TRUNCATE_CHARS
from schemas import Article


def normalize_text(text: str) -> str:
    """Normalize Vietnamese legal text."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = TRUNCATE_CHARS) -> str:
    """Truncate overly long articles."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def flatten_corpus(corpus_path) -> list[Article]:
    """Flatten nested corpus JSON into individual articles."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    articles = []
    for doc in corpus:
        law_id = doc["law_id"]
        for article in doc["content"]:
            raw_text = normalize_text(article["content_Article"])
            articles.append(Article(
                aid=article["aid"],
                law_id=law_id,
                text=raw_text,
                text_truncated=truncate_text(raw_text),
                char_len=len(raw_text),
            ))
    return articles


def validate_corpus(articles: list[Article], train_path) -> None:
    """Ensure all train aids exist in corpus."""
    corpus_aids = {a.aid for a in articles}
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)

    train_aids = {aid for item in train for aid in item["relevant_laws"]}
    missing = train_aids - corpus_aids
    if missing:
        raise ValueError(f"Missing {len(missing)} aids in corpus: {list(missing)[:10]}")
    print(f"[OK] All {len(train_aids)} train aids exist in corpus")


def export_articles(articles: list[Article], output_path) -> None:
    """Export flattened articles as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
    print(f"[OK] Exported {len(articles)} articles → {output_path}")


def main():
    print("=== Processing Legal Corpus ===")

    articles = flatten_corpus(CORPUS_FILE)
    print(f"Total articles: {len(articles)}")

    lengths = sorted(a.char_len for a in articles)
    print(f"Text length: min={lengths[0]}, median={lengths[len(lengths)//2]}, max={lengths[-1]}")

    empty = sum(1 for a in articles if a.char_len == 0)
    if empty:
        print(f"[WARN] {empty} articles have empty text")

    validate_corpus(articles, TRAIN_FILE)
    export_articles(articles, ARTICLES_FILE)

    aid_map = {a.aid: a.law_id for a in articles}
    with open(AID_LAW_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(aid_map, f, ensure_ascii=False)
    print(f"[OK] Exported aid→law_id map → {AID_LAW_MAP_FILE}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
