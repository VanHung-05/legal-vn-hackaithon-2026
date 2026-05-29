#!/usr/bin/env python3
"""Download large artifacts from Hugging Face Dataset."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    ROOT_DIR,
    CORPUS_FILE,
    ARTICLES_FILE,
    CORPUS_EMBEDDINGS_FILE,
    CORPUS_AIDS_FILE,
    QDRANT_DIR,
    HF_DATASET_REPO,
)

# HF path (trong dataset repo) → local path
ARTIFACTS = {
    "data/legal_corpus.json": CORPUS_FILE,
    "data/processed/articles.jsonl": ARTICLES_FILE,
    "data/processed/embeddings/corpus_embeddings.npy": CORPUS_EMBEDDINGS_FILE,
    "data/processed/embeddings/corpus_aids.npy": CORPUS_AIDS_FILE,
}

QDRANT_HF_PREFIX = "qdrant_data"


def download_file(repo_id: str, hf_path: str, local_path: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    if local_path.exists():
        print(f"[SKIP] {local_path.relative_to(ROOT_DIR)}")
        return

    print(f"[DOWN] {hf_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=hf_path,
        local_dir=ROOT_DIR,
        token=token,
    )
    print(f"    → {local_path.relative_to(ROOT_DIR)}")


def download_qdrant(repo_id: str, token: str | None) -> None:
    from huggingface_hub import list_repo_files, hf_hub_download

    if QDRANT_DIR.exists() and any(QDRANT_DIR.rglob("*")):
        print(f"[SKIP] {QDRANT_DIR.relative_to(ROOT_DIR)}/")
        return

    print(f"[DOWN] {QDRANT_HF_PREFIX}/ → {QDRANT_DIR.relative_to(ROOT_DIR)}/")
    files = list_repo_files(repo_id, repo_type="dataset", token=token)
    qdrant_files = [f for f in files if f.startswith(f"{QDRANT_HF_PREFIX}/")]

    if not qdrant_files:
        print("[WARN] Không tìm thấy qdrant_data/ trên HF. Chạy: python3 build_qdrant.py")
        return

    for hf_path in qdrant_files:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=hf_path,
            local_dir=ROOT_DIR,
            token=token,
        )
    print(f"       {len(qdrant_files)} files downloaded")


def main():
    parser = argparse.ArgumentParser(description="Download artifacts from Hugging Face")
    parser.add_argument("--repo", default=HF_DATASET_REPO, help="HF dataset repo id")
    parser.add_argument("--token", default=None, help="HF token (private repo)")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip qdrant_data (rebuild locally)")
    parser.add_argument("--only-qdrant", action="store_true", help="Only download qdrant_data")
    args = parser.parse_args()

    print(f"Dataset: {args.repo}\n")

    if not args.only_qdrant:
        for hf_path, local_path in ARTIFACTS.items():
            download_file(args.repo, hf_path, local_path, args.token)

    if not args.skip_qdrant:
        download_qdrant(args.repo, args.token)

    print("\n[OK] Download complete.")
    print("Verify: cd src && python3 validate_embeddings.py")


if __name__ == "__main__":
    main()
