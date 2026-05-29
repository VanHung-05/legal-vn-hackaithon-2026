#!/usr/bin/env python3
"""Upload large artifacts to Hugging Face Dataset (chạy 1 lần bởi Người A)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    ROOT_DIR,
    CORPUS_FILE,
    ARTICLES_FILE,
    CORPUS_EMBEDDINGS_FILE,
    CORPUS_AIDS_FILE,
    QDRANT_DIR,
    HF_DATASET_REPO,
)

UPLOAD_MAP = {
    CORPUS_FILE: "data/legal_corpus.json",
    ARTICLES_FILE: "data/processed/articles.jsonl",
    CORPUS_EMBEDDINGS_FILE: "data/processed/embeddings/corpus_embeddings.npy",
    CORPUS_AIDS_FILE: "data/processed/embeddings/corpus_aids.npy",
}


def main():
    parser = argparse.ArgumentParser(description="Upload artifacts to Hugging Face Dataset")
    parser.add_argument("--repo", default=HF_DATASET_REPO, help="HF dataset repo id")
    parser.add_argument("--token", required=True, help="HF write token")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip qdrant_data upload")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)

    print(f"Creating dataset repo (if not exists): {args.repo}")
    create_repo(args.repo, repo_type="dataset", exist_ok=True, token=args.token)

    for local_path, hf_path in UPLOAD_MAP.items():
        if not local_path.exists():
            print(f"[ERROR] Missing: {local_path}")
            sys.exit(1)
        size_mb = local_path.stat().st_size / 1024**2
        print(f"[UPLOAD] {local_path.name} ({size_mb:.0f} MB) → {hf_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=args.repo,
            repo_type="dataset",
            token=args.token,
        )

    if not args.skip_qdrant:
        if not QDRANT_DIR.exists():
            print(f"[ERROR] Missing: {QDRANT_DIR}")
            sys.exit(1)
        print(f"[UPLOAD] qdrant_data/ (~673 MB) — có thể mất vài phút...")
        api.upload_folder(
            folder_path=str(QDRANT_DIR),
            path_in_repo="qdrant_data",
            repo_id=args.repo,
            repo_type="dataset",
            token=args.token,
        )

    print(f"\n[OK] Upload complete: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
