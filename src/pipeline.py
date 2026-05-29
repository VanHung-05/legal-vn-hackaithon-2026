"""Pipeline orchestrator — run build steps in order."""

import sys
from pathlib import Path

from config import CORPUS_EMBEDDINGS_FILE, QDRANT_DIR


STEPS = {
    "process": ("process_corpus", "Flatten & normalize corpus"),
    "embed": ("embed_corpus", "Embed corpus with BGE-M3"),
    "validate": ("validate_embeddings", "Validate embedding files"),
    "index": ("build_qdrant", "Build Qdrant vector index"),
    "search": ("dense_search", "Smoke test dense search"),
    "eval": ("evaluate", "Evaluate on train holdout"),
    "benchmark": ("benchmark", "Benchmark inference speed"),
    "infer": ("run_inference", "Run inference on test set"),
}


def check_embeddings() -> bool:
    return CORPUS_EMBEDDINGS_FILE.exists()


def check_index() -> bool:
    return QDRANT_DIR.exists() and any(QDRANT_DIR.iterdir()) if QDRANT_DIR.exists() else False


def run_step(name: str) -> None:
    if name not in STEPS:
        print(f"Unknown step: {name}. Available: {', '.join(STEPS)}")
        sys.exit(1)

    module_name, desc = STEPS[name]
    print(f"\n{'='*50}")
    print(f"Step: {name} — {desc}")
    print(f"{'='*50}")

    if name == "index" and not check_embeddings():
        print(f"[ERROR] Missing embeddings at {CORPUS_EMBEDDINGS_FILE}")
        print("  Run 'embed' step first, or place Colab output files there.")
        sys.exit(1)

    if name == "validate":
        from validate_embeddings import validate
        if not validate():
            sys.exit(1)
        return

    import importlib
    mod = importlib.import_module(module_name)
    mod.main()


def run_all(from_step: str = "process") -> None:
    order = list(STEPS.keys())
    start = order.index(from_step) if from_step in order else 0
    for step in order[start:]:
        if step == "embed" and check_embeddings():
            print(f"\n[SKIP] embed — embeddings already exist at {CORPUS_EMBEDDINGS_FILE}")
            continue
        if step == "index" and check_index():
            print(f"\n[SKIP] index — Qdrant index already exists at {QDRANT_DIR}")
            continue
        run_step(step)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline.py all              # run full pipeline")
        print("  python pipeline.py all --from index  # resume from index step")
        print("  python pipeline.py <step>           # run single step")
        print(f"\nSteps: {', '.join(STEPS)}")
        return

    if sys.argv[1] == "all":
        from_step = "process"
        if "--from" in sys.argv:
            from_step = sys.argv[sys.argv.index("--from") + 1]
        run_all(from_step)
    else:
        run_step(sys.argv[1])


if __name__ == "__main__":
    main()
