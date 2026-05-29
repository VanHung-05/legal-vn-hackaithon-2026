"""Shared configuration for the legal retrieval pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODELS_DIR = ROOT_DIR / "models"
QDRANT_DIR = ROOT_DIR / "qdrant_data"

# Raw data files (actual names in /data)
CORPUS_FILE = DATA_DIR / "legal_corpus.json"
TRAIN_FILE = DATA_DIR / "train.json"
PUBLIC_TEST_FILE = DATA_DIR / "public_test.json"

# Processed artifacts
PROCESSED_DIR = DATA_DIR / "processed"
ARTICLES_FILE = PROCESSED_DIR / "articles.jsonl"
AID_LAW_MAP_FILE = PROCESSED_DIR / "aid_law_map.json"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
CORPUS_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "corpus_embeddings.npy"
CORPUS_AIDS_FILE = EMBEDDINGS_DIR / "corpus_aids.npy"

# ── BGE-M3 ─────────────────────────────────────────────────────────
BGE_M3_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
TRUNCATE_CHARS = 2048

# ── Qdrant ─────────────────────────────────────────────────────────
QDRANT_COLLECTION = "legal_articles"
HNSW_M = 16
HNSW_EF_CONSTRUCT = 200
HNSW_EF_SEARCH = 128

# ── Search ─────────────────────────────────────────────────────────
TOP_K_CANDIDATES = 50
TOP_K_FINAL = 5

# ── Evaluation ─────────────────────────────────────────────────────
EVAL_HOLDOUT_RATIO = 0.1
EVAL_SEED = 42
EVAL_K_VALUES = [1, 3, 5, 10, 20, 50]
