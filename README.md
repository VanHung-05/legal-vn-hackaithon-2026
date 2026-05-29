# Legal VN — HackAIthon 2026 (Bảng C INNOVATOR)

Legal retrieval pipeline: tìm điều luật liên quan cho câu hỏi pháp lý.

**Model:** BGE-M3 (embedding) + Qdrant (vector store) + Gemma-4 (rerank, Người B)

## Cấu trúc project

```
Law_VN/
├── src/                    # Source code
│   ├── config.py           # Config chung
│   ├── schemas.py          # Data classes
│   ├── data_io.py          # Load/save dữ liệu
│   ├── device.py           # CUDA / CPU
│   ├── process_corpus.py   # Flatten corpus
│   ├── embed_corpus.py     # Embed BGE-M3
│   ├── build_qdrant.py     # Build vector index
│   ├── dense_search.py     # DenseSearcher API
│   ├── evaluate.py         # Recall@K, MRR, F1
│   ├── benchmark.py        # Đo tốc độ
│   ├── run_inference.py    # Inference test set
│   └── pipeline.py         # Orchestrator
├── data/
│   ├── train.json
│   ├── public_test.json
│   ├── legal_corpus.json   # (gitignore — quá lớn)
│   └── processed/
│       ├── articles.jsonl  # Corpus flatten (BM25)
│       ├── aid_law_map.json
│       └── embeddings/     # (gitignore)
├── qdrant_data/            # (gitignore — rebuild được)
├── output/                 # (gitignore — kết quả chạy)
├── embed_colab.ipynb       # Embed trên Colab GPU
├── requirements.txt
├── HANDOFF_B.md            # Tài liệu bàn giao Người B
└── PLAN.md                 # Kế hoạch team
```

## Setup

```bash
pip install -r requirements.txt
```

## Chạy pipeline

```bash
cd src

# 1. Xử lý corpus (đã chạy)
python3 process_corpus.py

# 2. Embed — local hoặc Colab (embed_colab.ipynb)
python3 embed_corpus.py

# 3. Validate + build Qdrant index
python3 validate_embeddings.py
python3 build_qdrant.py

# 4. Eval baseline
python3 evaluate.py

# 5. Inference
python3 run_inference.py
```

Hoặc chạy gộp (bỏ qua bước đã có artifact):

```bash
python3 pipeline.py all --from validate
```

## Artifacts cần có (không trên Git)

| Artifact | Cách lấy |
|----------|----------|
| `data/processed/embeddings/*.npy` | Colab hoặc `embed_corpus.py` |
| `qdrant_data/` | `build_qdrant.py` (~1 phút) |
| `data/processed/articles.jsonl` | `process_corpus.py` hoặc Drive |

## Baseline (dense-only)

| Metric | Giá trị |
|--------|---------|
| Hit@5 | 78.1% |
| MRR | 0.62 |
| Tốc độ | 2.37 q/s (CPU) |

Chi tiết bàn giao cho Người B: xem [HANDOFF_B.md](HANDOFF_B.md).

## Phân công

| Người A | Người B |
|---------|---------|
| Corpus, BGE-M3, Qdrant, dense search | BM25, hybrid, Gemma rerank |
| Baseline eval, Docker (sau pipeline) | Pipeline + entrypoint, demo |
