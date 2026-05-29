# Tài liệu bàn giao — Người A → Người B

> **Dự án:** Legal Retrieval — Bảng C INNOVATOR  
> **Ngày bàn giao:** 30/05/2026  
> **Trạng thái:** Dense retrieval core hoàn thành, sẵn sàng tích hợp hybrid + Gemma

---

## 1. Tổng quan

Người A đã xây xong **retrieval core** (dense):

```
Corpus → flatten → BGE-M3 embed → Qdrant index → DenseSearcher
```

Bạn (Người B) cần build tiếp:

```
DenseSearcher + BM25 → Hybrid (RRF) → Gemma rerank → chọn aid → output
```

---

## 2. Artifacts đã bàn giao

### 2.1. Dữ liệu đã xử lý

| File | Mô tả | Kích thước |
|------|--------|------------|
| `data/processed/articles.jsonl` | 59.636 điều luật (flatten) — **dùng cho BM25** | ~185 MB |
| `data/processed/aid_law_map.json` | Map `aid → law_id` | nhỏ |
| `data/processed/embeddings/corpus_embeddings.npy` | Vector dense toàn corpus | ~233 MB |
| `data/processed/embeddings/corpus_aids.npy` | Thứ tự aid tương ứng embedding | ~466 KB |
| `qdrant_data/` | Qdrant index pre-built (59.636 points) | ~300+ MB |

### 2.2. Source code (`src/`)

| File | Vai trò | B có cần sửa? |
|------|---------|---------------|
| `config.py` | Paths, hyperparams chung | Có thể thêm config hybrid/Gemma |
| `schemas.py` | Data classes: `Article`, `Question`, `SearchResult`, `Prediction` | Dùng chung |
| `data_io.py` | Load/save JSON, JSONL, CSV | Dùng chung |
| `device.py` | Chọn device: CUDA nếu có, else CPU | Không cần sửa |
| **`dense_search.py`** | **`DenseSearcher`** — API chính cho B | **Gọi, không sửa** |
| `evaluate.py` | Metrics + hàm `evaluate()` | Dùng lại khi eval pipeline |
| `run_inference.py` | Inference mẫu (dense-only baseline) | Tham khảo, B viết entrypoint mới |
| `benchmark.py` | Đo tốc độ q/s | Chạy sau khi pipeline xong |

### 2.3. Kết quả baseline

| File | Nội dung |
|------|----------|
| `output/eval_dense_results.json` | Metrics dense-only |
| `output/eval_dense_failures.json` | 48/219 câu fail Hit@5 — để debug |

---

## 3. Format dữ liệu (đã thống nhất)

### 3.1. Input — câu hỏi

**Train / test (JSON):**
```json
{
  "qid": 933,
  "question": "Thưa luật sư tôi có đăng ký kết hôn...",
  "relevant_laws": [53877, 53875, 53929]
}
```

- Test set: `relevant_laws` rỗng `[]`
- Tên file thực tế: `train.json`, `public_test.json` (BTC có thể đổi tên → `data_io.find_test_file()` tự detect)

**Load bằng code:**
```python
from config import DATA_DIR, TRAIN_FILE
from data_io import load_questions, load_test_data

train = load_questions(TRAIN_FILE)       # list[Question]
test  = load_test_data(DATA_DIR)         # auto-detect test file
```

### 3.2. Article (corpus flatten)

Mỗi dòng trong `articles.jsonl`:
```json
{
  "aid": 53877,
  "law_id": "52/2014/QH13",
  "text": "Điều 19. Quyền kết hôn...",
  "text_truncated": "Điều 19. Quyền kết hôn...",
  "char_len": 1234
}
```

- `aid`: ID duy nhất, **dùng làm output**
- `text`: full text — **dùng cho BM25**
- `text_truncated`: cắt 2048 ký tự — dùng khi embed

### 3.3. Output — submission

**JSON:**
```json
[
  {"qid": 8631, "relevant_laws": [1234, 5678]}
]
```

**CSV (`pred.csv`):**
```csv
qid,relevant_laws
8631,1234,5678
```

**Lưu bằng code:**
```python
from config import OUTPUT_DIR
from data_io import save_predictions_json, save_predictions_csv
from schemas import Prediction

predictions = [Prediction(qid=8631, relevant_laws=[1234, 5678])]
save_predictions_json(predictions, OUTPUT_DIR / "pred.json")
save_predictions_csv(predictions, OUTPUT_DIR / "pred.csv")
```

### 3.4. Docker I/O (spec BTC)

```
/data/   ← mount input (test file)
/output/ ← mount output (pred.csv)
```

---

## 4. API chính — `DenseSearcher`

### 4.1. Khởi tạo (load 1 lần khi start container)

```python
from dense_search import DenseSearcher

searcher = DenseSearcher()          # load BGE-M3 + Qdrant
# searcher = DenseSearcher(load_model=False)  # chỉ Qdrant, không load model
```

- Model: `BAAI/bge-m3` (sentence-transformers)
- Device: CUDA nếu có GPU, else CPU
- Qdrant: local file tại `qdrant_data/`, collection `legal_articles`

### 4.2. Search — trả full metadata (cho rerank)

```python
results = searcher.search("Ai có quyền kết hôn?", top_k=50)
# → list[SearchResult]

for r in results:
    print(r.aid, r.score, r.law_id, r.text[:100])
```

**`SearchResult` fields:**
| Field | Type | Mô tả |
|-------|------|--------|
| `aid` | int | ID điều luật |
| `score` | float | Cosine similarity (0–1) |
| `law_id` | str | Số hiệu văn bản, vd. `"52/2014/QH13"` |
| `text` | str | Nội dung (truncated 1000 ký tự trong Qdrant payload) |

> **Lưu ý:** Text trong Qdrant payload bị cắt 1000 ký tự. Nếu Gemma rerank cần full text, load từ `articles.jsonl`:
> ```python
> from config import ARTICLES_FILE
> from data_io import load_articles_map
> articles = load_articles_map(ARTICLES_FILE)
> full_text = articles[aid].text
> ```

### 4.3. Search — chỉ trả aid (cho hybrid RRF)

```python
aids = searcher.search_aids("Ai có quyền kết hôn?", top_k=50)
# → [53877, 53875, 53929, ...]
```

### 4.4. Batch search

```python
queries = ["câu hỏi 1", "câu hỏi 2"]
batch_results = searcher.batch_search(queries, top_k=50)
# → list[list[SearchResult]]
```

### 4.5. Embed query riêng (cho query expansion)

```python
vec = searcher.embed_query("câu hỏi đã mở rộng bởi Gemma")
results = searcher.search_by_vector(vec, top_k=50)
```

### 4.6. Cleanup

```python
searcher.close()
```

---

## 5. Gợi ý tích hợp Hybrid (RRF)

```python
from dense_search import DenseSearcher

def rrf_fusion(dense_aids, sparse_aids, k=60, top_n=50):
    """Reciprocal Rank Fusion."""
    scores = {}
    for rank, aid in enumerate(dense_aids):
        scores[aid] = scores.get(aid, 0) + 1 / (k + rank + 1)
    for rank, aid in enumerate(sparse_aids):
        scores[aid] = scores.get(aid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


searcher = DenseSearcher()

def hybrid_search(question: str, bm25_searcher, top_k=50) -> list[int]:
    dense_aids = searcher.search_aids(question, top_k=top_k)
    sparse_aids = bm25_searcher.search(question, top_k=top_k)  # B implement
    return rrf_fusion(dense_aids, sparse_aids, top_n=top_k)
```

**Metadata cho BM25:** dùng `articles.jsonl`, field `text`, index theo `aid`.

---

## 6. Baseline metrics (dense-only)

Đo trên **219 câu holdout** (10% train, seed=42):

| Metric | Dense-only | Ghi chú |
|--------|-----------|---------|
| **Hit@5** | **78.1%** | ≥1 aid đúng trong top-5 |
| Recall@5 | 68.9% | Tỷ lệ aid đúng được retrieve |
| Hit@10 | 87.7% | |
| MRR | 0.62 | Rank aid đúng đầu tiên |
| F1@5 | 0.26 | Thấp vì hardcode top-5 (nhiều câu chỉ cần 1 aid) |
| Tốc độ | **2.37 q/s** (~422 ms/query, CPU) | GPU sẽ nhanh hơn |

**Mục tiêu sau hybrid + Gemma:** cải thiện F1 và Recall@5; Hit@5 đã đạt ngưỡng 75%.

**So sánh:** sau khi B xong pipeline, chạy `evaluate.py` với `search_fn` mới và so với `output/eval_dense_results.json`.

---

## 7. Config quan trọng (`config.py`)

```python
TOP_K_CANDIDATES = 50   # K ở giai đoạn candidate (hybrid input)
TOP_K_FINAL = 5         # K output cuối (baseline dense-only, B nên dynamic)

BGE_M3_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 256

QDRANT_COLLECTION = "legal_articles"
HNSW_EF_SEARCH = 128

EVAL_HOLDOUT_RATIO = 0.1
EVAL_SEED = 42          # Dùng cùng seed khi eval để so sánh công bằng
```

---

## 8. Cách chạy & kiểm tra

```bash
cd src
pip install -r ../requirements.txt

# Smoke test dense search (5 câu train)
python3 dense_search.py

# Eval baseline (219 holdout queries, ~2 phút CPU)
python3 evaluate.py

# Inference public test → output/pred.csv
python3 run_inference.py

# Benchmark tốc độ
python3 benchmark.py
```

**Eval pipeline của B** — dùng lại hàm `evaluate()`:

```python
from evaluate import evaluate, split_holdout
from data_io import load_questions
from config import TRAIN_FILE, TOP_K_CANDIDATES

train = load_questions(TRAIN_FILE)
_, holdout = split_holdout(train)

def my_search_fn(question: str) -> list[int]:
    # hybrid + rerank + chọn aid của B
    ...
    return predicted_aids

results = evaluate(holdout, my_search_fn)
```

---

## 9. Thống kê train (ảnh hưởng chiến lược output)

| Số aid / câu | Số câu | Tỷ lệ |
|---|---|---|
| 1 aid | 1.645 | ~75% |
| 2 aid | 415 | ~19% |
| 3+ aid | 130 | ~6% |
| Trung bình | 1.34 aid/câu | max 9 aid |

→ **Không hardcode top-5.** ~75% câu chỉ cần 1 aid — hardcode top-5 làm F1 thấp (baseline F1@5 = 0.26).

---

## 10. Việc B cần làm

| # | Task | Input từ A |
|---|------|------------|
| B1 | Đọc/ghi test + output đúng spec BTC | `data_io.py`, `schemas.py` |
| B2 | BM25 từ `articles.jsonl` | `articles.jsonl`, `aid_law_map.json` |
| B3 | Hybrid RRF (dense + BM25) | `DenseSearcher.search_aids()` |
| B4 | Gemma-4 rerank hoặc query expansion | `SearchResult` + full text từ articles |
| B5 | Dynamic số aid output (threshold / top-K động) | Baseline metrics |
| B6 | **Pipeline + entrypoint** (`/data` → `/output`) | Tham khảo `run_inference.py` |
| B7 | Eval end-to-end, so sánh vs baseline | `evaluate.py` |
| B8 | Demo Vòng 3 | `DenseSearcher.search()` |

---

## 11. Việc A làm sau khi B xong pipeline

| # | Task |
|---|------|
| A7 | **Dockerfile** — bake Qdrant index + model, mount `/data` `/output` |
| A6 | Benchmark final, ghi q/s vào README |
| A9 | Báo cáo phần A (Qdrant, ablation dense) |

---

## 12. Lưu ý kỹ thuật

1. **Qdrant local mode:** Cảnh báo khi >20K points — bình thường, chạy được. Trong Docker nên bake sẵn index, không rebuild mỗi lần inference.

2. **Full text cho rerank:** Qdrant payload text bị cắt 1000 chars. Gemma rerank nên load full text từ `articles.jsonl`.

3. **Device:** CUDA tự detect. Mac M1 chạy CPU (ổn định). Colab/server có GPU sẽ nhanh hơn.

4. **Model download:** Lần đầu chạy sẽ tải BGE-M3 từ HuggingFace (~2GB). Trong Docker nên cache sẵn.

5. **Git ignore:** `qdrant_data/`, `*.npy`, `output/` đã có trong `.gitignore`. B clone repo cần có sẵn `qdrant_data/` (share qua Drive hoặc rebuild từ embeddings).

6. **Rebuild index (nếu cần):**
   ```bash
   cd src
   python3 validate_embeddings.py
   python3 build_qdrant.py
   ```

---

## 13. Liên hệ / phân công

| Phần | Owner |
|------|-------|
| Dense retrieval, Qdrant, baseline eval | **A (xong)** |
| BM25, hybrid, Gemma, output strategy, entrypoint | **B** |
| Docker (sau pipeline B) | **A** |
| Demo Vòng 3 | **B** |
| Báo cáo | A: dense+Qdrant / B: hybrid+Gemma |

---

*Cập nhật: 30/05/2026*
