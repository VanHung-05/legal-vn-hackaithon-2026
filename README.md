# Legal VN — HackAIthon 2026 (Bảng C INNOVATOR)



Legal retrieval pipeline: tìm điều luật liên quan cho câu hỏi pháp lý.



**Model:** BGE-M3 (embedding) + Qdrant (vector store) + Gemma-4 (rerank, Người B)



## Cấu trúc project



```

Law_VN/

├── src/                    # Source code

│   ├── config.py           # Config chung (+ HF repo id)

│   ├── dense_search.py     # DenseSearcher API

│   └── ...

├── scripts/

│   └── download_artifacts.py # Tải artifacts từ Hugging Face

├── data/

│   ├── train.json            # ✓ trên Git

│   ├── public_test.json      # ✓ trên Git

│   ├── legal_corpus.json     # ✗ Hugging Face (~111 MB)

│   └── processed/

│       ├── aid_law_map.json  # ✓ trên Git

│       ├── articles.jsonl    # ✗ Hugging Face (~197 MB)

│       └── embeddings/       # ✗ Hugging Face (~233 MB)

├── qdrant_data/              # ✗ Hugging Face (~673 MB) hoặc rebuild

├── embed_colab.ipynb

├── requirements.txt

├── HANDOFF_B.md

└── PLAN.md

```



---



## Quick start (clone xong → chạy được)



```bash

git clone https://github.com/VanHung-05/legal-vn-hackaithon-2026.git

cd legal-vn-hackaithon-2026



pip install -r requirements.txt



# Tải artifacts lớn từ Hugging Face (~1.2 GB)

python3 scripts/download_artifacts.py



# Kiểm tra + chạy

cd src

python3 validate_embeddings.py

python3 dense_search.py

```



---



## Artifacts trên Hugging Face



Các file quá lớn cho Git được lưu tại:



**https://huggingface.co/datasets/nguyenvanhung05/legal-vn-hackaithon-artifact**



| File trên HF | Local path | Kích thước |

|---|---|---|

| `data/legal_corpus.json` | `data/legal_corpus.json` | ~111 MB |

| `data/processed/articles.jsonl` | `data/processed/articles.jsonl` | ~197 MB |

| `data/processed/embeddings/corpus_embeddings.npy` | same | ~233 MB |

| `data/processed/embeddings/corpus_aids.npy` | same | ~0.5 MB |

| `qdrant_data/` | `qdrant_data/` | ~673 MB |



### Tải artifacts (Người B / máy mới)



**Repo public** — không cần token:



```bash

pip install huggingface_hub

python3 scripts/download_artifacts.py

```



**Repo private** — cần token:



1. Vào https://huggingface.co/settings/tokens → tạo token (Read)

2. Login:

   ```bash

   huggingface-cli login

   ```

3. Tải:

   ```bash

   python3 scripts/download_artifacts.py

   # hoặc

   python3 scripts/download_artifacts.py --token hf_xxxxxxxx

   ```



**Tùy chọn:**



```bash

# Bỏ qua qdrant, tự build (~1 phút nếu đã có embeddings)

python3 scripts/download_artifacts.py --skip-qdrant

cd src && python3 build_qdrant.py



# Chỉ tải qdrant_data

python3 scripts/download_artifacts.py --only-qdrant

```

Chi tiết tải artifacts: [HANDOFF_B.md §2.2](HANDOFF_B.md#22-hugging-face--tải-artifacts)



---



## Setup & chạy pipeline



```bash

pip install -r requirements.txt

python3 scripts/download_artifacts.py   # nếu chưa có artifacts



cd src

python3 validate_embeddings.py

python3 build_qdrant.py      # bỏ qua nếu đã tải qdrant_data/

python3 evaluate.py          # eval baseline

python3 run_inference.py       # inference → output/pred.csv

```



Hoặc chạy gộp:



```bash

cd src

python3 pipeline.py all --from validate

```



---



## Baseline (dense-only)



| Metric | Giá trị |

|--------|---------|

| Hit@5 | 78.1% |

| MRR | 0.62 |

| Tốc độ | 2.37 q/s (CPU) |



Chi tiết bàn giao: [HANDOFF_B.md](HANDOFF_B.md)



---



## Phân công



| Người A | Người B |

|---------|---------|

| Corpus, BGE-M3, Qdrant, dense search | BM25, hybrid, Gemma rerank |

| Docker (sau pipeline) | Pipeline + entrypoint, demo |



---



## Lưu ý Git vs Hugging Face



| Lưu ở đâu | Nội dung |

|---|---|

| **GitHub** | Code (`src/`), config, docs, `train.json`, `public_test.json`, `aid_law_map.json` |

| **Hugging Face** | File lớn: corpus, articles, embeddings, qdrant index |

| **Không commit** | `output/`, `__pycache__/`, `.env` |


