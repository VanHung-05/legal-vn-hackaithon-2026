# Kế hoạch – Bảng C INNOVATOR (2 người)

> **Model:** Gemma-4 (LLM) + BGE-M3 (Embedding) + Qdrant (vector store)  
> **Bài toán:** Legal retrieval — tìm điều luật liên quan cho mỗi câu hỏi  
> **Team:** 2 người — plan chỉ mô tả **ai làm gì**

---

## Tổng quan

### Bài toán thực tế

| | |
|---|---|
| **Input** | `qid` + `question` |
| **Output** | `qid` + `relevant_laws` — danh sách `aid` (số nguyên) |
| **Corpus** | ~59.636 điều luật, mỗi điều có `aid`, `law_id`, nội dung text |
| **Train** | 2.190 câu, có ground truth `relevant_laws` |
| **Public test** | 312 câu, `relevant_laws` rỗng (ẩn label) |
| **Nộp** | Docker Hub + GitHub + báo cáo phương pháp |
| **Chấm V2** | Accuracy 80đ + Inference time 10đ + Ý tưởng 10đ |

### Thống kê train (quan trọng khi thiết kế output)

| Số điều luật / câu | Số câu | Tỷ lệ |
|---|---|---|
| 1 điều | 1.645 | ~75% |
| 2 điều | 415 | ~19% |
| 3+ điều | 130 | ~6% |
| Trung bình | 1,34 điều/câu | max 9 điều |

→ Phần lớn câu chỉ cần **1 aid đúng**; một số câu cần **nhiều aid**. Pipeline phải xử lý được cả hai.

### Luồng hệ thống

1. Đọc câu hỏi từ file test
2. Retrieve candidate điều luật (BGE-M3 + Qdrant + BM25 hybrid)
3. (Optional) Gemma-4 rerank / lọc / mở rộng query
4. Chọn danh sách `aid` cuối cùng → ghi output

> **Lưu ý:** Bảng C bắt buộc dùng Gemma-4 — nên dùng LLM ở bước rerank hoặc query expansion, không bỏ hẳn LLM.

---

## Người A — RETRIEVAL CORE

> Xử lý corpus, embedding, Qdrant, đánh giá retrieval, Docker, tốc độ.

### A1. Xử lý dữ liệu corpus

- Đọc corpus pháp luật, flatten thành ~59.636 điều luật
- Mỗi điều: `aid` (ID duy nhất, dùng làm output), `law_id` (số hiệu văn bản), `text`
- Chuẩn hóa text, cắt ngắn điều quá dài
- Xuất metadata dùng chung (Người B cần cho BM25)
- Validate: `aid` trong train đều tồn tại trong corpus

### A2. Embedding BGE-M3

- Embed toàn bộ corpus offline (một lần)
- Embed câu hỏi riêng, đúng cách dùng BGE-M3
- Cache embedding câu hỏi trùng lặp

### A3. Lưu vector bằng Qdrant

- Qdrant local, lưu vector + payload `{aid, law_id, text}`
- Point ID = `aid`
- Build offline, bake vào Docker — không build lại mỗi inference

### A4. Dense search (semantic)

- Câu hỏi → top-K candidate `aid` + score (K=20~50 ở giai đoạn candidate)
- Trả đủ metadata điều luật cho bước rerank

### A5. Đánh giá retrieval

- Metric chính trên train holdout:
  - **Recall@K** — có bao nhiêu % câu retrieve được ít nhất 1 aid đúng trong top-K
  - **MRR** — điều đúng xuất hiện ở rank nào
  - **F1 theo tập** — so sánh set `aid` predict vs ground truth (quan trọng vì câu có thể nhiều aid)
- Liệt kê câu fail để debug
- **Mục tiêu:** Recall@5 ≥ 75%, F1@5 càng cao càng tốt

### A6. Đo tốc độ

- Reg/s end-to-end
- Ghi vào README — phục vụ 10đ inference time

### A7. Docker

- Index Qdrant pre-built trong image
- Input `/data`, output `/output` đúng spec BTC
- Load index một lần khi start

### A8. Thống nhất cấu trúc dữ liệu

- Cùng B định nghĩa: cấu trúc câu hỏi, cấu trúc output `relevant_laws`, config
- **Thống nhất trước khi code**

### A9. Báo cáo (phần A)

- Tại sao chọn Qdrant
- Pipeline embedding + dense retrieval
- Ablation Recall@K / F1@K
- Tune tham số HNSW, top-K

### Bàn giao cho B

- Vector store Qdrant search được
- Metadata điều luật cho BM25
- Hàm dense search: câu hỏi → list `(aid, score, text)`
- Script eval để B chạy lại khi tune hybrid

---

## Người B — RETRIEVAL ENHANCE + PIPELINE

> BM25, hybrid, Gemma rerank, quyết định số aid output, ghép pipeline, demo, eval.

### B1. Đọc và ghi dữ liệu

- Đọc file test (public/private) — mỗi dòng: `qid` + `question`
- Ghi output: `qid` + `relevant_laws` (danh sách `aid`)
- Format output **phải khớp spec BTC** (JSON hoặc CSV — xác nhận khi BTC công bố sample)
- Validate: mọi `aid` output phải tồn tại trong corpus, không thiếu `qid`

### B2. BM25 (sparse retrieval)

- Build BM25 từ metadata điều luật
- Tokenize tiếng Việt
- Search top-K candidate theo keyword

### B3. Hybrid retrieval (dense + sparse)

- Gộp kết quả Qdrant (A) + BM25 bằng RRF
- Tune trọng số trên train holdout
- Output: danh sách candidate `aid` xếp hạng (top-20~50)

### B4. Gemma-4 rerank / query expansion

> Bắt buộc dùng Gemma-4 (spec Bảng C). Hai hướng (chọn 1 hoặc kết hợp):

**Hướng A — Query expansion (trước retrieve):**
- Gemma viết lại / mở rộng câu hỏi → embed câu mở rộng → retrieve tốt hơn

**Hướng B — Rerank (sau retrieve):**
- Lấy top-20 candidate từ hybrid
- Gemma đọc câu hỏi + từng điều luật → chấm điểm liên quan / chọn aid nào thực sự relevant
- Output danh sách `aid` cuối

Quantize INT4 nếu cần tốc độ. Warmup khi container start.

### B5. Quyết định số lượng aid output

- Train cho thấy ~75% câu chỉ có **1 aid**, nhưng ~25% có 2–9 aid
- Cần chiến lược linh hoạt:
  - **Ngưỡng score:** chỉ giữ aid trên threshold
  - **Top-K động:** 1 aid nếu score cao vượt trội, nhiều aid nếu nhiều candidate sát nhau
  - **Tune trên train:** chọn chiến lược F1 cao nhất
- Không hardcode luôn top-5 — sẽ thừa aid sai ở 75% câu 1-điều

### B6. Ghép pipeline hoàn chỉnh

- Load index + model một lần
- Mỗi câu: (expand query?) → hybrid retrieve → (Gemma rerank?) → chọn aid → ghi output
- Entrypoint: BTC chạy container → tự động xử lý toàn bộ test set

### B7. Đánh giá trên train

- Chạy pipeline trên holdout 10%
- Metric: Recall@K, Precision@K, F1 (set-based), MRR
- Lưu câu sai: thiếu aid, thừa aid, rank sai
- So sánh: hybrid-only vs hybrid+Gemma rerank

### B8. Demo Vòng 3

- Nhập câu hỏi pháp luật → hiện danh sách điều luận liên quan (aid, số hiệu văn bản, nội dung)
- Hiện score / thứ hạng
- Dùng trong pitch 8 phút

### B9. Báo cáo (phần B)

- Hybrid RRF (sparse + dense)
- Vai trò Gemma-4: rerank vs query expansion — ablation
- Chiến lược chọn số aid output
- Tradeoff accuracy vs tốc độ

### Cần từ A trước khi làm B3–B6

- Qdrant index sẵn sàng
- Metadata điều luật
- Dense search hoạt động ổn định

---

## Phần chung (cả hai)

### Thứ tự tích hợp

1. Thống nhất format input/output + cấu trúc dữ liệu
2. A: corpus + Qdrant + dense search + baseline eval
3. B: BM25 + hybrid + tune trên train
4. B: thêm Gemma rerank/expansion → so sánh metric
5. B: pipeline + entrypoint → A: Docker + benchmark
6. Cả hai optimize → nộp bài

### Phân chia module

| Phần | Người A | Người B |
|---|---|---|
| Xử lý corpus | ✓ | |
| BGE-M3 + Qdrant | ✓ | |
| BM25 + hybrid | | ✓ |
| Gemma rerank / query expansion | | ✓ |
| Quyết định số aid output | | ✓ |
| Pipeline + entrypoint | hỗ trợ | ✓ lead |
| Docker | ✓ | review |
| Demo | | ✓ |
| Eval (Recall, F1, MRR) | ✓ baseline | ✓ end-to-end |
| Báo cáo | dense + Qdrant | hybrid + Gemma + output strategy |

### Metric cần theo dõi

| Metric | Ý nghĩa | Ai đo |
|---|---|---|
| Recall@K | Có retrieve được aid đúng không | A + B |
| MRR | Aid đúng ở rank mấy | A + B |
| F1 (set) | Khớp bao nhiêu aid predict vs ground truth | B |
| Reg/s | Tốc độ inference | A |

### Checklist nộp bài

**Vòng 1 (23/6):**
- [ ] Docker chạy: input test → output `relevant_laws` đúng format
- [ ] GitHub + README reproduce
- [ ] Báo cáo phương pháp PDF
- [ ] Nộp đủ trong 72h sau Vòng 1

**Vòng 2 (26/6):**
- [ ] Docker final optimized
- [ ] Reg/s trong README
- [ ] Test trên máy sạch

**Vòng 3 (15/7):**
- [ ] Demo retrieval ổn định
- [ ] Slide 8 phút + Q&A

### Tối đa hóa điểm

| Điểm | Việc cần làm | Ai |
|---|---|---|
| **80đ Accuracy** | Recall@K + F1 cao; Gemma rerank cải thiện rõ so với hybrid-only | A + B |
| **10đ Speed** | Qdrant pre-built, Gemma INT4, batch rerank, cắt context rerank | A + B |
| **10đ Ý tưởng** | Ablation: dense vs hybrid vs +Gemma; chiến lược dynamic K | A + B |

### Rủi ro

| Rủi ro | Xử lý |
|---|---|
| Retrieve miss aid đúng | Tune hybrid; query expansion bằng Gemma |
| Thừa aid sai (giảm Precision/F1) | Ngưỡng score; Gemma rerank lọc |
| Câu nhiều aid (2–9) bị thiếu | Không hardcode K=1; dùng dynamic threshold |
| Gemma quá chậm | INT4; rerank top-10 thay top-50; cache |
| Format output sai | Xác nhận spec BTC sớm; viết validator |
| Overfit public test | Tune chỉ trên train holdout |

---

*Cập nhật: 29/5/2026 — sửa từ MCQ sang legal retrieval (`relevant_laws` / `aid`)*
