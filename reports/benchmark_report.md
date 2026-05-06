# Benchmark report: single-agent vs multi-agent

## Trace (Langfuse)

**Link trace (dán URL từ Langfuse sau khi chạy pipeline):**

- _Ví dụ: mở [Langfuse Cloud](https://cloud.langfuse.com) → Traces → chọn trace mới nhất của `MultiAgentWorkflow` / `openai.chat.completions` → Copy link._

`https://cloud.langfuse.com/project/cmotjkczr02myad07lorcpryu/traces?peek=af8a626a52a78c37&observation=af8a626a52a78c37&traceId=8f00947fe44342b7ebfdd861f48946d9&timestamp=2026-05-06T04%3A27%3A23.563Z`

**Screenshot:** lưu ảnh chụp màn hình trace tại `reports/trace-screenshot.png`

---

## Phương pháp

- **Single-agent (baseline):** một lần gọi `LLMClient.complete` (xem `baseline_runner_factory` trong `evaluation/benchmark.py`).
- **Multi-agent:** LangGraph workflow: supervisor → researcher (Tavily + LLM) → analyst → writer (xem `multi_agent_runner_factory`).
- **Cùng câu hỏi** để so sánh tương đối (latency wall-clock, token từ `ResearchState`, cost ước lượng theo bảng giá trong benchmark, citation coverage heuristic).

**Query dùng cho bảng dưới:** `Define GraphRAG in 2 sentences for engineers.`

_(Số liệu ghi tại thời điểm chạy benchmark cục bộ; chạy lại có thể lệch nhẹ.)_

## Kết quả

| Metric | Single-agent baseline | Multi-agent workflow |
|--------|----------------------:|---------------------:|
| Latency (s) | ~5.83 | ~18.48 |
| Input tokens | 48 | 2585 |
| Output tokens | 58 | 662 |
| Est. cost (USD) | ~0.000042 | ~0.000785 |
| Citation coverage (heuristic) | 0.00 | 0.50 |

### Nhận xét ngắn

- **Baseline** nhanh và rẻ hơn rõ rệt; phù hợp câu hỏi ngắn, không cần nguồn hay cấu trúc phức tạp.
- **Multi-agent** tốn latency và token hơn (nhiều bước LLM + Tavily); đổi lại có `sources`, `research_notes`, `analysis_notes`, và bản trả lời cuối thường gắn trích dẫn `[n]` nên heuristic citation coverage cao hơn.
- **Chất lượng (rubric peer review)** cần chấm tay theo `docs/peer_review_rubric.md`; benchmark chỉ tự điền khi truyền `peer_review_score`.

---

## Failure modes và cách xử lý

**1) LangGraph trả về `dict` thay vì `ResearchState`.** Một số phiên bản / cấu hình LangGraph trả state dạng dict sau `invoke`. Nếu code giả định object có `.route_history` sẽ lỗi kiểu `'dict' object has no attribute 'route_history'`. **Cách fix:** chuẩn hoá ngay sau `invoke`, ví dụ `ResearchState(**raw)` nếu `raw` là dict (đã làm trong `graph/workflow.py` qua `_ensure_research_state`).

**2) Tavily / mạng lỗi hoặc timeout.** Researcher gọi API ngoài; lỗi HTTP hoặc timeout làm cả run fail nếu không bắt. **Cách fix:** bọc `_tavily_search` trong try/except, ghi `state.errors`, và cho supervisor route fallback (ví dụ bỏ qua web, dùng nhánh desk research như khi không có API key).

**3) Vượt `MAX_ITERATIONS` hoặc loop routing.** Supervisor có trần iteration; nếu một agent không làm đầy field state (ví dụ không set `research_notes`), graph có thể lặp vô ích đến khi đạt max. **Cách fix:** kiểm tra invariant sau mỗi agent; thêm route `done` với `reason` + nội dung lỗi cho người dùng; hoặc retry một bước có điều kiện.

**4) Langfuse export lỗi / timeout.** Trace cục bộ vẫn có trong `ResearchState.trace`, nhưng span trên cloud có thể không lên nếu mạng chặn hoặc key sai. **Cách fix:** kiểm tra `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` + `LANGFUSE_BASE_URL`; tăng timeout mạng hoặc chạy lại; dùng screenshot/link khi export thành công.

**5) Chi phí và rate limit OpenAI.** Multi-agent gọi nhiều generation; dễ chạm limit hoặc tốn phí. **Cách fix:** giảm `max_sources`, model nhỏ hơn, cache, hoặc chạy benchmark trên tập query nhỏ.
