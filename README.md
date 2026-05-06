# Lab 20: Multi-Agent Research System Starter

Starter repo cho bài lab **Multi-Agent Systems**: xây dựng hệ thống nghiên cứu gồm **Supervisor + Researcher + Analyst + Writer** và benchmark với single-agent baseline.

> Repo này bắt đầu từ **production-grade skeleton**. Trong phiên bản hiện tại, các milestone chính (LLM client, routing, workers, workflow, tracing, benchmark report) đã được implement để bạn có thể chạy end-to-end và kiểm tra deliverables.

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Agent interfaces + skeletons
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # LangGraph workflow skeleton
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark/evaluation skeleton
│   ├── observability/       # Logging/tracing hooks
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Unit tests for skeleton behavior
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev,llm]"
cp .env.example .env
```

### 2. Cấu hình API keys

Mở `.env` và điền key cần thiết.

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
# optional (Langfuse Cloud EU/US/JP hoặc self-host)
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 3. Chạy smoke test

```bash
python -m pytest
python -m multi_agent_research_lab.cli --help
```

> Windows/PowerShell mặc định không có `make`. Nếu bạn muốn chạy các target trong `Makefile`, dùng lệnh tương đương:
>
> - `make test` → `python -m pytest`
> - `make lint` → `python -m ruff check src tests`
> - `make format` → `python -m ruff format src tests`
> - `make typecheck` → `python -m mypy src`

### 4. Chạy baseline skeleton

```bash
python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Baseline hiện **đã gọi LLM thật** qua `src/multi_agent_research_lab/services/llm_client.py` và in latency/tokens.

### 5. Chạy multi-agent skeleton

```bash
python -m multi_agent_research_lab.cli multi-agent --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Multi-agent hiện **đã chạy end-to-end** theo router: `researcher → analyst → writer → done` và trả về `ResearchState` dạng JSON.

### 6. Kiểm tra tracing (Langfuse)

- Khi chạy `baseline` / `multi-agent`, hệ thống sẽ tạo trace:
  - `MultiAgentWorkflow` (chain)
  - `tavily.search` (retriever)
  - `openai.chat.completions` (generation, có usage token)
- Sau khi chạy xong, mở Langfuse UI để copy link trace và dán vào `reports/benchmark_report.md`.

### 7. Kiểm tra benchmark & report

- Report mẫu đã có tại `reports/benchmark_report.md` (so sánh single vs multi-agent + failure modes).
- Nếu muốn chạy benchmark nhanh (1 query) và xem metrics in terminal:

```bash
python -c "from multi_agent_research_lab.evaluation.benchmark import baseline_runner_factory, multi_agent_runner_factory, run_benchmark; q='Define GraphRAG in 2 sentences for engineers.'; print(run_benchmark('baseline', q, baseline_runner_factory())[1].model_dump()); print(run_benchmark('multi', q, multi_agent_runner_factory())[1].model_dump())"
```

## Milestones trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0-15' | Setup, chạy baseline skeleton | `cli.py`, `services/llm_client.py` |
| 15-45' | Build Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45-75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75-95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95-115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115-120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard-code API key trong code.
- Tất cả input/output chính dùng Pydantic schema.
- Có type hints, linting, formatting, unit test tối thiểu.
- Có logging/tracing hook ngay từ đầu.
- Không để agent chạy vô hạn: dùng `max_iterations`, `timeout_seconds`.
- Có benchmark report thay vì chỉ demo output đẹp.

## TODO chính cho học viên

Tìm trong code các marker:

```bash
grep -R "TODO(student)" -n src tests docs
```

Nếu bạn đang dùng repo như “starter”, đây là checklist milestone gốc. Với phiên bản hiện tại, các mục này đã được implement để bạn có thể chạy/đánh giá trực tiếp:

1. LLM client (OpenAI).
2. Web/search client (Tavily) + fallback desk research.
3. Routing policy trong Supervisor.
4. Worker agents: Researcher/Analyst/Writer.
5. LangGraph workflow.
6. Tracing provider thật: **Langfuse**.
7. Benchmark + report markdown.

## Deliverables

Học viên nộp:

1. GitHub repo cá nhân.
2. Screenshot trace hoặc link trace.
3. `reports/benchmark_report.md` so sánh single vs multi-agent.
4. Một đoạn giải thích failure mode và cách fix.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
