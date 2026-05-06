"""Benchmark helpers: latency, token-based cost estimate, citation coverage, suite failure rate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import append_span_to_state, trace_span

Runner = Callable[[str], ResearchState]

# USD per 1M tokens (approximate; override behavior via model string match only).
_DEFAULT_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


def _price_per_million_for_model(model: str) -> tuple[float, float]:
    key = model.strip().lower()
    return _DEFAULT_PRICE_PER_1M.get(key, (0.50, 1.50))


def estimate_cost_usd(
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    model: str | None = None,
) -> float | None:
    """Rough provider cost from token counts (for dashboards; not billing-grade)."""

    if input_tokens is None and output_tokens is None:
        return None
    m = model or get_settings().openai_model
    pin, pout = _price_per_million_for_model(m)
    it = input_tokens or 0
    ot = output_tokens or 0
    return (it / 1_000_000) * pin + (ot / 1_000_000) * pout


_CITATION_RE = re.compile(r"\[\d+\]")


def estimate_citation_coverage(final_answer: str | None) -> float:
    """Heuristic: share of substantive sentences that contain a bracket citation like ``[1]``.

    This is a cheap proxy for lab dashboards; peer review still judges real citation quality.
    """

    text = (final_answer or "").strip()
    if not text:
        return 0.0

    # Strip markdown code fences / headers noise lightly
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    sentences = [p.strip() for p in parts if len(p.strip()) >= 20]
    if not sentences:
        return 0.0

    cited = sum(1 for s in sentences if _CITATION_RE.search(s))
    return cited / len(sentences)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    *,
    peer_review_score: float | None = None,
) -> tuple[ResearchState | None, BenchmarkMetrics]:
    """Wall-clock run; enrich metrics from ``ResearchState`` totals and final answer."""

    settings = get_settings()
    started_outer = perf_counter()
    try:
        with trace_span(
            "benchmark_run",
            {"run_name": run_name, "query": query},
        ) as span:
            state = runner(query)
        latency = span["duration_seconds"] or (perf_counter() - started_outer)
        append_span_to_state(state, span)

        in_tok = state.total_input_tokens
        out_tok = state.total_output_tokens
        if in_tok == 0 and out_tok == 0:
            in_tok, out_tok = None, None

        cov = estimate_citation_coverage(state.final_answer)
        cost = estimate_cost_usd(in_tok, out_tok, model=settings.openai_model)

        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=float(latency),
            estimated_cost_usd=cost,
            quality_score=peer_review_score,
            input_tokens=in_tok,
            output_tokens=out_tok,
            citation_coverage=cov,
            failed=False,
            notes="",
        )
        return state, metrics
    except Exception as exc:  # noqa: BLE001 — benchmark harness captures all failures
        latency = perf_counter() - started_outer
        return None, BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=float(latency),
            failed=True,
            notes=str(exc),
        )


@dataclass(frozen=True)
class BenchmarkSuiteSummary:
    """Aggregate stats over one or more benchmark runs."""

    total_runs: int
    failure_count: int
    failure_rate: float
    mean_latency_seconds_success: float | None
    mean_citation_coverage_success: float | None
    mean_quality_score_success: float | None
    total_estimated_cost_usd_success: float | None


def run_benchmark_suite(
    run_name_prefix: str,
    queries: list[str],
    runner: Runner,
    *,
    peer_review_scores: Sequence[float | None] | None = None,
) -> tuple[list[BenchmarkMetrics], BenchmarkSuiteSummary]:
    """Run many queries; compute failure rate and means over successful runs."""

    metrics_list: list[BenchmarkMetrics] = []
    scores = list(peer_review_scores) if peer_review_scores is not None else []

    for i, q in enumerate(queries):
        name = f"{run_name_prefix}-{i + 1}"
        peer = scores[i] if i < len(scores) else None
        _, m = run_benchmark(name, q, runner, peer_review_score=peer)
        metrics_list.append(m)

    failures = [m for m in metrics_list if m.failed]
    oks = [m for m in metrics_list if not m.failed]
    n = len(metrics_list)
    fail_rate = (len(failures) / n) if n else 0.0

    mean_lat = sum(m.latency_seconds for m in oks) / len(oks) if oks else None
    covs = [m.citation_coverage for m in oks if m.citation_coverage is not None]
    mean_cov = sum(covs) / len(covs) if covs else None
    quals = [m.quality_score for m in oks if m.quality_score is not None]
    mean_qual = sum(quals) / len(quals) if quals else None
    costs = [m.estimated_cost_usd for m in oks if m.estimated_cost_usd is not None]
    total_cost = sum(costs) if costs else None

    summary = BenchmarkSuiteSummary(
        total_runs=n,
        failure_count=len(failures),
        failure_rate=fail_rate,
        mean_latency_seconds_success=mean_lat,
        mean_citation_coverage_success=mean_cov,
        mean_quality_score_success=mean_qual,
        total_estimated_cost_usd_success=total_cost,
    )
    return metrics_list, summary


def multi_agent_runner_factory() -> Runner:
    """Default runner for this lab: LangGraph multi-agent workflow."""

    from multi_agent_research_lab.core.schemas import ResearchQuery
    from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow

    workflow = MultiAgentWorkflow()

    def _run(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        return workflow.run(state)

    return _run


def baseline_runner_factory() -> Runner:
    """Single LLM call baseline (same behavior as CLI baseline)."""

    from multi_agent_research_lab.core.schemas import ResearchQuery
    from multi_agent_research_lab.services.llm_client import LLMClient

    llm = LLMClient()

    def _run(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        resp = llm.complete(
            system_prompt=(
                "You are a helpful research assistant. Provide a concise, accurate summary. "
                "If uncertain, state uncertainty explicitly. Do not fabricate citations."
            ),
            user_prompt=query,
        )
        state.final_answer = resp.content
        if resp.input_tokens is not None:
            state.total_input_tokens += resp.input_tokens
        if resp.output_tokens is not None:
            state.total_output_tokens += resp.output_tokens
        state.add_trace_event(
            "baseline_metrics",
            {
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            },
        )
        return state

    return _run
