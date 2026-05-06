"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.benchmark import BenchmarkSuiteSummary


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    *,
    summary: BenchmarkSuiteSummary | None = None,
) -> str:
    """Render benchmark metrics to markdown (per-run table + optional suite aggregates)."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | OK | Latency (s) | In tok | Out tok | Cost (USD est.) | Citation cov. | Quality (0–10) | Notes |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        ok = "no" if item.failed else "yes"
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        cov = "" if item.citation_coverage is None else f"{item.citation_coverage:.2f}"
        inn = "" if item.input_tokens is None else str(item.input_tokens)
        out = "" if item.output_tokens is None else str(item.output_tokens)
        notes = item.notes.replace("|", "\\|")
        lines.append(
            f"| {item.run_name} | {ok} | {item.latency_seconds:.2f} | {inn} | {out} | {cost} | {cov} | {quality} | {notes} |"
        )

    if summary is not None:
        lines.extend(
            [
                "",
                "## Aggregate (successful runs unless noted)",
                "",
                f"- **Total runs**: {summary.total_runs}",
                f"- **Failures**: {summary.failure_count} (**failure rate**: {summary.failure_rate:.1%})",
            ]
        )
        if summary.mean_latency_seconds_success is not None:
            lines.append(f"- **Mean latency (success)**: {summary.mean_latency_seconds_success:.2f}s")
        if summary.total_estimated_cost_usd_success is not None:
            lines.append(
                f"- **Sum estimated cost (success)**: USD {summary.total_estimated_cost_usd_success:.4f} "
                "(token-pricing heuristic; not billing-grade)"
            )
        if summary.mean_citation_coverage_success is not None:
            lines.append(
                f"- **Mean citation coverage heuristic (success)**: {summary.mean_citation_coverage_success:.2f} "
                "(sentences with `[n]` / substantive sentences)"
            )
        if summary.mean_quality_score_success is not None:
            lines.append(
                f"- **Mean peer-review score (success)**: {summary.mean_quality_score_success:.1f} / 10"
            )

    return "\n".join(lines) + "\n"
