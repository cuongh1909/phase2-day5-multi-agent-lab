from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    estimate_citation_coverage,
    run_benchmark,
    run_benchmark_suite,
)


def test_citation_coverage_counts_bracket_cites() -> None:
    text = "Short. " + "This is a longer sentence that includes a citation [1]. " * 2
    cov = estimate_citation_coverage(text)
    assert 0.0 < cov <= 1.0


def test_run_benchmark_pulls_tokens_and_coverage() -> None:
    def runner(q: str) -> ResearchState:
        s = ResearchState(request=ResearchQuery(query=q))
        s.final_answer = (
            "GraphRAG combines graphs and retrieval. It helps with multi-hop queries [1]. "
            "Enterprises adopt it for trust [2]."
        )
        s.total_input_tokens = 100
        s.total_output_tokens = 50
        return s

    state, m = run_benchmark("unit", "hello world query", runner, peer_review_score=7.5)
    assert state is not None
    assert not m.failed
    assert m.input_tokens == 100
    assert m.output_tokens == 50
    assert m.quality_score == 7.5
    assert m.citation_coverage is not None and m.citation_coverage > 0
    assert m.estimated_cost_usd is not None


def test_suite_failure_rate() -> None:
    def boom(q: str) -> ResearchState:
        raise RuntimeError("planned failure")

    metrics, summary = run_benchmark_suite("fail", ["a", "b"], boom)
    assert len(metrics) == 2
    assert summary.failure_count == 2
    assert summary.failure_rate == 1.0
    assert summary.mean_latency_seconds_success is None
