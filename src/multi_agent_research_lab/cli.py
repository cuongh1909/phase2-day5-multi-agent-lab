"""Command-line entrypoint for the lab starter."""

import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import flush_langfuse
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    try:
        llm = LLMClient()
        started = time.perf_counter()
        resp = llm.complete(
            system_prompt=(
                "You are a helpful research assistant. Provide a concise, accurate summary. "
                "If uncertain, state uncertainty explicitly. Do not fabricate citations."
            ),
            user_prompt=request.query,
        )
        latency = time.perf_counter() - started

        state.final_answer = resp.content
        state.add_trace_event(
            "baseline_metrics",
            {
                "latency_seconds": latency,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "estimated_cost_usd": resp.cost_usd,
            },
        )

        subtitle = f"{latency:.2f}s"
        if resp.input_tokens is not None or resp.output_tokens is not None:
            subtitle += f" | tokens in/out: {resp.input_tokens}/{resp.output_tokens}"
        console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline", subtitle=subtitle))
    finally:
        flush_langfuse()


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    finally:
        flush_langfuse()
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
