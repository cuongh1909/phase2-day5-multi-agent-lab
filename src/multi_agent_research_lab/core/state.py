"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery, SourceDocument


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None

    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)

    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def record_route(self, route: str) -> None:
        self.route_history.append(route)
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def record_llm_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is not None:
            self.total_input_tokens += input_tokens
        if output_tokens is not None:
            self.total_output_tokens += output_tokens

    def append_agent_result(
        self,
        agent: AgentName,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent_results.append(
            AgentResult(agent=agent, content=content, metadata=dict(metadata or {}))
        )
