"""Analyst agent: structure insights from research notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = LLMClient(self._settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        notes = (state.research_notes or "").strip()
        if not notes:
            state.analysis_notes = "No research notes available to analyze."
            state.append_agent_result(AgentName.ANALYST, state.analysis_notes, metadata={"skipped": True})
            state.add_trace_event("analyst_done", {"skipped": True})
            return state

        source_lines = [f"- {s.title}" + (f" ({s.url})" if s.url else "") for s in state.sources]
        source_block = "\n".join(source_lines) if source_lines else "(no separate source list)"

        resp = self._llm.complete(
            system_prompt=(
                "You are an analyst. From the research notes, produce structured insights.\n"
                "Use markdown with these sections exactly:\n"
                "## Key claims\n"
                "## Evidence quality\n"
                "## Contradictions or debates\n"
                "## Risks and limitations\n"
                "## Suggested angles for the final answer\n"
                "Flag weak or missing evidence. Do not invent citations."
            ),
            user_prompt=(
                f"Original question:\n{state.request.query}\n\n"
                f"Source titles:\n{source_block}\n\n"
                f"Research notes:\n{notes}"
            ),
        )
        state.analysis_notes = resp.content
        state.record_llm_usage(resp.input_tokens, resp.output_tokens)
        state.append_agent_result(
            AgentName.ANALYST,
            resp.content,
            metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
        )
        state.add_trace_event(
            "analyst_done",
            {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
        )
        return state
