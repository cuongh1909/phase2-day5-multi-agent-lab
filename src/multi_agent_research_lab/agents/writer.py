"""Writer agent: final answer from research + analysis."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = LLMClient(self._settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        research = (state.research_notes or "").strip()
        analysis = (state.analysis_notes or "").strip()

        bib: list[str] = []
        for i, s in enumerate(state.sources):
            url = f" — {s.url}" if s.url else ""
            bib.append(f"[{i + 1}] {s.title}{url}")

        source_block = "\n".join(bib) if bib else "(no numbered sources)"

        resp = self._llm.complete(
            system_prompt=(
                "You are the writer for a research assistant. Produce the final answer for the user.\n"
                "- Match the user request (length, format) when specified.\n"
                "- Ground claims in the provided research and analysis; prefer cautious wording when evidence is weak.\n"
                "- Use numbered bracket citations [1], [2] that refer to the provided source list only.\n"
                "- End with a 'Sources' section listing the same numbers.\n"
                "Do not invent URLs or sources beyond the list given."
            ),
            user_prompt=(
                f"Audience: {state.request.audience}\n\n"
                f"Question:\n{state.request.query}\n\n"
                f"Source list (for citations):\n{source_block}\n\n"
                f"Research notes:\n{research or '(none)'}\n\n"
                f"Analysis:\n{analysis or '(none)'}"
            ),
        )
        state.final_answer = resp.content
        state.record_llm_usage(resp.input_tokens, resp.output_tokens)
        state.append_agent_result(
            AgentName.WRITER,
            resp.content,
            metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
        )
        state.add_trace_event(
            "writer_done",
            {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
        )
        return state
