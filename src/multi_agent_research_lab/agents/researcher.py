"""Researcher agent: gather sources and draft research notes."""

from __future__ import annotations

import httpx

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import get_langfuse_client
from multi_agent_research_lab.services.llm_client import LLMClient


def _tavily_search(settings: Settings, query: str, max_results: int) -> list[SourceDocument]:
    if not settings.tavily_api_key:
        return []

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
    }

    def _execute() -> list[SourceDocument]:
        with httpx.Client(timeout=settings.timeout_seconds) as client:
            response = client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()

        out: list[SourceDocument] = []
        for i, item in enumerate(data.get("results", [])):
            title = (item.get("title") or "").strip() or f"Result {i + 1}"
            snippet = (item.get("content") or item.get("raw_content") or "").strip()
            url = item.get("url")
            if isinstance(url, str):
                url = url.strip() or None
            out.append(SourceDocument(title=title, url=url, snippet=snippet))
        return out

    lf = get_langfuse_client()
    if lf is None:
        return _execute()

    with lf.start_as_current_observation(
        name="tavily.search",
        as_type="retriever",
        input={"query": query, "max_results": max_results},
        metadata={"provider": "tavily"},
    ) as retriever:
        found = _execute()
        retriever.update(output={"result_count": len(found), "titles": [s.title for s in found]})
        return found


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = LLMClient(self._settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        max_n = min(state.request.max_sources, 10)
        query = state.request.query

        sources = _tavily_search(self._settings, query, max_n)

        if sources:
            lines: list[str] = []
            for i, s in enumerate(sources):
                ref = f"[{i + 1}]"
                url_part = f" ({s.url})" if s.url else ""
                lines.append(f"{ref} **{s.title}**{url_part}\n{s.snippet}")
            context = "\n\n".join(lines)
            resp = self._llm.complete(
                system_prompt=(
                    "You are a research assistant. Given search snippets, write clear research notes: "
                    "bullets, key entities, definitions, and open questions. "
                    "Reference sources using [1], [2] markers matching the provided list. "
                    "Do not invent URLs or quotes not implied by the snippets."
                ),
                user_prompt=f"User query:\n{query}\n\nSources:\n{context}",
            )
            state.sources = sources
            state.research_notes = resp.content
            state.record_llm_usage(resp.input_tokens, resp.output_tokens)
            state.append_agent_result(
                AgentName.RESEARCHER,
                resp.content,
                metadata={
                    "source_count": len(sources),
                    "used_tavily": True,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                },
            )
            state.add_trace_event(
                "researcher_done",
                {"sources": len(sources), "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
            )
            return state

        resp = self._llm.complete(
            system_prompt=(
                "You are a research assistant without live web access. "
                "Produce careful research notes from general knowledge: bullets, definitions, typical use cases, "
                "and limitations. Mark uncertainty explicitly. Do not fabricate specific paper titles, URLs, or "
                "statistics. Add a short 'Gaps' section listing what live search would clarify."
            ),
            user_prompt=(
                f"Topic / question:\n{query}\n\n"
                f"Audience: {state.request.audience}\n"
                f"Target length: roughly {max(300, max_n * 80)} words of notes."
            ),
        )
        state.sources = [
            SourceDocument(
                title="Model-derived desk research (no live web)",
                url=None,
                snippet=resp.content[:4000],
            )
        ]
        state.research_notes = resp.content
        state.record_llm_usage(resp.input_tokens, resp.output_tokens)
        state.append_agent_result(
            AgentName.RESEARCHER,
            resp.content,
            metadata={
                "source_count": 1,
                "used_tavily": False,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            },
        )
        state.add_trace_event(
            "researcher_done",
            {"sources": 1, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
        )
        return state
