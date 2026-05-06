"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        TODO(student): Implement routing policy. Suggested steps:
        - Inspect request, current notes, and missing fields.
        - Choose one of: researcher, analyst, writer, done.
        - Enforce max iterations and failure fallback.
        """

        settings = get_settings()

        # Stop conditions.
        if state.final_answer:
            state.record_route("done")
            state.add_trace_event("route_decision", {"route": "done", "reason": "final_answer_present"})
            return state

        if state.iteration >= settings.max_iterations:
            state.record_route("done")
            state.add_trace_event(
                "route_decision",
                {"route": "done", "reason": "max_iterations_reached", "max_iterations": settings.max_iterations},
            )
            return state

        # Simple, state-based routing policy:
        # - Need sources/notes? run researcher.
        # - Have research notes but no analysis? run analyst.
        # - Have analysis but no final answer? run writer.
        if not state.sources or not state.research_notes:
            route = "researcher"
            reason = "missing_sources_or_research_notes"
        elif not state.analysis_notes:
            route = "analyst"
            reason = "missing_analysis_notes"
        else:
            route = "writer"
            reason = "ready_to_write"

        state.record_route(route)
        state.add_trace_event(
            "route_decision",
            {
                "route": route,
                "reason": reason,
                "iteration": state.iteration,
                "max_iterations": settings.max_iterations,
            },
        )
        return state
