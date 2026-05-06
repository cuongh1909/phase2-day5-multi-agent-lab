"""LangGraph workflow skeleton."""

from typing import Any

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import flush_langfuse, get_langfuse_client


def _ensure_research_state(raw: ResearchState | dict[str, Any]) -> ResearchState:
    if isinstance(raw, ResearchState):
        return raw
    if isinstance(raw, dict):
        return ResearchState(**raw)
    raise TypeError(f"Unexpected workflow result type: {type(raw)}")


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Create a LangGraph graph.

        TODO(student): Implement nodes, edges, conditional routing, and stop condition.
        Suggested nodes: supervisor, researcher, analyst, writer, optional critic.
        """

        from langgraph.graph import END, StateGraph

        from multi_agent_research_lab.agents.analyst import AnalystAgent
        from multi_agent_research_lab.agents.researcher import ResearcherAgent
        from multi_agent_research_lab.agents.supervisor import SupervisorAgent
        from multi_agent_research_lab.agents.writer import WriterAgent

        graph = StateGraph(ResearchState)

        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        graph.add_node("supervisor", supervisor.run)
        graph.add_node("researcher", researcher.run)
        graph.add_node("analyst", analyst.run)
        graph.add_node("writer", writer.run)

        graph.set_entry_point("supervisor")

        def _route(state: ResearchState) -> str:
            # Supervisor records the next route in `route_history`.
            if not state.route_history:
                return "supervisor"
            next_route = state.route_history[-1]
            if next_route == "done":
                return END
            return next_route

        graph.add_conditional_edges(
            "supervisor",
            _route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
                # Fallback: if supervisor didn't append, keep it safe.
                "supervisor": "supervisor",
            },
        )

        # After each worker finishes, ask supervisor what to do next.
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        TODO(student): Compile graph, invoke it, and convert result back to ResearchState.
        """

        graph = self.build()
        app = graph.compile()
        lf = get_langfuse_client()
        try:
            if lf is not None:
                with lf.start_as_current_observation(
                    name="MultiAgentWorkflow",
                    as_type="chain",
                    input=state.request.model_dump(),
                    metadata={"package": "multi_agent_research_lab"},
                ) as chain:
                    raw = app.invoke(state)
                    result = _ensure_research_state(raw)
                    chain.update(
                        output={
                            "route_history": result.route_history,
                            "iteration": result.iteration,
                            "total_input_tokens": result.total_input_tokens,
                            "total_output_tokens": result.total_output_tokens,
                            "has_final_answer": bool(result.final_answer),
                        }
                    )
            else:
                raw = app.invoke(state)
                result = _ensure_research_state(raw)
        finally:
            flush_langfuse()

        return result
