from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_notes_missing() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    SupervisorAgent().run(state)
    assert state.route_history[-1] == "researcher"
    assert any(e["name"] == "route_decision" for e in state.trace)


def test_supervisor_routes_to_analyst_when_research_ready() -> None:
    from multi_agent_research_lab.core.schemas import SourceDocument

    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="S1", url="https://example.com", snippet="x")],
        research_notes="Some notes.",
    )
    SupervisorAgent().run(state)
    assert state.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_when_analysis_ready() -> None:
    from multi_agent_research_lab.core.schemas import SourceDocument

    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="S1", url=None, snippet="x")],
        research_notes="Notes.",
        analysis_notes="Analysis.",
    )
    SupervisorAgent().run(state)
    assert state.route_history[-1] == "writer"
