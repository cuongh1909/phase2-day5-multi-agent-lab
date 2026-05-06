"""Tracing hooks.

Provider-agnostic spans: LangSmith, Langfuse, OpenTelemetry, or JSON in `ResearchState.trace`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

_langfuse_client: object | None = None


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    sink: list[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Wall-clock span. On exit, sets ``duration_seconds`` and optionally appends to ``sink``."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": dict(attributes or {}),
        "duration_seconds": None,
        "started_at_perf": started,
    }
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        if sink is not None:
            sink.append(span)


def append_span_to_state(state: ResearchState, span: dict[str, Any]) -> None:
    """Record a completed span dict (e.g. from :func:`trace_span`) on shared state."""

    name = str(span.get("name", "span"))
    payload = {
        "duration_seconds": span.get("duration_seconds"),
        "attributes": span.get("attributes") or {},
    }
    state.add_trace_event(name, payload)


def append_spans_to_state(state: ResearchState, spans: Sequence[dict[str, Any]]) -> None:
    for sp in spans:
        append_span_to_state(state, sp)


def get_langfuse_client() -> object | None:
    """Return a Langfuse client if ``langfuse`` is installed and keys are set, else ``None``.

    Requires ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` (and optionally
    ``LANGFUSE_BASE_URL`` / ``LANGFUSE_HOST``).
    """

    global _langfuse_client
    try:
        from langfuse import Langfuse
    except ModuleNotFoundError:
        return None

    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    if _langfuse_client is None:
        base = (settings.langfuse_base_url or "").strip() or "https://cloud.langfuse.com"
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=base,
        )
    return _langfuse_client


def flush_langfuse() -> None:
    """Flush Langfuse buffers so CLI / short runs export spans before exit."""

    client = _langfuse_client
    if client is not None:
        flush = getattr(client, "flush", None)
        if callable(flush):
            flush()


def reset_langfuse_client_for_tests() -> None:
    """Drop cached Langfuse client (tests only)."""

    global _langfuse_client
    shutdown = getattr(_langfuse_client, "shutdown", None) if _langfuse_client else None
    if callable(shutdown):
        shutdown()
    _langfuse_client = None
