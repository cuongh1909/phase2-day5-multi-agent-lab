"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.observability.tracing import get_langfuse_client


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        TODO(student): Connect OpenAI, Azure OpenAI, or another provider.
        Keep retry, timeout, and token logging here rather than inside agents.
        """

        settings = self._settings
        if not settings.openai_api_key:
            raise StudentTodoError(
                "Missing OPENAI_API_KEY. Set it in your environment or `.env` (see `src/.../core/config.py`)."
            )

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise StudentTodoError(
                "Missing optional dependencies for LLM provider. "
                'Install with `python -m pip install -e ".[llm]"` (or `".[dev,llm]"`).'
            ) from exc

        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.timeout_seconds)

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
        )
        def _openai_call() -> LLMResponse:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            return LLMResponse(
                content=content,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                cost_usd=None,
            )

        try:
            lf = get_langfuse_client()
            if lf is not None:
                with lf.start_as_current_observation(
                    name="openai.chat.completions",
                    as_type="generation",
                    model=settings.openai_model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    metadata={"provider": "openai"},
                ) as generation:
                    result = _openai_call()
                    usage_details: dict[str, int] = {}
                    if result.input_tokens is not None:
                        usage_details["input"] = result.input_tokens
                    if result.output_tokens is not None:
                        usage_details["output"] = result.output_tokens
                    out_preview = result.content if len(result.content) <= 16_000 else result.content[:16_000]
                    generation.update(output=out_preview, usage_details=usage_details or None)
                    return result
            return _openai_call()
        except RetryError as exc:  # pragma: no cover
            raise AgentExecutionError(f"LLM call failed after retries: {exc}") from exc
        except Exception as exc:
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc
