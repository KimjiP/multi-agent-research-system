"""Claude calls for every step, with structured outputs and LangSmith tracing.

Every node calls Claude through these two functions. Structured calls return a
validated Pydantic object (the Messages API constrains the output to the schema),
so no node parses JSON by hand. Token usage is accumulated in `usage` so the
benchmarks can report the cost of each run.

LangSmith's wrap_anthropic does not support anthropic 1.x, so calls are traced
with @traceable instead. It does nothing unless tracing is enabled in .env.
"""

from dataclasses import dataclass
from typing import TypeVar

import anthropic
from langsmith import traceable
from pydantic import BaseModel

from src.config import MAX_TOKENS, MODEL_NAME, PRICE_PER_MTOK

T = TypeVar("T", bound=BaseModel)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


class LLMError(RuntimeError):
    """Claude returned no usable output: a refusal, a cut-off response, or no parsed object."""


@dataclass
class Usage:
    """Tokens used since the last reset, across all calls."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_PER_MTOK["input"]
            + self.output_tokens * PRICE_PER_MTOK["output"]
        ) / 1_000_000

    def reset(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0


usage = Usage()


def _record(response) -> None:
    usage.calls += 1
    usage.input_tokens += response.usage.input_tokens
    usage.output_tokens += response.usage.output_tokens
    if response.stop_reason == "refusal":
        raise LLMError("Claude declined the request")
    if response.stop_reason == "max_tokens":
        raise LLMError(f"Claude's output was cut off at max_tokens={MAX_TOKENS}")


@traceable(run_type="llm", name="claude")
def generate_text(system: str, prompt: str, effort: str) -> str:
    """Plain-text response."""
    response = _get_client().messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        system=system,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    _record(response)
    return "".join(block.text for block in response.content if block.type == "text")


@traceable(run_type="llm", name="claude_structured")
def generate_structured(system: str, prompt: str, output_model: type[T], effort: str) -> T:
    """Response parsed into `output_model`, with the output constrained to its schema."""
    response = _get_client().messages.parse(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        system=system,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
        output_format=output_model,
    )
    _record(response)
    if response.parsed_output is None:
        raise LLMError("Claude returned no structured output")
    return response.parsed_output
