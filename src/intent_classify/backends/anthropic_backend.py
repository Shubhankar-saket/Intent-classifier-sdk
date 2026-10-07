"""Claude adapter (official ``anthropic`` SDK, structured outputs, cached system prompt)."""

from __future__ import annotations

import json
from typing import Any

from ..types import Usage
from .base import LLMError, LLMResponse

# USD per 1M tokens: (input, output, cache read). Cache writes (5-min TTL) are 1.25x input.
# Note: the minimum cacheable prefix is 512 tokens on Opus 5.5 / Sonnet 5.5 but 4096 on
# Haiku 4.5, so a small taxonomy prompt may silently not cache on Haiku.
PRICING: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}
# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# Models that reject output_config.effort.
_NO_EFFORT_MODELS = {"claude-haiku-4-5"}


class AnthropicBackend:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str | None = "low",
        max_tokens: int = 4096,
        fallbacks: bool = True,
        client: Any = None,
        **client_kwargs: Any,
    ):
        import anthropic

        self.model = model
        self.effort = None if model in _NO_EFFORT_MODELS else effort
        self.max_tokens = max_tokens  # headroom for thinking, which counts against it
        self.fallbacks = fallbacks and model in _FALLBACK_MODELS
        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic(**client_kwargs)

    def complete_json(self, system: str, user: str, schema: dict[str, Any], schema_name: str) -> LLMResponse:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # The taxonomy prompt is identical for every message of a project -> cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        if self.fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

        try:
            resp = self.client.beta.messages.create(**kwargs)
        except self._anthropic.BadRequestError as e:
            raise LLMError(f"anthropic rejected the request: {e.message}") from e
        except self._anthropic.APIStatusError as e:  # 429/5xx already retried by the SDK
            raise LLMError(f"anthropic API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError(f"anthropic connection error: {e}") from e

        usage = self._usage(resp)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return LLMResponse(data=None, usage=usage, refusal=category or "refusal")
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"hit max_tokens={self.max_tokens}; raise it or lower effort")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise LLMError(f"no text block in response (stop_reason={resp.stop_reason})")
        return LLMResponse(data=json.loads(text), usage=usage)

    def _usage(self, resp: Any) -> Usage:
        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            llm_calls=1,
        )
        price = PRICING.get(self.model)
        if price:
            p_in, p_out, p_cache_read = price
            usage.estimated_cost_usd = (
                usage.input_tokens * p_in
                + usage.cache_read_tokens * p_cache_read
                + usage.cache_write_tokens * p_in * 1.25
                + usage.output_tokens * p_out
            ) / 1e6
        return usage
