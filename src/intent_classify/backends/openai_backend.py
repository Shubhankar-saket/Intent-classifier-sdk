"""OpenAI-compatible adapter (OpenAI, Azure OpenAI, vLLM, Ollama, ... via ``base_url`` or ``client``)."""

from __future__ import annotations

import json
from typing import Any

from ..types import Usage
from .base import LLMError, LLMResponse


class OpenAIBackend:
    name = "openai"

    def __init__(
        self,
        model: str,
        client: Any = None,
        price_per_mtok: tuple[float, float] | None = None,
        reasoning_effort: str | None = None,  # reasoning models only, e.g. "low"
        **client_kwargs: Any,
    ):
        import openai

        self.model = model
        self.price = price_per_mtok
        self.reasoning_effort = reasoning_effort
        self._openai = openai
        self.client = client or openai.OpenAI(**client_kwargs)

    def complete_json(self, system: str, user: str, schema: dict[str, Any], schema_name: str) -> LLMResponse:
        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        try:
            resp = self.client.chat.completions.create(
                **extra,
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
            )
        except self._openai.APIStatusError as e:
            raise LLMError(f"openai API error {e.status_code}: {e.message}") from e
        except self._openai.APIConnectionError as e:
            raise LLMError(f"openai connection error: {e}") from e

        msg = resp.choices[0].message
        usage = Usage(llm_calls=1)
        if resp.usage:
            cached = getattr(getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            usage.input_tokens = resp.usage.prompt_tokens - cached
            usage.cache_read_tokens = cached
            usage.output_tokens = resp.usage.completion_tokens
            if self.price:
                p_in, p_out = self.price
                usage.estimated_cost_usd = (
                    usage.input_tokens * p_in + cached * p_in * 0.5 + usage.output_tokens * p_out
                ) / 1e6
        if getattr(msg, "refusal", None):
            return LLMResponse(data=None, usage=usage, refusal="refusal")
        if not msg.content:
            raise LLMError(f"empty response (finish_reason={resp.choices[0].finish_reason})")
        return LLMResponse(data=json.loads(msg.content), usage=usage)
