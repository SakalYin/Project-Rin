"""
OpenAI-compatible LLM provider.

Works with any server that exposes the OpenAI chat-completions API
(llama.cpp, vLLM, Ollama in OpenAI mode, etc.).

Optionally manages a llama.cpp server subprocess via the ``server``
config block.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from src.service.llm.base import LLMProvider
from src.service.llm.providers._llama_server import LlamaServer

log = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._server_cfg = config.get("server", {})
        self._server: LlamaServer | None = None

        self._model = config.get("model", "local-model")
        self._temperature = config.get("temperature", 1.1)
        self._max_tokens = config.get("max_tokens")
        self._frequency_penalty = config.get("frequency_penalty", 0.0)
        self._presence_penalty = config.get("presence_penalty", 0.0)
        self._timeout = config.get("timeout", 30.0)

        # Build initial client — may be replaced in start() if server
        # provides its own host/port.
        self._client = self._make_client(
            config.get("base_url", "http://localhost:8080/v1"),
        )

    def _make_client(self, base_url: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=base_url,
            api_key=self._config.get("api_key", "not-needed"),
            timeout=self._timeout,
        )

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._server_cfg.get("enabled", False):
            self._server = LlamaServer(self._server_cfg)
            await self._server.start()
            # Re-create client to point at the server's actual address
            host = self._server_cfg.get("host", "127.0.0.1")
            port = self._server_cfg.get("port", 8080)
            self._client = self._make_client(
                f"http://{host}:{port}/v1"
            )

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop()
            self._server = None

    # ── streaming ────────────────────────────────────────────────────

    async def stream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()
        token_count = 0
        finish_reason = None

        create_kwargs: dict = dict(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            stream=True,
        )
        if self._max_tokens is not None:
            create_kwargs["max_tokens"] = self._max_tokens

        try:
            stream = await self._client.chat.completions.create(**create_kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    log.debug("Chunk with empty choices: %s", chunk)
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                text = getattr(choice.delta, "content", None)
                if text:
                    token_count += 1
                    yield text

        except Exception as e:
            log.error("LLM streaming failed: %s", e)
            raise

        elapsed = time.perf_counter() - t0
        log.info(
            "LLM done — %d chunks in %.2fs (%.1f tok/s) [finish_reason=%s]",
            token_count,
            elapsed,
            token_count / elapsed if elapsed > 0 else 0,
            finish_reason,
        )
        if finish_reason == "length":
            log.warning(
                "Response was TRUNCATED (hit token limit). "
                "Increase server.n_predict in config.yaml for longer replies.",
            )
