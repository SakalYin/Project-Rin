"""
Async streaming client for a local llama.cpp server (OpenAI-compatible API).
"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

from openai import AsyncOpenAI

from src.core.config import AppConfig
from src.core.prompt import SYSTEM_PROMPT

log = logging.getLogger(__name__)


class LLMClient:
    """Thin async wrapper around the OpenAI client pointing at llama.cpp."""

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config.llm
        self._client = AsyncOpenAI(
            base_url=self._cfg.base_url,
            api_key=self._cfg.api_key,
            timeout=self._cfg.timeout,
        )

    async def stream_response(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Yield text chunks as they arrive from the LLM.

        *messages* should already include chat history; this method prepends
        the system prompt automatically.
        """
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        t0 = time.perf_counter()
        token_count = 0
        finish_reason = None

        # Build request kwargs — only include max_tokens if explicitly set,
        # otherwise the server's --n-predict default applies.
        create_kwargs: dict = dict(
            model=self._cfg.model,
            messages=full_messages,
            temperature=self._cfg.temperature,
            stream=True,
        )
        if self._cfg.max_tokens is not None:
            create_kwargs["max_tokens"] = self._cfg.max_tokens

        try:
            stream = await self._client.chat.completions.create(**create_kwargs)

            async for chunk in stream:
                # Guard against empty choices (llama.cpp sometimes sends these)
                if not chunk.choices:
                    log.debug("Chunk with empty choices: %s", chunk)
                    continue

                choice = chunk.choices[0]

                # Track why generation stopped
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                # delta.content can be None, "", or actual text
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

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        max_retries: int = 3,
    ) -> AsyncGenerator[str, None]:
        """
        Like ``stream_response`` but retries up to *max_retries* times if
        the model returns an empty reply (a known llama.cpp quirk).

        Yields text chunks to the caller. On retry the generator simply
        starts a fresh stream — the caller sees a seamless sequence of chunks.
        """
        for attempt in range(1, max_retries + 1):
            got_content = False

            async for chunk in self.stream_response(messages):
                got_content = True
                yield chunk

            if got_content:
                return

            if attempt < max_retries:
                log.warning(
                    "LLM returned empty response — retrying (%d/%d)",
                    attempt,
                    max_retries,
                )
            else:
                log.error(
                    "LLM returned empty after %d attempts", max_retries
                )
