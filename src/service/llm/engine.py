"""
LLM engine — wraps any LLMProvider with system prompt injection and retry logic.

This is the only LLM interface that app.py should use.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

from src.service.llm.providers import get_llm_provider
from src.core.prompt import SYSTEM_PROMPT

log = logging.getLogger(__name__)


class LLMEngine:
    """
    Orchestrates an LLM provider.

    Responsibilities (engine-level):
      - Prepend the system prompt to every request
      - Retry on empty responses (known llama.cpp quirk)
      - Lifecycle management (start/stop the provider)

    The actual streaming is delegated to the selected provider.
    """

    def __init__(self, config) -> None:
        self._cfg = config.llm
        provider_cls = get_llm_provider(self._cfg.provider)
        self._provider = provider_cls(self._cfg.provider_config)

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._provider.start()

    async def stop(self) -> None:
        await self._provider.stop()

    async def __aenter__(self) -> LLMEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # ── generation ───────────────────────────────────────────────────

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        max_retries: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Prepend system prompt, stream via provider, retry on empty.

        Yields text chunks to the caller.
        """
        max_retries = max_retries or self._cfg.max_retries
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + messages

        for attempt in range(1, max_retries + 1):
            got_content = False
            t0 = time.perf_counter()

            async for chunk in self._provider.stream(full_messages):
                got_content = True
                yield chunk

            elapsed = time.perf_counter() - t0

            if got_content:
                log.info("LLM turn completed in %.2fs", elapsed)
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
