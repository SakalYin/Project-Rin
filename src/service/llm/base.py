"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class LLMProvider(ABC):
    """
    Base class for LLM providers.

    Providers handle raw API/network interaction to generate text.
    System prompt injection, retries, and logging are handled by the engine.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize from a provider-specific config dict.

        The dict contains only provider-specific keys extracted from the
        ``llm:`` section of config.yaml (everything except engine-level keys
        like ``provider`` and ``max_retries``).
        """
        ...

    @abstractmethod
    async def stream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Send messages and yield text chunks as they arrive.

        ``messages`` already includes the system prompt (prepended by engine).
        Must yield at least one non-empty string for a successful generation.
        """
        ...
        # Make this a valid async generator for type checking
        yield  # pragma: no cover

    async def start(self) -> None:
        """Optional lifecycle hook — called before first use."""

    async def stop(self) -> None:
        """Optional lifecycle hook — called on shutdown."""

    async def __aenter__(self) -> LLMProvider:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()
