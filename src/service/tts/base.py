"""Abstract base class for TTS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TTSProvider(ABC):
    """
    Base class for TTS providers.

    Providers handle raw speech synthesis only.
    Text cleaning, sentence splitting, and audio playback are handled
    by the engine.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize from a provider-specific config dict.

        The dict contains only provider-specific keys extracted from the
        ``tts:`` section of config.yaml.
        """
        ...

    @abstractmethod
    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize already-cleaned text into audio.

        Returns ``(samples, sample_rate)`` where *samples* is a 1-D
        float32 numpy array.
        """
        ...

    async def initialize(self) -> None:
        """Optional: load models, warm up, etc. Called once at startup."""

    async def shutdown(self) -> None:
        """Optional: release resources."""
