"""Abstract base class for STT providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class STTProvider(ABC):
    """
    Base class for STT providers.

    Providers handle raw audio transcription only.
    Mic recording, VAD, silence detection, and the active listening loop
    are handled by the engine.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize from a provider-specific config dict.

        The dict contains only provider-specific keys extracted from the
        ``stt:`` section of config.yaml.
        """
        ...

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe raw audio to text.

        *audio* is a 1-D float32 numpy array at *sample_rate* Hz.
        Returns transcribed text (empty string if nothing detected).
        """
        ...

    async def initialize(self) -> None:
        """Optional: load models onto GPU, etc. Called once at startup."""

    async def shutdown(self) -> None:
        """Optional: release resources."""
