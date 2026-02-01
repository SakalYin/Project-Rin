"""Faster-whisper STT provider (distil-medium.en on GPU)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from src.core.stt.base import STTProvider

log = logging.getLogger(__name__)


class FasterWhisperProvider(STTProvider):
    """Transcribes audio using faster-whisper (CTranslate2 backend)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._model_name = config.get("model", "distil-medium.en")
        self._device = config.get("device", "cuda")
        self._compute_type = config.get("compute_type", "float16")
        self._language = config.get("language", "en")
        self._model = None

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load)

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        log.info(
            "Loading Whisper model=%s device=%s compute=%s …",
            self._model_name,
            self._device,
            self._compute_type,
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )
        log.info("Whisper loaded in %.1fs", time.perf_counter() - t0)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if len(audio) == 0:
            return ""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.info(
            "Transcribed (lang=%s prob=%.2f): %r",
            info.language,
            info.language_probability,
            text,
        )
        return text

    async def shutdown(self) -> None:
        self._model = None
