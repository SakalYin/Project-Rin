"""Kokoro-ONNX TTS provider."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from src.core.tts.base import TTSProvider

log = logging.getLogger(__name__)


class KokoroProvider(TTSProvider):
    """Synthesizes speech using kokoro-onnx (v1.0 model files)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._model_path = config.get("model_path", "models/kokoro-v1.0.onnx")
        self._voices_path = config.get("voices_path", "models/voices-v1.0.bin")
        self._voice = config.get("voice", "af_heart")
        self._speed = config.get("speed", 1.1)
        self._lang = config.get("lang", "en-us")
        self._kokoro = None

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load)

    def _load(self) -> None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise RuntimeError(
                "kokoro-onnx is not installed. Run:  pip install kokoro-onnx"
            )
        log.info("Loading Kokoro model from %s …", self._model_path)
        t0 = time.perf_counter()
        self._kokoro = Kokoro(self._model_path, self._voices_path)
        log.info("Kokoro loaded in %.1fs", time.perf_counter() - t0)

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synth, text)

    def _synth(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        )
        return samples, sr

    async def shutdown(self) -> None:
        self._kokoro = None
