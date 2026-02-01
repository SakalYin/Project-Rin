"""
TTS engine — wraps any TTSProvider with text cleaning, sentence splitting,
and audio playback.

This is the only TTS interface that app.py should use.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from functools import partial
import numpy as np
import sounddevice as sd

from src.core.tts.providers import get_tts_provider

log = logging.getLogger(__name__)

# ── Sentence splitting helpers ───────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?~])\s+")

_KAOMOJI = re.compile(
    r"[\(（][\s]*[>≧≦╥ᗒᗣᗕ°▽TQOoUuXx;:'^,.*_\-\+~!?ಥ☆★♡♥ω\\/|]+"
    r"[^\)）]*[\)）]"
)
_UNSPEAKABLE = re.compile(r"[>:;()\[\]{}<>^_~=*#@|/\\]+")


def _clean_for_speech(text: str) -> str:
    """Strip kaomoji, emoticons, and other unspeakable symbols."""
    text = _KAOMOJI.sub("", text)
    for emote in ("uwu", "owo", "OwO", "UwU", ":3", ">:3", "T_T"):
        text = text.replace(emote, "")
    text = _UNSPEAKABLE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def extract_sentences(buffer: str, min_chars: int = 12) -> tuple[list[str], str]:
    """
    Split *buffer* into complete sentences and leftover text.

    Returns ``(sentences, remaining)``. Sentences shorter than *min_chars*
    are merged with the next one to avoid tiny TTS fragments.
    """
    parts = _SENTENCE_SPLIT.split(buffer)
    if len(parts) <= 1:
        return [], buffer

    sentences: list[str] = []
    accum = ""
    for part in parts[:-1]:
        accum = (accum + " " + part).strip() if accum else part.strip()
        if len(accum) >= min_chars:
            sentences.append(accum)
            accum = ""

    remaining = parts[-1]
    if accum:
        remaining = accum + " " + remaining

    return sentences, remaining.strip()


# ── TTS Engine ───────────────────────────────────────────────────────────

class TTSEngine:
    """
    Orchestrates a TTS provider.

    Responsibilities (engine-level):
      - Text cleaning (strip kaomoji, unspeakable symbols)
      - Audio playback via sounddevice

    Synthesis is delegated to the selected provider.
    """

    def __init__(self, config) -> None:
        self._cfg = config.tts
        provider_cls = get_tts_provider(self._cfg.provider)
        self._provider = provider_cls(self._cfg.provider_config)

    async def initialize(self) -> None:
        """Load the TTS provider's model."""
        await self._provider.initialize()

    async def shutdown(self) -> None:
        """Release provider resources."""
        await self._provider.shutdown()

    @staticmethod
    def _play_audio(samples: np.ndarray, sample_rate: int) -> None:
        """Synchronous playback — run inside ``run_in_executor``."""
        sd.play(samples, samplerate=sample_rate)
        sd.wait()

    async def speak(self, text: str) -> None:
        """
        Clean *text*, synthesize via provider, and play through speakers.

        Both synthesis and playback are offloaded so the event loop stays free.
        """
        clean = _clean_for_speech(text)
        if not clean:
            log.debug("Nothing speakable in: %r", text)
            return

        loop = asyncio.get_running_loop()

        # ── synthesize via provider ──────────────────────────────────
        t0 = time.perf_counter()
        try:
            samples, sr = await self._provider.synthesize(clean)
        except Exception as e:
            log.error("TTS synthesis failed: %s", e)
            print(f"\n[TTS ERROR] synthesis failed: {e}")
            return

        synth_time = time.perf_counter() - t0
        duration = len(samples) / sr
        log.info(
            "TTS synthesised %.1fs audio in %.2fs | text=%r",
            duration, synth_time, clean,
        )

        # ── play ─────────────────────────────────────────────────────
        try:
            await loop.run_in_executor(
                None, partial(self._play_audio, samples, sr)
            )
        except Exception as e:
            log.error("TTS playback failed: %s", e)
            print(f"\n[TTS ERROR] playback failed: {e}")
