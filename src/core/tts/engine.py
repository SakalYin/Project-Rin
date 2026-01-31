"""
Kokoro-ONNX text-to-speech engine with sounddevice playback.

Synthesis uses kokoro_onnx.Kokoro (v1.0 model files).
Blocking calls (create / sd.play) are offloaded to a thread-pool executor
so the async event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from functools import partial

import numpy as np
import sounddevice as sd

from src.core.config import AppConfig

log = logging.getLogger(__name__)

# ── Sentence splitting helpers ───────────────────────────────────────────

# Split on sentence-ending punctuation followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?~])\s+")

# Kaomoji / emote patterns that shouldn't be spoken.
_KAOMOJI = re.compile(
    r"[\(（][\s]*[>≧≦╥ᗒᗣᗕ°▽TQOoUuXx;:'^,.*_\-\+~!?ಥ☆★♡♥ω\\/|]+"
    r"[^\)）]*[\)）]"
)
# Leftover decorative symbols.
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
    """Wraps kokoro-onnx synthesis + sounddevice playback."""

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config.tts
        self._kokoro = None  # lazy-loaded on first speak()

    # ── lazy loading ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Import and instantiate Kokoro on first use."""
        if self._kokoro is not None:
            return
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise RuntimeError(
                "kokoro-onnx is not installed. Run:  pip install kokoro-onnx"
            )
        log.info("Loading Kokoro model from %s …", self._cfg.model_path)
        self._kokoro = Kokoro(self._cfg.model_path, self._cfg.voices_path)
        log.info("Kokoro model loaded.")

    # ── blocking helpers (run in executor) ───────────────────────────

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous synthesis — run inside ``run_in_executor``."""
        self._ensure_loaded()
        samples, sr = self._kokoro.create(
            text,
            voice=self._cfg.voice,
            speed=self._cfg.speed,
            lang=self._cfg.lang,
        )
        return samples, sr

    @staticmethod
    def _play_audio(samples: np.ndarray, sample_rate: int) -> None:
        """Synchronous playback — run inside ``run_in_executor``."""
        sd.play(samples, samplerate=sample_rate)
        sd.wait()

    # ── public async API ─────────────────────────────────────────────

    async def speak(self, text: str) -> None:
        """
        Clean *text*, synthesize it with Kokoro, and play through speakers.

        Both synthesis and playback are offloaded to the default thread-pool
        so the event loop is never blocked.
        """
        clean = _clean_for_speech(text)
        if not clean:
            log.debug("Nothing speakable in: %r", text)
            return

        loop = asyncio.get_running_loop()

        # ── synthesize ───────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            samples, sr = await loop.run_in_executor(
                None, partial(self._synthesize, clean)
            )
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
