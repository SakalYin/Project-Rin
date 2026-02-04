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

import numpy as np
import sounddevice as sd

from src.service.tts.providers import get_tts_provider

log = logging.getLogger(__name__)

# ── Sentence splitting helpers ───────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?~])\s+")

_KAOMOJI = re.compile(
    r"[\(（][\s]*[>≧≦╥ᗒᗣᗕ°▽TQOoUuXx;:'^,.*_\-\+~!?ಥ☆★♡♥ω\\/|]+"
    r"[^\)）]*[\)）]"
)

# Symbols that can't be spoken
_UNSPEAKABLE = re.compile(r"[>:;()\[\]{}<>^_~=*#@|/\\`\"]+")

# Asterisk actions like *sighs* or *winks*
_ASTERISK_ACTIONS = re.compile(r"\*[^*]+\*")

# Emoji pattern (covers most Unicode emoji ranges)
_EMOJI = re.compile(
    r"[\U0001F300-\U0001F9FF]"  # Misc symbols, emoticons, dingbats
    r"|[\U0001FA00-\U0001FAFF]"  # Extended symbols
    r"|[\U00002600-\U000027BF]"  # Misc symbols
    r"|[\U0001F600-\U0001F64F]"  # Emoticons
    r"|[\U0001F680-\U0001F6FF]"  # Transport/map
    r"|[\U0001F1E0-\U0001F1FF]"  # Flags
    r"|[\U00002700-\U000027BF]"  # Dingbats
    r"|[\U0000FE00-\U0000FE0F]"  # Variation selectors
    r"|[\U0001F900-\U0001F9FF]"  # Supplemental symbols
)

# Common text emoticons
_TEXT_EMOTICONS = (
    "uwu", "owo", "OwO", "UwU", ":3", ">:3", "T_T", "o_o", "O_O",
    ":)", ":(", ":D", ":P", ";)", "xD", "XD", ":O", ":/", ":|",
    "<3", "</3", "^^", "^_^", "-_-", ">_<", "._.", "o.o",
)


def _clean_for_speech(text: str) -> str:
    """
    Normalize text for TTS by removing unspeakable elements.

    Removes:
      - Kaomoji (Japanese emoticons)
      - Asterisk actions (*sighs*, *winks*)
      - Emojis
      - Text emoticons (uwu, :3, etc.)
      - Unspeakable symbols
    """
    # Remove kaomoji
    text = _KAOMOJI.sub("", text)

    # Remove asterisk actions like *sighs* or *winks*
    text = _ASTERISK_ACTIONS.sub("", text)

    # Remove emojis
    text = _EMOJI.sub("", text)

    # Remove text emoticons
    for emote in _TEXT_EMOTICONS:
        text = text.replace(emote, "")

    # Remove unspeakable symbols
    text = _UNSPEAKABLE.sub(" ", text)

    # Collapse multiple spaces
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

    async def _play_audio_async(self, samples: np.ndarray, sample_rate: int) -> None:
        """
        Non-blocking audio playback using OutputStream with finished callback.

        Uses an asyncio.Event to signal completion, allowing the event loop
        to remain fully responsive (GUI updates work) while audio plays.
        """
        loop = asyncio.get_running_loop()
        finished_event = asyncio.Event()

        # Track playback position
        position = 0
        samples_len = len(samples)

        def audio_callback(outdata, frames, time_info, status):
            nonlocal position
            if status:
                log.warning("Audio callback status: %s", status)

            chunk_end = position + frames
            if chunk_end <= samples_len:
                outdata[:, 0] = samples[position:chunk_end]
                position = chunk_end
            else:
                # End of audio — fill remaining with zeros
                valid = samples_len - position
                if valid > 0:
                    outdata[:valid, 0] = samples[position:]
                outdata[valid:, 0] = 0
                position = samples_len
                raise sd.CallbackStop()

        def finished_callback():
            # Called from audio thread when stream finishes
            loop.call_soon_threadsafe(finished_event.set)

        try:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                callback=audio_callback,
                finished_callback=finished_callback,
            )
            with stream:
                stream.start()
                await finished_event.wait()
        except Exception as e:
            log.error("Audio playback error: %s", e)
            raise

    async def speak(self, text: str) -> None:
        """
        Clean *text*, synthesize via provider, and play through speakers.

        Uses non-blocking playback so the GUI remains responsive.
        """
        clean = _clean_for_speech(text)
        if not clean:
            log.debug("Nothing speakable in: %r", text)
            return

        # ── synthesize via provider ──────────────────────────────────
        t0 = time.perf_counter()
        try:
            samples, sr = await self._provider.synthesize(clean)
        except Exception as e:
            log.error("TTS synthesis failed: %s", e)
            return

        synth_time = time.perf_counter() - t0
        duration = len(samples) / sr
        log.info(
            "TTS synthesised %.1fs audio in %.2fs | text=%r",
            duration, synth_time, clean,
        )

        # ── play (non-blocking) ───────────────────────────────────────
        try:
            await self._play_audio_async(samples, sr)
        except Exception as e:
            log.error("TTS playback failed: %s", e)
