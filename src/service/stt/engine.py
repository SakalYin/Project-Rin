"""
STT engine — wraps any STTProvider with Silero VAD-based speech detection
and background transcription.

This is the only STT interface that app.py should use.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

import numpy as np

from src.service.asr.asr import ASRProcessor
from src.service.stt.providers import get_stt_provider

log = logging.getLogger(__name__)


class STTEngine:
    """
    Orchestrates an STT provider with active (always-on) listening.

    Uses Silero VAD (via ASRProcessor) for speech detection.  Detected
    speech segments are sent to the selected STT provider for
    transcription and pushed to ``input_queue``.
    """

    def __init__(self, config) -> None:
        self._cfg = config.stt
        provider_cls = get_stt_provider(self._cfg.provider)
        self._provider = provider_cls(self._cfg.provider_config)

        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._listener_task: asyncio.Task | None = None
        self._asr: ASRProcessor | None = None
        self._asr_thread: threading.Thread | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load the STT provider's model (heavy — do once at startup)."""
        await self._provider.initialize()

    async def start_listening(self) -> None:
        """Create ASRProcessor and launch background VAD + transcription."""
        if self._running:
            return
        self._running = True

        self._asr = ASRProcessor(
            buffer_span=int(self._cfg.max_duration),
            long_pause_thres=self._cfg.silence_duration,
            vad_threshold=self._cfg.vad_threshold,
        )
        self._asr_thread = threading.Thread(
            target=self._asr.process_audio_stream, daemon=True
        )
        self._asr_thread.start()

        self._listener_task = asyncio.create_task(self._listener_loop())
        log.info("STT active listening started (Silero VAD).")

    async def stop_listening(self) -> None:
        """Stop the ASR processor and background listener."""
        self._running = False
        if self._asr is not None:
            self._asr.stop()
        if self._asr_thread is not None:
            self._asr_thread.join(timeout=5)
            self._asr_thread = None
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        await self._provider.shutdown()
        log.info("STT active listening stopped.")

    # ── background listener ──────────────────────────────────────────

    async def _listener_loop(self) -> None:
        """Pull speech segments from ASRProcessor and transcribe them."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                audio = await loop.run_in_executor(
                    None,
                    lambda: self._asr.speech_segments_queue.get(timeout=0.5),
                )
            except queue.Empty:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Error getting speech segment: %s", e)
                continue

            if len(audio) == 0:
                continue

            try:
                text = await self._provider.transcribe(
                    audio, self._cfg.sample_rate
                )
            except Exception as e:
                log.error("Transcription error: %s", e)
                continue

            text = text.strip()
            if text:
                log.info("STT heard: %r", text)
                await self.input_queue.put(text)
