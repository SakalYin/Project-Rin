"""
Conversation turn handler — orchestrates LLM, TTS, and DB.

Handles:
  - LLM streaming with concurrent TTS playback
  - Memory system integration (StatusManager + background updates)
  - Persisting messages to the database
  - UI output (GUI window or terminal)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from src.service.llm.engine import LLMEngine
from src.service.tts.engine import TTSEngine, extract_sentences
from src.db.db_manager import ChatDatabase
from src.core.config import AppConfig
from src.utils.plugins.dynamic_personality import (
    StatusManager,
    update_memory_background,
)

if TYPE_CHECKING:
    from src.core.terminal_ui import TerminalUI
    from src.core.chat_window import AsyncChatWindow

log = logging.getLogger(__name__)


def _build_memory_context(status: StatusManager | None) -> str | None:
    """Build the memory context to inject into system prompt."""
    if status is None:
        return None

    # Just inject the current memory state (no instructions needed)
    memory_block = status.format_for_prompt()
    if memory_block:
        log.debug("Memory context injected (%d chars)", len(memory_block))
    return memory_block or None


async def run_turn_with_window(
    user_input: str,
    llm: LLMEngine,
    tts: TTSEngine,
    db: ChatDatabase,
    config: AppConfig,
    window: AsyncChatWindow,
    status: StatusManager | None = None,
) -> str:
    """
    Run one conversation turn with GUI window output.

    Streams LLM tokens while concurrently playing TTS for finished sentences.
    A third task keeps the GUI responsive during playback.
    Memory is updated via background LLM call after the turn.
    """
    history = await db.get_history()
    history.append({"role": "user", "content": user_input})

    # Build memory context for LLM
    memory_context = _build_memory_context(status)
    if memory_context:
        log.info("[MEMORY] Injecting %d chars into system prompt", len(memory_context))

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    full_reply_parts: list[str] = []
    turn_done = asyncio.Event()

    async def _stream_llm() -> None:
        buffer = ""
        try:
            async for chunk in llm.generate_response(
                history,
                max_retries=config.llm.max_retries,
                memory_context=memory_context,
            ):
                full_reply_parts.append(chunk)
                window.append_ai_chunk(chunk)
                buffer += chunk

                sentences, buffer = extract_sentences(
                    buffer, config.streaming.min_sentence_chars
                )
                for sentence in sentences:
                    await sentence_queue.put(sentence)
        except Exception as e:
            log.error("LLM stream error: %s", e)
            window.show_error(f"LLM error: {e}")

        leftover = buffer.strip()
        if leftover:
            await sentence_queue.put(leftover)
        await sentence_queue.put(None)

    async def _speak_sentences() -> None:
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            await tts.speak(sentence)

    async def _update_gui() -> None:
        """Keep GUI responsive while LLM streams and TTS plays."""
        while not turn_done.is_set():
            if window._window and not window._window.is_closed:
                window._window.update()
            await asyncio.sleep(0.016)  # ~60fps

    t0 = time.perf_counter()

    window.start_ai_message()

    # Run LLM streaming, TTS playback, and GUI updates concurrently
    gui_task = asyncio.create_task(_update_gui())
    try:
        await asyncio.gather(_stream_llm(), _speak_sentences())
    finally:
        turn_done.set()
        await gui_task

    window.end_ai_message()

    elapsed = time.perf_counter() - t0
    full_reply = "".join(full_reply_parts)

    log.info("Turn complete in %.2fs | reply=%r", elapsed, full_reply[:100])

    # Persist messages to database
    await db.save_message("user", user_input)
    await db.save_message("assistant", full_reply)

    # Update memory in background (non-blocking)
    if status and config.persona.enabled:
        # Get recent turns for memory context (x2 for user+assistant pairs)
        recent = await db.get_history(config.persona.context_turns * 2)
        if config.persona.update_in_background:
            asyncio.create_task(
                update_memory_background(recent, status, llm)
            )
        else:
            await update_memory_background(recent, status, llm)

    return full_reply


async def run_turn_with_ui(
    user_input: str,
    llm: LLMEngine,
    tts: TTSEngine,
    db: ChatDatabase,
    config: AppConfig,
    ui: TerminalUI,
    status: StatusManager | None = None,
) -> str:
    """
    Run one conversation turn with terminal UI output.

    Streams LLM tokens while concurrently playing TTS for finished sentences.
    Memory is updated via background LLM call after the turn.
    """
    history = await db.get_history()
    history.append({"role": "user", "content": user_input})

    # Build memory context for LLM
    memory_context = _build_memory_context(status)

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    full_reply_parts: list[str] = []

    async def _stream_llm() -> None:
        buffer = ""
        try:
            async for chunk in llm.generate_response(
                history,
                max_retries=config.llm.max_retries,
                memory_context=memory_context,
            ):
                full_reply_parts.append(chunk)
                ui.print_ai_chunk(chunk)
                buffer += chunk

                sentences, buffer = extract_sentences(
                    buffer, config.streaming.min_sentence_chars
                )
                for sentence in sentences:
                    await sentence_queue.put(sentence)
        except Exception as e:
            log.error("LLM stream error: %s", e)
            ui.print_error(f"LLM error: {e}")

        leftover = buffer.strip()
        if leftover:
            await sentence_queue.put(leftover)
        await sentence_queue.put(None)

    async def _speak_sentences() -> None:
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            await tts.speak(sentence)

    t0 = time.perf_counter()

    ui.start_ai_response()
    await asyncio.gather(_stream_llm(), _speak_sentences())
    ui.end_ai_response()

    elapsed = time.perf_counter() - t0
    full_reply = "".join(full_reply_parts)

    log.info("Turn complete in %.2fs | reply=%r", elapsed, full_reply[:100])

    # Persist messages to database
    await db.save_message("user", user_input)
    await db.save_message("assistant", full_reply)

    # Update memory in background (non-blocking)
    if status and config.persona.enabled:
        # Get recent turns for memory context (x2 for user+assistant pairs)
        recent = await db.get_history(config.persona.context_turns * 2)
        if config.persona.update_in_background:
            asyncio.create_task(
                update_memory_background(recent, status, llm)
            )
        else:
            await update_memory_background(recent, status, llm)

    return full_reply


async def run_turn(
    user_input: str,
    llm: LLMEngine,
    tts: TTSEngine,
    db: ChatDatabase,
    config: AppConfig,
    status: StatusManager | None = None,
) -> str:
    """
    Run one conversation turn (plain output, no UI).

    For backwards compatibility or headless use.
    """
    history = await db.get_history()
    history.append({"role": "user", "content": user_input})

    # Build memory context for LLM
    memory_context = _build_memory_context(status)

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    full_reply_parts: list[str] = []

    async def _stream_llm() -> None:
        buffer = ""
        try:
            async for chunk in llm.generate_response(
                history,
                max_retries=config.llm.max_retries,
                memory_context=memory_context,
            ):
                full_reply_parts.append(chunk)
                print(chunk, end="", flush=True)
                buffer += chunk

                sentences, buffer = extract_sentences(
                    buffer, config.streaming.min_sentence_chars
                )
                for sentence in sentences:
                    await sentence_queue.put(sentence)
        except Exception as e:
            log.error("LLM stream error: %s", e)
            print(f"\n[LLM ERROR] {e}")

        leftover = buffer.strip()
        if leftover:
            await sentence_queue.put(leftover)
        await sentence_queue.put(None)

    async def _speak_sentences() -> None:
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            await tts.speak(sentence)

    t0 = time.perf_counter()

    print("\nRin: ", end="", flush=True)
    await asyncio.gather(_stream_llm(), _speak_sentences())
    print()

    elapsed = time.perf_counter() - t0
    full_reply = "".join(full_reply_parts)

    log.info("Turn complete in %.2fs | reply=%r", elapsed, full_reply[:100])

    # Persist messages to database
    await db.save_message("user", user_input)
    await db.save_message("assistant", full_reply)

    # Update memory in background (non-blocking)
    if status and config.persona.enabled:
        # Get recent turns for memory context (x2 for user+assistant pairs)
        recent = await db.get_history(config.persona.context_turns * 2)
        if config.persona.update_in_background:
            asyncio.create_task(
                update_memory_background(recent, status, llm)
            )
        else:
            await update_memory_background(recent, status, llm)

    return full_reply
