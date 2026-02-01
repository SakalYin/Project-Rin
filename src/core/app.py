"""
Main async loop — ties LLM, TTS, STT engines and chat persistence together.

Architecture:
  0. LLM engine starts (provider may launch a subprocess, etc.)
  1. TTS + STT engines initialize (load models)
  2. STT begins active background listening (VAD loop)
  3. User speaks OR types — whichever comes first
  4. LLM streams tokens → sentences queued → TTS plays concurrently
  5. Full reply is saved to the database
  6. Loop until "quit" or Ctrl+C
  7. Engines shut down gracefully
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from src.core.config import AppConfig, load_config
from src.core.llm.engine import LLMEngine
from src.core.tts.engine import TTSEngine, extract_sentences
from src.core.stt.engine import STTEngine
from src.db.db_manager import ChatDatabase

log = logging.getLogger(__name__)


# ── Input handling ───────────────────────────────────────────────────────

async def _get_user_input(stt: STTEngine | None) -> str:
    """
    Return user input from either keyboard or voice, whichever arrives first.

    If STT is disabled (None), falls back to keyboard-only.
    """
    loop = asyncio.get_running_loop()
    text_task = asyncio.ensure_future(
        loop.run_in_executor(None, lambda: input("You: "))
    )

    if stt is None:
        return (await text_task).strip()

    voice_task = asyncio.ensure_future(stt.input_queue.get())

    done, pending = await asyncio.wait(
        [text_task, voice_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, EOFError):
            pass

    result = done.pop().result()
    return result.strip() if isinstance(result, str) else ""


# ── Generate + speak ─────────────────────────────────────────────────────

async def generate_and_speak(
    user_input: str,
    llm: LLMEngine,
    tts: TTSEngine,
    db: ChatDatabase,
    config: AppConfig,
) -> str:
    """
    Stream a reply from the LLM while concurrently sending finished
    sentences to TTS for immediate playback.

    Returns the full assistant reply.
    """
    history = await db.get_history()
    history.append({"role": "user", "content": user_input})

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    full_reply_parts: list[str] = []

    async def _stream_llm() -> None:
        buffer = ""
        try:
            async for chunk in llm.generate_response(
                history, max_retries=config.llm.max_retries
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
    log.info("Turn complete in %.2fs | reply=%r", elapsed, full_reply)

    await db.save_message("user", user_input)
    await db.save_message("assistant", full_reply)

    return full_reply


# ── Chat loop ────────────────────────────────────────────────────────────

async def _chat_loop(config: AppConfig, llm: LLMEngine) -> None:
    """Inner chat loop — engines already running at this point."""
    tts = TTSEngine(config)
    await tts.initialize()

    db = ChatDatabase(config)
    await db.initialize()

    stt: STTEngine | None = None
    if config.stt.enabled:
        stt = STTEngine(config)
        print("Loading STT model…")
        await stt.initialize()
        await stt.start_listening()

    print("=" * 60)
    print("  Project Rin — Local AI Voice Agent")
    if stt:
        print("  Speak or type your message. Always listening.")
    else:
        print("  Type your message and press Enter.")
    print('  Type "quit" or press Ctrl+C to exit.')
    print("=" * 60)
    print()

    try:
        while True:
            try:
                user_input = await _get_user_input(stt)
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("\nBye bye~")
                break

            # Show what STT heard (so user can confirm)
            if stt and user_input:
                print(f"You: {user_input}")

            await generate_and_speak(user_input, llm, tts, db, config)

    except KeyboardInterrupt:
        print("\n\nInterrupted — shutting down.")
    finally:
        if stt:
            await stt.stop_listening()
        await tts.shutdown()
        await db.close()
        log.info("All engines shut down.")


# ── Entry points ─────────────────────────────────────────────────────────

async def main_loop() -> None:
    """Top-level entry: start engines, then run the chat loop."""
    config = load_config()

    llm = LLMEngine(config)
    try:
        async with llm:
            await _chat_loop(config, llm)
    except (FileNotFoundError, ValueError, TimeoutError) as e:
        log.error("LLM engine startup failed: %s", e)
        print(f"\n[ENGINE ERROR] {e}")


def run() -> None:
    """Configure logging and start the async loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("rin_debug.log", encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    asyncio.run(main_loop())
