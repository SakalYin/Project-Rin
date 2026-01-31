"""
Main async loop — ties LLM streaming, TTS playback, and chat persistence together.

Architecture:
  0. (Optional) Start llama.cpp server as a subprocess
  1. User types input
  2. LLM streams tokens → accumulated into a sentence buffer
  3. Complete sentences are queued for TTS immediately (concurrent with LLM)
  4. TTS synthesizes + plays each sentence while LLM keeps generating
  5. Full reply is printed to console and saved to the database
  6. Loop until "quit" or Ctrl+C
  7. Server subprocess is terminated on exit
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from src.core.config import AppConfig, load_config
from src.core.llm.client import LLMClient
from src.core.llm.server import LlamaServer
from src.core.tts.engine import TTSEngine, extract_sentences
from src.db.db_manager import ChatDatabase

log = logging.getLogger(__name__)


async def generate_and_speak(
    user_input: str,
    llm: LLMClient,
    tts: TTSEngine,
    db: ChatDatabase,
    config: AppConfig,
) -> str:
    """
    Stream a reply from the LLM while concurrently sending finished
    sentences to TTS for immediate playback.

    Returns the full assistant reply.
    """
    # Build message list from DB history
    history = await db.get_history()
    history.append({"role": "user", "content": user_input})

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    full_reply_parts: list[str] = []

    # ── Producer: stream LLM tokens, split into sentences ────────────
    async def _stream_llm() -> None:
        buffer = ""
        try:
            async for chunk in llm.generate_response(
                history, max_retries=config.llm.max_retries
            ):
                full_reply_parts.append(chunk)
                # Print each token as it arrives
                print(chunk, end="", flush=True)
                buffer += chunk

                # Try to extract complete sentences
                sentences, buffer = extract_sentences(
                    buffer, config.streaming.min_sentence_chars
                )
                for sentence in sentences:
                    await sentence_queue.put(sentence)
        except Exception as e:
            log.error("LLM stream error: %s", e)
            print(f"\n[LLM ERROR] {e}")

        # Flush remaining buffer
        leftover = buffer.strip()
        if leftover:
            await sentence_queue.put(leftover)

        # Signal the consumer that we're done
        await sentence_queue.put(None)

    # ── Consumer: synthesise + play each sentence ────────────────────
    async def _speak_sentences() -> None:
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            await tts.speak(sentence)

    t0 = time.perf_counter()

    # Run producer and consumer concurrently
    print("\nRin: ", end="", flush=True)
    await asyncio.gather(_stream_llm(), _speak_sentences())
    print()  # newline after streamed output

    elapsed = time.perf_counter() - t0
    full_reply = "".join(full_reply_parts)
    log.info("Turn complete in %.2fs | reply=%r", elapsed, full_reply)

    # Persist both sides of the conversation
    await db.save_message("user", user_input)
    await db.save_message("assistant", full_reply)

    return full_reply


async def _chat_loop(config: AppConfig) -> None:
    """Inner chat loop (server already running at this point)."""
    llm = LLMClient(config)
    tts = TTSEngine(config)
    db = ChatDatabase(config)
    await db.initialize()

    print("=" * 60)
    print("  Project Rin — Local AI Voice Agent")
    print("  Type your message and press Enter.")
    print('  Type "quit" or press Ctrl+C to exit.')
    print("=" * 60)
    print()

    try:
        while True:
            try:
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: input("You: ")
                )
            except EOFError:
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("\nBye bye~")
                break

            await generate_and_speak(user_input, llm, tts, db, config)

    except KeyboardInterrupt:
        print("\n\nInterrupted — shutting down.")
    finally:
        await db.close()
        log.info("Database closed.")


async def main_loop() -> None:
    """Top-level entry: optionally start the server, then run the chat loop."""
    config = load_config()

    if config.server.enabled:
        server = LlamaServer(config)
        try:
            async with server:
                await _chat_loop(config)
        except (FileNotFoundError, ValueError, TimeoutError) as e:
            log.error("Server startup failed: %s", e)
            print(f"\n[SERVER ERROR] {e}")
            return
    else:
        # Server managed externally — go straight to the chat loop
        await _chat_loop(config)


def run() -> None:
    """Entry point — configure logging and start the async loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("rin_debug.log", encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    asyncio.run(main_loop())
