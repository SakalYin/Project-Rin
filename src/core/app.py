"""
Main async loop — ties engines, orchestrator and chat persistence together.

Architecture:
  0. LLM engine starts (provider may launch a subprocess, etc.)
  1. TTS + STT engines initialize (load models)
  2. Memory system loads (persistent AI context)
  3. STT begins active background listening (VAD loop)
  4. User speaks OR types in GUI window — whichever comes first
  5. Orchestrator: LLM streams → sentences queued → TTS plays
  6. Memory tags processed, full reply saved to database
  7. Loop until window closed or Ctrl+C
  8. Engines shut down gracefully

The chat UI runs in a separate GUI window.
Logging output goes to the terminal and log file.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from src.core.config import AppConfig, load_config
from src.core.chat_window import AsyncChatWindow, SessionDialog
from src.service.llm.engine import LLMEngine
from src.service.tts.engine import TTSEngine
from src.service.stt.engine import STTEngine
from src.db.db_manager import ChatDatabase
from src.orchestrator.conversation import run_turn_with_window
from src.utils.plugins.dynamic_personality.status_manager import StatusManager

log = logging.getLogger(__name__)


# ── Chat loop ────────────────────────────────────────────────────────────

async def _chat_loop(config: AppConfig, llm: LLMEngine) -> None:
    """Inner chat loop — engines already running at this point."""
    # Initialize database first for session selection
    db = ChatDatabase(config)
    await db.initialize()
    log.info("Database initialized")

    # Initialize persona system (if enabled)
    status = None
    if config.persona.enabled:
        status = StatusManager(filepath=config.persona.state_file)
        log.info("Persona system loaded from: %s", config.persona.state_file)
    else:
        log.info("Persona system disabled")

    # Show session selection dialog
    sessions = await db.get_sessions()
    dialog = SessionDialog(sessions)
    action, session_id = dialog.run()

    continuing = False
    if action == "continue" and session_id is not None:
        await db.load_session(session_id)
        session_name = await db.get_session_name()
        log.info("Continuing session: %s", session_name)
        continuing = True
    else:
        session_id = await db.create_session()
        session_name = await db.get_session_name()
        log.info("Created new session: %s", session_name)

    # Initialize other services
    tts = TTSEngine(config)
    await tts.initialize()
    log.info("TTS engine initialized")

    stt: STTEngine | None = None
    if config.stt.enabled:
        stt = STTEngine(config)
        log.info("Loading STT model...")
        await stt.initialize()
        await stt.start_listening()
        log.info("STT engine initialized and listening")

    # Create chat window
    window = AsyncChatWindow(ai_name="Rin")
    await window.start()

    if stt:
        window.set_status(f"🎤 Session: {session_name}")
    else:
        window.set_status(f"Session: {session_name}")

    # Load previous messages if continuing a session
    if continuing:
        history = await db.get_history()
        for msg in history:
            if msg["role"] == "user":
                window.append_user_message(msg["content"])
            else:
                window.start_ai_message()
                window.append_ai_chunk(msg["content"])
                window.end_ai_message()
        if history:
            window.show_info("— Previous messages loaded —")

    window.show_info("Ready to chat!")

    try:
        while not window.is_closed:
            # Get user input (from GUI or voice)
            user_input = await window.get_input(stt)

            if not user_input:
                if window.is_closed:
                    break
                continue

            if user_input.lower() in ("quit", "exit"):
                break

            # Show user message in window
            window.append_user_message(user_input)

            # Run conversation turn with memory system
            await run_turn_with_window(
                user_input, llm, tts, db, config, window, status
            )

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        window.close()
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
    # Log to both terminal and file
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

    print("Starting Project Rin...")
    print("Chat window will open. Logs will appear here.")
    print("-" * 50)

    asyncio.run(main_loop())
