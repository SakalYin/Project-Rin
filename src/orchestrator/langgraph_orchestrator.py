"""
LangGraph-based LLM pipeline orchestrator.

This module handles ONLY the LLM streaming pipeline:
  - Fetch chat history
  - Stream LLM response with sentence extraction

Other concerns (TTS playback, DB persistence) are handled by the caller
in conversation.py, which consumes sentences from the async generator.

Graph structure:
    START -> stream_llm -> END
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from src.service.llm.engine import LLMEngine
from src.service.tts.engine import extract_sentences
from src.db.db_manager import ChatDatabase
from src.core.config import AppConfig

log = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────

class LLMPipelineState(TypedDict):
    """Typed state for the LLM pipeline."""
    user_input: str
    history: list[dict[str, str]]
    full_reply: str
    sentences: list[str]
    elapsed: float
    error: str | None


# ── Resource helpers ─────────────────────────────────────────────────

class PipelineResources(TypedDict):
    """Resources passed via LangGraph's config dict."""
    llm: LLMEngine
    db: ChatDatabase
    app_config: AppConfig


def _get_resources(config: RunnableConfig) -> PipelineResources:
    """Extract pipeline resources from LangGraph config."""
    return config["configurable"]


# ── Node functions ───────────────────────────────────────────────────

async def fetch_history(state: LLMPipelineState, config: RunnableConfig) -> dict:
    """Fetch chat history from the database and append user message."""
    res = _get_resources(config)
    db: ChatDatabase = res["db"]

    history = await db.get_history()
    history.append({"role": "user", "content": state["user_input"]})

    return {"history": history}


async def stream_llm(state: LLMPipelineState, config: RunnableConfig) -> dict:
    """
    Stream LLM response and extract sentences.

    Collects the full reply and a list of sentences for the caller to use.
    """
    res = _get_resources(config)
    llm: LLMEngine = res["llm"]
    app_config: AppConfig = res["app_config"]

    history = state["history"]
    full_reply_parts: list[str] = []
    sentences: list[str] = []
    buffer = ""

    t0 = time.perf_counter()

    print("\nRin: ", end="", flush=True)
    try:
        async for chunk in llm.generate_response(
            history, max_retries=app_config.llm.max_retries
        ):
            full_reply_parts.append(chunk)
            print(chunk, end="", flush=True)
            buffer += chunk

            extracted, buffer = extract_sentences(
                buffer, app_config.streaming.min_sentence_chars
            )
            sentences.extend(extracted)
    except Exception as e:
        log.error("LLM stream error: %s", e)
        print(f"\n[LLM ERROR] {e}")

    # Flush remaining buffer
    leftover = buffer.strip()
    if leftover:
        sentences.append(leftover)

    print()

    elapsed = time.perf_counter() - t0
    full_reply = "".join(full_reply_parts)
    log.info("LLM complete in %.2fs | reply=%r", elapsed, full_reply)

    return {
        "full_reply": full_reply,
        "sentences": sentences,
        "elapsed": elapsed,
        "error": None if full_reply else "Empty LLM response",
    }


# ── Graph construction ───────────────────────────────────────────────

def build_llm_pipeline() -> Any:
    """
    Build and compile the LLM pipeline graph.

    Returns a compiled graph for LLM streaming only.
    """
    graph = StateGraph(LLMPipelineState)

    graph.add_node("fetch_history", fetch_history)
    graph.add_node("stream_llm", stream_llm)

    graph.add_edge(START, "fetch_history")
    graph.add_edge("fetch_history", "stream_llm")
    graph.add_edge("stream_llm", END)

    return graph.compile()


# Compile once at module level
_llm_pipeline = build_llm_pipeline()


# ── Public API ───────────────────────────────────────────────────────

async def run_llm_pipeline(
    user_input: str,
    llm: LLMEngine,
    db: ChatDatabase,
    config: AppConfig,
) -> tuple[str, list[str], float]:
    """
    Run the LLM pipeline and return the result.

    Returns:
        (full_reply, sentences, elapsed_time)
    """
    initial_state: LLMPipelineState = {
        "user_input": user_input,
        "history": [],
        "full_reply": "",
        "sentences": [],
        "elapsed": 0.0,
        "error": None,
    }

    result = await _llm_pipeline.ainvoke(
        initial_state,
        config={
            "configurable": {
                "llm": llm,
                "db": db,
                "app_config": config,
            }
        },
    )

    return result["full_reply"], result["sentences"], result["elapsed"]
