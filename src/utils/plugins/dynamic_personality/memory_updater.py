"""
Memory updater — background LLM call to update AI memory.

Analyzes conversation and updates status.txt without
requiring the main LLM to output special tags.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.service.llm.engine import LLMEngine
    from src.utils.plugins.dynamic_personality.status_manager import StatusManager

log = logging.getLogger(__name__)

# Prompt for memory extraction
_MEMORY_PROMPT = """Analyze if this exchange warrants a memory update.

CURRENT MEMORY (turns since last update):
{current_memory}

LATEST EXCHANGE:
User: {user_message}
AI: {ai_response}

SECTION RULES - what each section is based on:
- notes: ONLY from USER's message - facts user explicitly stated (name, hobbies, preferences)
  → AI suggestions/questions are NOT user facts. "AI asks about X" ≠ "user wants X"
  → ALWAYS preserve existing notes when updating - copy them + add new fact
- impression: ONLY from USER's message - user's current mood/behavior
- mood: ONLY from AI's response - AI's emotional state expressed in its reply
- relationship: from BOTH messages - how they interact together

FREQUENCY - avoid unnecessary updates:
- notes: only when user reveals new permanent facts (most messages don't!)
- mood/impression: when deemed necessary, but avoid frequent changes
- relationship: only if clear shift in interaction style

When updating sections, make sure to not overwrite existing important info.
Try to summarize when get too long but keep previous details that may be relevants.


Return {{}} if no updates needed (most exchanges need none).
If updating notes, preserve existing: {{"notes": "existing fact. existing fact. NEW fact."}}
"""

# Pattern to extract JSON from response
_JSON_PATTERN = re.compile(r'\{[^{}]*\}', re.DOTALL)


def _format_memory_with_turns(status: "StatusManager") -> str:
    """Format memory with turn counts for the analyzer."""
    turn_info = status.get_turn_info()
    lines = []

    for key in ["notes", "mood", "impression", "relationship"]:
        content = status.get_section(key)
        turns = turn_info.get(key, 0)
        if content:
            lines.append(f"• {key} ({turns} turns ago): {content}")
        else:
            lines.append(f"• {key} ({turns} turns ago): (empty)")

    return "\n".join(lines) if lines else "(empty)"


async def update_memory_from_conversation(
    user_message: str,
    ai_response: str,
    status: StatusManager,
    llm: LLMEngine,
) -> bool:
    """
    Analyze conversation and update memory via separate LLM call.

    Returns True if memory was updated.
    """
    # Increment turn counter first
    status.increment_turn()

    # Build current memory string with turn info
    current_memory = _format_memory_with_turns(status)

    # Build the analysis prompt
    prompt = _MEMORY_PROMPT.format(
        current_memory=current_memory,
        user_message=user_message,
        ai_response=ai_response,
    )

    # Make LLM call (non-streaming, just get full response)
    try:
        response_parts = []
        async for chunk in llm.generate_response(
            messages=[{"role": "user", "content": prompt}],
            max_retries=1,
            memory_context=None,  # Don't inject memory into this call
        ):
            response_parts.append(chunk)

        response = "".join(response_parts).strip()
        log.debug("Memory analyzer response: %s", response[:200])

    except Exception as e:
        log.error("Memory update LLM call failed: %s", e)
        return False

    # Parse JSON from response
    try:
        # Try to find JSON object in response
        json_match = _JSON_PATTERN.search(response)
        if not json_match:
            log.debug("No JSON found in memory analyzer response")
            return False

        updates = json.loads(json_match.group())

        if not updates:
            log.debug("Memory analyzer: no updates needed")
            return False

    except json.JSONDecodeError as e:
        log.warning("Failed to parse memory update JSON: %s", e)
        return False

    # Apply updates
    changes_made = False
    for section, content in updates.items():
        section = section.lower()
        if content and status.set_section(section, content):
            log.info("Memory [%s] updated: %s", section, content[:50])
            changes_made = True

    if changes_made:
        status.save()

    return changes_made


async def update_memory_background(
    user_message: str,
    ai_response: str,
    status: StatusManager,
    llm: LLMEngine,
) -> None:
    """
    Fire-and-forget memory update.

    Runs update_memory_from_conversation but catches all errors
    so it never affects the main conversation flow.
    """
    try:
        changed = await update_memory_from_conversation(
            user_message, ai_response, status, llm
        )
        if changed:
            log.info("Background memory update completed")
    except Exception as e:
        log.error("Background memory update failed: %s", e)
