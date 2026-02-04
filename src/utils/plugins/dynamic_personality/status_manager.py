"""
Interaction status manager — persistent cross-session memory for AI personality.

Memory is updated via a separate background LLM call (see memory_updater.py),
not by parsing tags from the main conversation.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Sentence boundary pattern — handles abbreviations better
# Splits on .!? followed by space(s) and uppercase letter
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ── Section Limits ───────────────────────────────────────────────────────

@dataclass
class SectionLimit:
    """Configurable limits for a memory section."""
    display_name: str
    max_sentences: int = 5
    priority: int = 1  # higher = more important (for future use)

    def __post_init__(self):
        self.max_sentences = max(1, self.max_sentences)


# Default limits — can be overridden at runtime
DEFAULT_LIMITS: dict[str, SectionLimit] = {
    "mood": SectionLimit(
        display_name="Current mood",
        max_sentences=2,
        priority=2,
    ),
    "impression": SectionLimit(
        display_name="User impression",
        max_sentences=2,
        priority=2,
    ),
    "relationship": SectionLimit(
        display_name="Relationship",
        max_sentences=3,
        priority=3,
    ),
    "notes": SectionLimit(
        display_name="Notes about user",
        max_sentences=10,
        priority=1,
    ),
}


# ── Status Manager ───────────────────────────────────────────────────────

class StatusManager:
    """
    Manages persistent AI status/memory across sessions.

    Sections:
      - mood: Current AI emotional state
      - impression: What AI thinks of user right now
      - relationship: How AI and user interact
      - notes: Facts about user to remember

    Memory is updated via background LLM call (memory_updater.py),
    and injected into the main LLM's system prompt for context.
    """

    def __init__(
        self,
        filepath: Path | str | None = None,
        limits: dict[str, SectionLimit] | None = None,
    ):
        if filepath is None:
            # Default to same folder as this module
            filepath = Path(__file__).resolve().parent / "status.txt"

        self.filepath = Path(filepath)
        self._limits = limits if limits is not None else DEFAULT_LIMITS.copy()
        self._sections: dict[str, str] = {key: "" for key in self._limits}
        # Turn tracking: how many turns since each section was last updated
        self._turn_count: int = 0
        self._turns_since_update: dict[str, int] = {key: 0 for key in self._limits}
        self._load()

    # ── Limit Management ─────────────────────────────────────────────────

    def get_limit(self, key: str) -> SectionLimit | None:
        """Get the limit config for a section."""
        return self._limits.get(key)

    def set_limit(self, key: str, limit: SectionLimit) -> None:
        """Update limits for a section (creates section if new)."""
        self._limits[key] = limit
        if key not in self._sections:
            self._sections[key] = ""
        else:
            self._truncate(key)

    def update_limit(
        self,
        key: str,
        max_sentences: int | None = None,
        priority: int | None = None,
    ) -> bool:
        """Update specific limit values for a section. Returns False if section doesn't exist."""
        if key not in self._limits:
            return False

        limit = self._limits[key]
        if max_sentences is not None:
            limit.max_sentences = max(1, max_sentences)
        if priority is not None:
            limit.priority = priority

        self._truncate(key)
        return True

    def add_section(
        self,
        key: str,
        display_name: str,
        max_sentences: int = 5,
        priority: int = 1,
    ) -> None:
        """Add a new section dynamically."""
        self._limits[key] = SectionLimit(
            display_name=display_name,
            max_sentences=max_sentences,
            priority=priority,
        )
        self._sections[key] = ""

    def remove_section(self, key: str) -> bool:
        """Remove a section. Returns False if it didn't exist."""
        if key not in self._limits:
            return False
        del self._limits[key]
        self._sections.pop(key, None)
        return True

    # ── Truncation ───────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, handling edge cases."""
        text = text.strip()
        if not text:
            return []

        # Split on sentence boundaries
        parts = _SENTENCE_SPLIT.split(text)

        # Clean up and filter empty
        sentences = []
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                # Ensure ends with punctuation
                if cleaned[-1] not in ".!?":
                    cleaned += "."
                sentences.append(cleaned)

        return sentences

    def _truncate(self, key: str) -> None:
        """Enforce sentence limit by keeping newest content."""
        if key not in self._limits:
            return

        limit = self._limits[key]
        text = self._sections[key].strip()
        if not text:
            return

        sentences = self._split_sentences(text)
        if len(sentences) > limit.max_sentences:
            # Keep the newest sentences
            sentences = sentences[-limit.max_sentences:]
            self._sections[key] = " ".join(sentences)

    # ── File I/O ─────────────────────────────────────────────────────────

    # Map old-style headers to short keys
    _HEADER_MAP = {
        "ai mood and dynamic": "mood",
        "user impression": "impression",
        "relationship dynamic": "relationship",
        "notes on user": "notes",
    }

    def _parse_header(self, line: str) -> str | None:
        """Parse a header line and return the section key, or None if not a header."""
        line = line.strip()

        # New format: [section]
        if line.startswith("[") and line.endswith("]"):
            key = line[1:-1].strip().lower()
            return key if key in self._limits else None

        # Old format: Section Name:
        if line.endswith(":"):
            header = line[:-1].strip().lower()
            # Direct match
            if header in self._limits:
                return header
            # Map old names to new keys
            return self._HEADER_MAP.get(header)

        return None

    def _load(self) -> None:
        """Load sections from file if it exists."""
        if not self.filepath.is_file():
            log.info("Memory file not found, starting fresh: %s", self.filepath)
            return

        current_key = None
        lines: list[str] = []

        with self.filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")

                # Check if this is a header line
                parsed_key = self._parse_header(line)
                if parsed_key is not None:
                    # Save previous section
                    if current_key and lines:
                        self._sections[current_key] = " ".join(lines).strip()
                        self._truncate(current_key)
                    # Start new section
                    current_key = parsed_key
                    lines = []
                elif current_key and line.strip():
                    lines.append(line.strip())

        # Save last section
        if current_key and lines:
            self._sections[current_key] = " ".join(lines).strip()
            self._truncate(current_key)

        log.info("Memory loaded from: %s", self.filepath)

    def save(self) -> None:
        """Write current sections to file."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with self.filepath.open("w", encoding="utf-8") as f:
            for key, content in self._sections.items():
                if content.strip():
                    f.write(f"[{key}]\n")
                    f.write(content.strip())
                    f.write("\n\n")
        log.info("Memory saved to: %s", self.filepath)

    # ── Prompt Formatting ────────────────────────────────────────────────

    def format_for_prompt(self) -> str:
        """
        Format memory block for injection into system prompt.

        Returns empty string if no memory exists.
        """
        if not any(self._sections.values()):
            return ""

        lines = ["[Your current memory — use this to stay consistent:]"]
        for key, content in self._sections.items():
            if content.strip():
                display_name = self._limits[key].display_name
                lines.append(f"• {display_name}: {content}")

        return "\n".join(lines)

    # ── Section Access ───────────────────────────────────────────────────

    def get_section(self, key: str) -> str:
        """Get current content of one section."""
        return self._sections.get(key, "").strip()

    def set_section(self, key: str, content: str) -> bool:
        """Directly set a section's content. Returns False if section doesn't exist."""
        if key not in self._limits:
            return False
        self._sections[key] = content.strip()
        self._truncate(key)
        # Reset turn counter for this section
        self._turns_since_update[key] = 0
        return True

    def get_stats(self) -> dict[str, dict]:
        """Get current stats for all sections (useful for debugging/UI)."""
        stats = {}
        for key, content in self._sections.items():
            limit = self._limits[key]
            sentences = self._split_sentences(content)
            stats[key] = {
                "display_name": limit.display_name,
                "sentence_count": len(sentences),
                "sentence_limit": limit.max_sentences,
                "is_empty": not content.strip(),
                "turns_since_update": self._turns_since_update.get(key, 0),
            }
        return stats

    # ── Turn Tracking ─────────────────────────────────────────────────────

    def increment_turn(self) -> None:
        """Call after each conversation turn to track staleness."""
        self._turn_count += 1
        for key in self._turns_since_update:
            self._turns_since_update[key] += 1

    def get_turn_count(self) -> int:
        """Get total turns since session started."""
        return self._turn_count

    def get_turns_since_update(self, key: str) -> int:
        """Get turns since a specific section was last updated."""
        return self._turns_since_update.get(key, 0)

    def get_turn_info(self) -> dict[str, int]:
        """Get turn info for all sections (for memory updater)."""
        return self._turns_since_update.copy()

    def clear_all(self) -> None:
        """Clear all memory sections."""
        for key in self._sections:
            self._sections[key] = ""
        self.save()
        log.info("Memory: cleared all sections")

    def clear_section(self, key: str) -> bool:
        """Clear a specific section. Returns False if it doesn't exist."""
        if key not in self._sections:
            return False
        self._sections[key] = ""
        self.save()
        log.info("Memory: cleared section [%s]", key)
        return True
