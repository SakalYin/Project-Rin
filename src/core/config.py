"""
YAML configuration loader for Project Rin.

Each subsystem (LLM, TTS, STT) has:
  - A ``provider`` key selecting the backend implementation
  - Engine-level keys consumed by the engine (typed fields below)
  - Everything else flows into ``provider_config`` as a plain dict

This means adding a new provider with novel config keys requires
zero changes to this file.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Subsystem configs ────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """LLM engine + provider configuration."""
    provider: str = "openai_compat"
    max_retries: int = 3
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSConfig:
    """TTS engine + provider configuration."""
    provider: str = "kokoro"
    sample_rate: int = 24000              # used by engine for playback
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class STTConfig:
    """STT engine + provider configuration."""
    provider: str = "faster_whisper"
    enabled: bool = True
    sample_rate: int = 16000              # recording sample rate
    silence_duration: float = 1.5         # seconds of silence to end utterance
    max_duration: float = 30.0            # max buffer before forced processing
    vad_threshold: float = 0.65           # Silero VAD sensitivity (0-1)
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseConfig:
    path: str = "data/chat_history.db"
    history_limit: int = 20


@dataclass
class StreamingConfig:
    min_sentence_chars: int = 12


@dataclass
class PersonaConfig:
    """Dynamic AI personality configuration."""
    enabled: bool = True
    state_file: str = "src/utils/plugins/dynamic_personality/status.txt"
    update_in_background: bool = True
    context_turns: int = 3  # Number of recent turns to analyze for memory updates


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)


# ── Helpers ──────────────────────────────────────────────────────────────

# Engine-level keys for each subsystem — everything else is provider config.
_LLM_ENGINE_KEYS = {"provider", "max_retries"}
_TTS_ENGINE_KEYS = {"provider", "sample_rate"}
_STT_ENGINE_KEYS = {
    "provider", "enabled", "sample_rate",
    "silence_duration", "max_duration",
    "vad_threshold",
}


def _resolve_paths(d: dict[str, Any], keys: set[str] | None = None) -> None:
    """Resolve values whose keys look like paths against PROJECT_ROOT."""
    for key, val in list(d.items()):
        if isinstance(val, str) and (
            key.endswith("_path") or key == "executable"
        ):
            d[key] = str(PROJECT_ROOT / val)
        elif isinstance(val, dict):
            _resolve_paths(val)


def _split_section(
    raw: dict[str, Any], engine_keys: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a YAML section into engine kwargs and provider config."""
    engine = {}
    provider = {}
    for k, v in raw.items():
        if k in engine_keys:
            engine[k] = v
        else:
            provider[k] = v
    return engine, provider


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    config_path = path or PROJECT_ROOT / "config.yaml"
    cfg = AppConfig()

    if not config_path.exists():
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # ── LLM ──────────────────────────────────────────────────────
    if "llm" in raw:
        engine, provider = _split_section(raw["llm"], _LLM_ENGINE_KEYS)
        _resolve_paths(provider)
        cfg.llm = LLMConfig(**engine, provider_config=provider)

    # ── TTS ──────────────────────────────────────────────────────
    if "tts" in raw:
        engine, provider = _split_section(raw["tts"], _TTS_ENGINE_KEYS)
        _resolve_paths(provider)
        cfg.tts = TTSConfig(**engine, provider_config=provider)

    # ── STT ──────────────────────────────────────────────────────
    if "stt" in raw:
        engine, provider = _split_section(raw["stt"], _STT_ENGINE_KEYS)
        _resolve_paths(provider)
        cfg.stt = STTConfig(**engine, provider_config=provider)

    # ── Database & Streaming (unchanged) ─────────────────────────
    if "database" in raw:
        cfg.database = DatabaseConfig(**raw["database"])
    if "streaming" in raw:
        cfg.streaming = StreamingConfig(**raw["streaming"])
    if "persona" in raw:
        cfg.persona = PersonaConfig(**raw["persona"])

    # Resolve paths
    cfg.database.path = str(PROJECT_ROOT / cfg.database.path)
    cfg.persona.state_file = str(PROJECT_ROOT / cfg.persona.state_file)

    return cfg
