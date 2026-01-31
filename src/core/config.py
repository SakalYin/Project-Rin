"""
YAML configuration loader for Project Rin.
Reads config.yaml from the project root and exposes values as typed dataclasses.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ServerConfig:
    """llama.cpp server subprocess settings."""
    enabled: bool = True
    executable: str = "llama.ccp/llama-server.exe"
    model_path: str = ""               # REQUIRED — path to .gguf file
    context_size: int = 2048            # -c  (total context window)
    n_predict: int = 512                # --n-predict  (max completion tokens)
    gpu_layers: int = -1                # -ngl  (-1 = offload everything)
    port: int = 8080
    host: str = "127.0.0.1"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    """OpenAI-compatible API settings (pointed at the llama.cpp server)."""
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    model: str = "local-model"
    temperature: float = 1.1
    max_tokens: Optional[int] = None    # None = let the server's n_predict decide
    timeout: float = 30.0
    max_retries: int = 3


@dataclass
class TTSConfig:
    model_path: str = "models/kokoro-v1.0.onnx"
    voices_path: str = "models/voices-v1.0.bin"
    voice: str = "af_heart"
    speed: float = 1.1
    lang: str = "en-us"
    sample_rate: int = 24000


@dataclass
class DatabaseConfig:
    path: str = "data/chat_history.db"
    history_limit: int = 20


@dataclass
class StreamingConfig:
    min_sentence_chars: int = 12


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    config_path = path or PROJECT_ROOT / "config.yaml"
    cfg = AppConfig()

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "server" in raw:
            cfg.server = ServerConfig(**raw["server"])
        if "llm" in raw:
            cfg.llm = LLMConfig(**raw["llm"])
        if "tts" in raw:
            cfg.tts = TTSConfig(**raw["tts"])
        if "database" in raw:
            cfg.database = DatabaseConfig(**raw["database"])
        if "streaming" in raw:
            cfg.streaming = StreamingConfig(**raw["streaming"])

    # Resolve relative paths against project root
    cfg.server.executable = str(PROJECT_ROOT / cfg.server.executable)
    if cfg.server.model_path:
        cfg.server.model_path = str(PROJECT_ROOT / cfg.server.model_path)
    cfg.tts.model_path = str(PROJECT_ROOT / cfg.tts.model_path)
    cfg.tts.voices_path = str(PROJECT_ROOT / cfg.tts.voices_path)
    cfg.database.path = str(PROJECT_ROOT / cfg.database.path)

    # Auto-derive llm.base_url from server config if server is enabled
    if cfg.server.enabled:
        cfg.llm.base_url = f"http://{cfg.server.host}:{cfg.server.port}/v1"

    return cfg
