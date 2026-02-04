"""TTS provider registry — lazy-loaded to avoid importing unused dependencies."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.service.tts.base import TTSProvider

_REGISTRY: dict[str, tuple[str, str]] = {
    "kokoro": (
        "src.service.tts.providers.kokoro",
        "KokoroProvider",
    ),
}


def get_tts_provider(name: str) -> type[TTSProvider]:
    """Return the TTSProvider class for *name*, importing lazily."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown TTS provider {name!r}. Available: {available}"
        )
    module_path, class_name = _REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
