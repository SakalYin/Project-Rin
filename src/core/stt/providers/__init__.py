"""STT provider registry — lazy-loaded to avoid importing unused dependencies."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.stt.base import STTProvider

_REGISTRY: dict[str, tuple[str, str]] = {
    "faster_whisper": (
        "src.core.stt.providers.faster_whisper",
        "FasterWhisperProvider",
    ),
}


def get_stt_provider(name: str) -> type[STTProvider]:
    """Return the STTProvider class for *name*, importing lazily."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown STT provider {name!r}. Available: {available}"
        )
    module_path, class_name = _REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
