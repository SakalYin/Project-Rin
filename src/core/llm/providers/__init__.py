"""LLM provider registry — lazy-loaded to avoid importing unused dependencies."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.llm.base import LLMProvider

# Maps provider name → (module_path, class_name).
# Add new providers here.
_REGISTRY: dict[str, tuple[str, str]] = {
    "openai_compat": (
        "src.core.llm.providers.openai_compat",
        "OpenAICompatProvider",
    ),
}


def get_llm_provider(name: str) -> type[LLMProvider]:
    """Return the LLMProvider class for *name*, importing lazily."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown LLM provider {name!r}. Available: {available}"
        )
    module_path, class_name = _REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
