"""
Screenshot capture module — capture screen with monitor selection support.

Saves screenshots to temp/screenshots/ directory.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

if TYPE_CHECKING:
    from mss.screenshot import ScreenShot
    from src.core.config import ScreenshotConfig

log = logging.getLogger(__name__)

# Default screenshot directory (used when no config provided)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "temp" / "screenshots"


@dataclass
class MonitorInfo:
    """Information about a monitor."""
    index: int  # 0 = all monitors combined, 1+ = individual monitors
    left: int
    top: int
    width: int
    height: int

    @property
    def is_combined(self) -> bool:
        """True if this represents all monitors combined."""
        return self.index == 0

    def __str__(self) -> str:
        if self.is_combined:
            return f"All monitors ({self.width}x{self.height})"
        return f"Monitor {self.index} ({self.width}x{self.height})"


@dataclass
class ScreenshotResult:
    """Result of a screenshot capture."""
    path: Path
    width: int
    height: int
    monitor_index: int
    timestamp: float
    _png_bytes: bytes = field(default=b"", repr=False)

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def base64(self) -> str:
        """Return base64-encoded PNG for VLM APIs."""
        return base64.b64encode(self._png_bytes).decode("utf-8")

    @property
    def png_bytes(self) -> bytes:
        """Return raw PNG bytes."""
        return self._png_bytes

    def to_vlm_content(self, detail: str = "auto") -> dict:
        """
        Format for OpenAI-compatible VLM message content.

        Args:
            detail: "auto", "low", or "high" for image detail level.

        Returns:
            Dict ready to use in message content array.
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{self.base64}",
                "detail": detail,
            },
        }


def _ensure_mss() -> None:
    """Raise ImportError if mss is not available."""
    if not MSS_AVAILABLE:
        raise ImportError(
            "mss package is required for screenshots. "
            "Install with: pip install mss"
        )


def get_monitors() -> list[MonitorInfo]:
    """
    Get list of available monitors.

    Returns:
        List of MonitorInfo objects. Index 0 is all monitors combined,
        indices 1+ are individual monitors.
    """
    _ensure_mss()

    with mss.mss() as sct:
        monitors = []
        for i, mon in enumerate(sct.monitors):
            monitors.append(MonitorInfo(
                index=i,
                left=mon["left"],
                top=mon["top"],
                width=mon["width"],
                height=mon["height"],
            ))
        return monitors


def take_screenshot(
    monitor: int | None = None,
    output_dir: Path | str | None = None,
    filename: str | None = None,
    config: ScreenshotConfig | None = None,
) -> ScreenshotResult:
    """
    Capture a screenshot of the specified monitor.

    Args:
        monitor: Monitor index (0 = all monitors, 1 = primary, 2+ = others).
                 Defaults to config.default_monitor or 1.
        output_dir: Directory to save screenshot. Defaults to config.output_dir
                    or temp/screenshots/.
        filename: Custom filename (without extension). Defaults to timestamp.
        config: Optional ScreenshotConfig for defaults.

    Returns:
        ScreenshotResult with path and metadata.

    Raises:
        ImportError: If mss is not installed.
        ValueError: If monitor index is invalid.
    """
    _ensure_mss()

    # Apply config defaults
    if monitor is None:
        monitor = config.default_monitor if config else 1
    if output_dir is None:
        output_dir = config.output_dir if config else _DEFAULT_OUTPUT_DIR

    # Resolve output directory
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.time()

    # Generate filename
    if filename:
        fname = f"{filename}.png"
    else:
        # Format: screenshot_YYYYMMDD_HHMMSS_monitor.png
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        fname = f"screenshot_{time_str}_mon{monitor}.png"

    output_path = save_dir / fname

    with mss.mss() as sct:
        # Validate monitor index
        if monitor < 0 or monitor >= len(sct.monitors):
            available = len(sct.monitors) - 1
            raise ValueError(
                f"Invalid monitor index {monitor}. "
                f"Available: 0 (all) or 1-{available}"
            )

        # Capture
        mon = sct.monitors[monitor]
        screenshot: ScreenShot = sct.grab(mon)

        # Convert to PNG bytes
        png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        # Save to file
        with open(output_path, "wb") as f:
            f.write(png_bytes)

        log.info(
            "Screenshot captured: %s (%dx%d, monitor %d)",
            output_path.name, screenshot.width, screenshot.height, monitor
        )

        return ScreenshotResult(
            path=output_path,
            width=screenshot.width,
            height=screenshot.height,
            monitor_index=monitor,
            timestamp=timestamp,
            _png_bytes=png_bytes,
        )


def take_screenshot_all_monitors(
    output_dir: Path | str | None = None,
    filename: str | None = None,
    config: ScreenshotConfig | None = None,
) -> ScreenshotResult:
    """
    Capture a screenshot of all monitors combined.

    Convenience function that calls take_screenshot with monitor=0.
    """
    return take_screenshot(
        monitor=0, output_dir=output_dir, filename=filename, config=config
    )
