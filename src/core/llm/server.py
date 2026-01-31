"""
Manages the llama.cpp server as a child subprocess.

Responsibilities:
  - Build the CLI command from config
  - Start the process and pipe its output to the logger
  - Poll the /health endpoint until the server is ready
  - Gracefully terminate on shutdown
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import httpx

from src.core.config import AppConfig

log = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 1.0   # seconds between /health checks
_HEALTH_TIMEOUT = 120.0       # seconds before giving up on server start


class LlamaServer:
    """Async context-manager that owns the llama-server process."""

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config.server
        self._health_url = f"http://{self._cfg.host}:{self._cfg.port}/health"
        self._process: asyncio.subprocess.Process | None = None
        self._log_task: asyncio.Task | None = None

    # ── build command ────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        exe = self._cfg.executable
        if not Path(exe).exists():
            raise FileNotFoundError(
                f"llama-server executable not found: {exe}"
            )
        if not self._cfg.model_path:
            raise ValueError(
                "server.model_path is required in config.yaml "
                "(path to your .gguf model file)"
            )
        if not Path(self._cfg.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self._cfg.model_path}"
            )

        cmd = [
            exe,
            "-m", self._cfg.model_path,
            "-c", str(self._cfg.context_size),
            "-n", str(self._cfg.n_predict),
            "-ngl", str(self._cfg.gpu_layers),
            "--host", self._cfg.host,
            "--port", str(self._cfg.port),
        ]
        cmd.extend(self._cfg.extra_args)
        return cmd

    # ── log forwarding ───────────────────────────────────────────────

    async def _forward_output(self) -> None:
        """Read stderr from the server process and forward to our logger."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            async for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    log.debug("[llama-server] %s", text)
        except asyncio.CancelledError:
            pass

    # ── health polling ───────────────────────────────────────────────

    async def _wait_ready(self) -> None:
        """Block until the server's /health endpoint returns 200."""
        elapsed = 0.0
        async with httpx.AsyncClient() as client:
            while elapsed < _HEALTH_TIMEOUT:
                try:
                    resp = await client.get(self._health_url, timeout=5.0)
                    if resp.status_code == 200:
                        log.info("llama-server is ready.")
                        return
                except httpx.ConnectError:
                    pass
                except httpx.ReadError:
                    pass

                await asyncio.sleep(_HEALTH_POLL_INTERVAL)
                elapsed += _HEALTH_POLL_INTERVAL

        raise TimeoutError(
            f"llama-server did not become healthy within {_HEALTH_TIMEOUT}s"
        )

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the server subprocess and wait until it's healthy."""
        cmd = self._build_cmd()
        log.info("Starting llama-server: %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        # Forward server logs in the background
        self._log_task = asyncio.create_task(self._forward_output())

        print("Waiting for llama-server to load model...", flush=True)
        await self._wait_ready()

    async def stop(self) -> None:
        """Gracefully terminate the server."""
        if self._process is None:
            return

        log.info("Stopping llama-server (pid=%d)...", self._process.pid)
        self._process.terminate()

        try:
            await asyncio.wait_for(self._process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            log.warning("llama-server did not exit in time — killing.")
            self._process.kill()
            await self._process.wait()

        if self._log_task:
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass

        log.info("llama-server stopped.")
        self._process = None

    # ── async context-manager ────────────────────────────────────────

    async def __aenter__(self) -> LlamaServer:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()
