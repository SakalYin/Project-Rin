"""
Manages the llama.cpp server as a child subprocess.

This is an internal module used by the ``openai_compat`` provider.
It is NOT part of the public provider API.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 1.0   # seconds between /health checks
_HEALTH_TIMEOUT = 120.0       # seconds before giving up on server start


class LlamaServer:
    """Async context-manager that owns the llama-server process."""

    def __init__(self, config: dict) -> None:
        self._cfg = config
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8080)
        self._health_url = f"http://{host}:{port}/health"
        self._process: asyncio.subprocess.Process | None = None
        self._log_task: asyncio.Task | None = None

    # ── build command ────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        exe = self._cfg.get("executable", "")
        if not exe or not Path(exe).exists():
            raise FileNotFoundError(
                f"llama-server executable not found: {exe}"
            )
        model_path = self._cfg.get("model_path", "")
        if not model_path:
            raise ValueError(
                "server.model_path is required in config.yaml "
                "(path to your .gguf model file)"
            )
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        cmd = [
            exe,
            "-m", model_path,
            "-c", str(self._cfg.get("context_size", 2048)),
            "-n", str(self._cfg.get("n_predict", 512)),
            "-ngl", str(self._cfg.get("gpu_layers", -1)),
            "--host", self._cfg.get("host", "127.0.0.1"),
            "--port", str(self._cfg.get("port", 8080)),
        ]
        cmd.extend(self._cfg.get("extra_args", []))
        return cmd

    # ── log forwarding ───────────────────────────────────────────────

    async def _forward_output(self) -> None:
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
        cmd = self._build_cmd()
        log.info("Starting llama-server: %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._log_task = asyncio.create_task(self._forward_output())

        print("Waiting for llama-server to load model...", flush=True)
        await self._wait_ready()

    async def stop(self) -> None:
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

    async def __aenter__(self) -> LlamaServer:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()
