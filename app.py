from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import msgpack
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rebind.pipeline import PipelineWs, resolve_path
from rebind.preprocess import run_preprocessors

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# One pipeline at a time across all connections (GPU-bound work). The event
# loop makes the check-and-set in handle_start atomic, so no lock is needed.
_busy = False


def _unpack(data: bytes) -> dict[str, Any]:
    message = msgpack.unpackb(data, raw=False)
    if not isinstance(message, dict):
        raise ValueError("frame must be a map")
    return message


class Connection:
    """Per-connection state machine for the WS_API.md protocol."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.phase = "idle"
        self.root: str | None = None
        self.configs_dir: str | None = None
        self.cancel_event = asyncio.Event()
        self.job_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    async def send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(msgpack.packb(message))

    async def reply(self, request_id: int, m: str, d: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"m": m, "id": request_id}
        if d is not None:
            message["d"] = d
        await self.send(message)

    # -- method handlers -------------------------------------------------

    async def handle_start(self, request_id: int, d: dict[str, Any]) -> None:
        global _busy
        if self.phase != "idle":
            await self.reply(request_id, "error", {"message": "запуск уже идёт", "fatal": False})
            return
        if _busy:
            await self.reply(request_id, "error", {"message": "worker busy", "fatal": False})
            return

        raw = d.get("pipeline")
        if not isinstance(raw, str):
            await self.reply(request_id, "error", {"message": "pipeline must be a JSON string", "fatal": False})
            return
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as e:
            await self.reply(request_id, "error", {"message": f"invalid pipeline json: {e}", "fatal": False})
            return

        root = d.get("root")
        if isinstance(root, str) and root:
            self.root = os.path.abspath(root)
        configs = d.get("configs")
        if isinstance(configs, str) and configs:
            self.configs_dir = resolve_path(configs, self.root)
            os.makedirs(self.configs_dir, exist_ok=True)

        try:
            nodes, preprocess = PipelineWs.prepare_config(config, self.root)
        except Exception as e:
            logger.debug("invalid pipeline config: %s", e)
            await self.reply(request_id, "error", {"message": f"invalid pipeline config: {e}", "fatal": False})
            return

        _busy = True
        self.phase = "running"
        self.cancel_event.clear()
        await self.reply(request_id, "accepted")
        self.job_task = asyncio.create_task(self._run_job(request_id, nodes, preprocess, bool(preprocess)))

    async def _run_job(
        self, request_id: int, nodes: list[dict], preprocess: list[dict], has_preprocess: bool
    ) -> None:

        global _busy

        async def send_event(m: str, d: dict[str, Any]) -> None:
            await self.send({"m": m, "id": request_id, "d": d})

        async def send_preprocess_progress(percent: int) -> None:
            await send_event("progress", {"percent": percent, "step": "preprocess"})

        finished = True
        try:
            pre_ok = await run_preprocessors(
                preprocess, nodes, self.root, send_preprocess_progress, self.cancel_event
            )
            if not pre_ok:
                finished = False
            else:
                # built only after preprocess: download may rewrite the
                # upscale model paths to the installed files
                pipeline = PipelineWs.build(nodes)
                finished = await pipeline.process_ws(send_event, self.cancel_event, has_preprocess)
        except Exception as e:
            logger.exception("pipeline error")
            await send_event("error", {"message": str(e), "fatal": False})

        _busy = False
        self.phase = "idle"
        if finished:
            await send_event("done", {"ok": True, "cancelled": False})
        else:
            await send_event("done", {"ok": False, "cancelled": True})

    async def handle_stop(self, request_id: int) -> None:
        was_running = self.phase != "idle"
        self.cancel_event.set()
        await self.reply(request_id, "ok")

    async def handle_ls(self, request_id: int, d: dict[str, Any]) -> None:
        query = d.get("path")
        if not isinstance(query, str) or not query:
            await self.reply(request_id, "error", {"message": "path required", "fatal": False})
            return
        files_only = d.get("files_only") is True
        raw_ext = d.get("ext")
        ext = None
        if isinstance(raw_ext, list):
            ext = {str(e).lower().lstrip(".") for e in raw_ext}
        resolved = resolve_path(query, self.root)
        directory, prefix = os.path.split(resolved)
        directory = directory or "."
        entries: list[str] = []
        dirs: list[str] = []
        if os.path.isdir(directory):
            for name in sorted(os.listdir(directory)):
                if not name.startswith(prefix):
                    continue
                is_dir = os.path.isdir(os.path.join(directory, name))
                if files_only and is_dir:
                    continue
                # ext narrows files only: directories stay so the client can
                # keep browsing deeper (model path picker needs this)
                if ext and not is_dir and os.path.splitext(name)[1].lstrip(".").lower() not in ext:
                    continue
                entries.append(name)
                if is_dir:
                    dirs.append(name)
        await self.reply(request_id, "ls", {"entries": entries, "dirs": dirs})

    def _config_path(self, name: Any) -> str | None:
        """Config presets live flat inside configs_dir; anything missing,
        nested or escaping (../) is rejected."""
        if not isinstance(name, str) or not name or "/" in name or ".." in name:
            return None
        if self.configs_dir is None:
            return None
        return os.path.join(self.configs_dir, name)

    async def handle_config_list(self, request_id: int) -> None:
        names: list[str] = []
        if self.configs_dir and os.path.isdir(self.configs_dir):
            names = sorted(
                name
                for name in os.listdir(self.configs_dir)
                if os.path.isfile(os.path.join(self.configs_dir, name))
            )
        await self.reply(request_id, "config_list", {"entries": names})

    async def handle_config_read(self, request_id: int, d: dict[str, Any]) -> None:
        path = self._config_path(d.get("name"))
        if path is None:
            await self.reply(request_id, "error", {"message": "name required (and configs dir set)", "fatal": False})
            return
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            await self.reply(request_id, "error", {"message": str(e), "fatal": False})
            return
        await self.reply(request_id, "config_read", {"name": d["name"], "content": content})

    async def handle_config_delete(self, request_id: int, d: dict[str, Any]) -> None:
        path = self._config_path(d.get("name"))
        if path is None:
            await self.reply(request_id, "error", {"message": "name required (and configs dir set)", "fatal": False})
            return
        try:
            os.remove(path)
        except OSError as e:
            await self.reply(request_id, "error", {"message": str(e), "fatal": False})
            return
        await self.reply(request_id, "ok")

    async def dispatch(self, message: dict[str, Any]) -> None:
        m = message.get("m")
        request_id = int(message.get("id") or 0)
        d = message.get("d") or {}
        if m == "echo":
            await self.reply(request_id, "echo", d)
        elif m == "start":
            await self.handle_start(request_id, d)
        elif m == "stop":
            await self.handle_stop(request_id)
        elif m == "ls":
            await self.handle_ls(request_id, d)
        elif m == "config_list":
            await self.handle_config_list(request_id)
        elif m == "config_read":
            await self.handle_config_read(request_id, d)
        elif m == "config_delete":
            await self.handle_config_delete(request_id, d)
        else:
            await self.reply(request_id, "error", {"message": f"unknown method {m!r}", "fatal": False})


@app.websocket("/run")
async def run_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    conn = Connection(ws)
    try:
        while True:
            try:
                frame = await ws.receive_bytes()
            except (WebSocketDisconnect, RuntimeError):
                break
            try:
                message = _unpack(frame)
            except Exception:
                await conn.reply(0, "error", {"message": "invalid msgpack frame", "fatal": False})
                continue
            await conn.dispatch(message)
    except Exception:
        logger.exception("ws handler error")
    finally:
        conn.cancel_event.set()
        if conn.job_task is not None and not conn.job_task.done():
            try:
                await conn.job_task
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass
