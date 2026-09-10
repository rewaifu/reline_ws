"""Per-connection state machine for the WS_API.md protocol.

One connection = one request/response channel plus at most one run. The
receive loop hands frames to `dispatch`, which is a pure routing table: the
handlers live in `handlers/` and every blocking thing they do (filesystem,
network, the pipeline itself) happens in a worker thread or a task, so the
socket keeps answering `echo` while work is in flight.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .gate import BusyGate
from .handlers import configs, fs, run
from .protocol import (
    E_ECHO,
    E_ERROR,
    M_CONFIG_DELETE,
    M_CONFIG_LIST,
    M_CONFIG_READ,
    M_ECHO,
    M_LS,
    M_START,
    M_STOP,
    error_payload,
    event,
    pack,
)
from .pipeline import resolve_path

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger("uvicorn.error")

Handler = Callable[["Connection", int, dict[str, Any]], Awaitable[None]]

#: method name -> handler. `echo` is handled inline in `dispatch`: it is the
#: heartbeat and must never queue behind anything else.
ROUTES: dict[str, Handler] = {
    M_START: run.handle_start,
    M_STOP: run.handle_stop,
    M_LS: fs.handle_ls,
    M_CONFIG_LIST: configs.handle_list,
    M_CONFIG_READ: configs.handle_read,
    M_CONFIG_DELETE: configs.handle_delete,
}


class Connection:
    def __init__(
        self,
        ws: "WebSocket",
        gate: BusyGate,
        *,
        models_dir: str | None = None,
    ) -> None:
        self.ws = ws
        self.gate = gate
        #: server-wide fallback for model files (`RELINE_MODELS_DIR`)
        self.default_models_dir = models_dir
        #: per-run override: `models` from the last `start`
        self.models_dir: str | None = None
        self.phase = "idle"
        self.root: str | None = None
        self.configs_dir: str | None = None
        #: request id of the current run: every run event carries it
        self.run_id = 0
        self.cancel_event = asyncio.Event()
        self.job_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()

    # -- transport -------------------------------------------------------

    async def send(self, message: dict[str, Any]) -> None:
        """One binary frame; the lock keeps concurrent writers (job events
        next to request replies) from interleaving partial frames."""
        payload = pack(message)
        async with self._send_lock:
            await self.ws.send_bytes(payload)

    async def reply(self, request_id: int, method: str, d: dict[str, Any] | None = None) -> None:
        await self.send(event(method, request_id, d))

    async def send_event(self, method: str, d: dict[str, Any] | None = None) -> None:
        await self.send(event(method, self.run_id, d))

    # -- run state -------------------------------------------------------

    def resolve_root(self, raw: Any) -> str | None:
        """The base every path of this connection is resolved against.

        Sent by `start`, and optionally by `ls` (browsing before a run, or a
        different base while the run is going). Once set it sticks for the
        connection: an `ls` without `root` keeps browsing where `start` pointed.
        Absolute from now on, so relative node paths resolve against one stable
        base, and an old config full of absolute paths is left alone.
        """
        if isinstance(raw, str) and raw.strip():
            self.root = os.path.abspath(raw)
        return self.root

    def resolve_models(self, raw: Any) -> str | None:
        """`models` from `start`: where downloads land and upscale models are
        looked up. Resolved like node paths, so `models: "weights"` under
        `root: "/data"` means `/data/weights`. Absent -> the server default,
        which keeps every existing deployment working unchanged.
        """
        if isinstance(raw, str) and raw.strip():
            self.models_dir = resolve_path(raw, self.root)
        return self.models_dir or self.default_models_dir

    def prepare_configs_dir(self, raw: Any) -> None:
        if isinstance(raw, str) and raw:
            self.configs_dir = resolve_path(raw, self.root)
            os.makedirs(self.configs_dir, exist_ok=True)

    def finish_run(self) -> None:
        """The job is over: whatever it ended with, the server is free."""
        self.gate.release()
        self.phase = "idle"
        self.job_task = None

    async def shutdown(self) -> None:
        """Socket is going away: stop the run and wait for it to unwind, so
        the busy gate is released before the next connection arrives."""
        self.cancel_event.set()
        task = self.job_task
        if task is not None and not task.done():
            try:
                await task
            except Exception:
                logger.debug("job failed while the connection was closing", exc_info=True)
        try:
            await self.ws.close()
        except Exception:
            pass

    # -- routing ---------------------------------------------------------

    async def dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("m")
        raw_id = message.get("id")
        request_id = raw_id if isinstance(raw_id, int) and raw_id >= 0 else 0
        payload = message.get("d")
        d = payload if isinstance(payload, dict) else {}
        if method == M_ECHO:
            await self.reply(request_id, E_ECHO, d)
            return
        handler = ROUTES.get(str(method))
        if handler is None:
            await self.reply(request_id, E_ERROR, error_payload(f"unknown method {method!r}"))
            return
        await handler(self, request_id, d)
