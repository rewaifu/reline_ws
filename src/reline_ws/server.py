"""FastAPI application: one WebSocket endpoint, `/run`.

Everything protocol-shaped lives next door: `protocol.py` (frames),
`session.py` (routing and per-connection state), `handlers/` (methods),
`pipeline.py` + `preprocess/` (the work itself), `progress.py` (the numbers
the UI draws). This module only owns the socket lifecycle.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import patches
from .gate import BusyGate
from .protocol import E_ERROR, FrameError, error_payload, unpack
from .session import Connection

logger = logging.getLogger("uvicorn.error")


def create_app(*, models_dir: str | None = None, gate: BusyGate | None = None) -> FastAPI:
    """App factory: tests inject their own gate and models folder.

    `models_dir` is only the server-wide default for model files; a `start`
    may override it per run with `d.models`.
    """
    patches.apply()
    app = FastAPI(title="reline_ws", version="0.2.0")
    busy_gate = gate if gate is not None else BusyGate()

    @app.websocket("/run")
    async def run_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        conn = Connection(ws, busy_gate, models_dir=models_dir)
        try:
            while True:
                try:
                    frame = await ws.receive_bytes()
                except (WebSocketDisconnect, RuntimeError):
                    break
                try:
                    message = unpack(frame)
                except FrameError as exc:
                    await conn.reply(0, E_ERROR, error_payload(f"invalid msgpack frame: {exc}"))
                    continue
                try:
                    await conn.dispatch(message)
                except Exception:
                    logger.exception("handler failed for %r", message.get("m"))
                    await conn.reply(0, E_ERROR, error_payload("internal error"))
        except Exception:
            logger.exception("ws handler error")
        finally:
            await conn.shutdown()

    return app


app = create_app()
