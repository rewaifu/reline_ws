"""FastAPI application: one WebSocket endpoint, `/run`.

Everything protocol-shaped lives next door: `protocol.py` (frames),
`session.py` (routing and per-connection state), `handlers/` (methods),
`pipeline.py` + `preprocess/` (the work itself), `progress.py` (the numbers
the UI draws). This module only owns the socket lifecycle.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import __version__, patches
from .gate import BusyGate
from .pipeline import resolve_path
from .protocol import E_ERROR, FrameError, error_payload, unpack
from .session import Connection

logger = logging.getLogger("uvicorn.error")


def create_app(
    *,
    root: str | None = None,
    models_dir: str | None = None,
    gate: BusyGate | None = None,
) -> FastAPI:
    """App factory.

    `root` and `models_dir` are the deployment's path bases: `RELINE_ROOT` /
    `RELINE_MODELS_DIR` (or the `--root` / `--models` flags of `app.py`) are how
    a machine says where its data and models live, so the UI only has to know
    the address. A run may still override either one in `start`.
    """
    patches.apply()
    app = FastAPI(title="reline_ws", version=__version__)
    busy_gate = gate if gate is not None else BusyGate()
    raw_root = root if root is not None else os.environ.get("RELINE_ROOT")
    raw_models = (
        models_dir
        if models_dir is not None
        else os.environ.get("RELINE_MODELS_DIR")
    )
    default_root = os.path.abspath(raw_root) if raw_root else None
    # `--models weights` under `--root /data` means `/data/weights`, the same
    # rule every other path follows
    default_models = resolve_path(raw_models, default_root) if raw_models else None

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Plain HTTP probe, for whoever is debugging a deployment.

        A reverse proxy that cannot reach this process answers 502/504 for
        every path, WebSocket or not; a 200 here proves the origin is alive
        and the WebSocket trouble is in the proxy's upgrade path instead.
        The path bases are part of the answer: "which folder does this
        deployment read from" is the first question when a run finds nothing.
        """
        return {
            "ok": True,
            "version": __version__,
            "busy": busy_gate.busy,
            "root": default_root,
            "models": default_models,
        }

    @app.websocket("/run")
    async def run_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        conn = Connection(
            ws,
            busy_gate,
            root=default_root,
            models_dir=default_models,
        )
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
