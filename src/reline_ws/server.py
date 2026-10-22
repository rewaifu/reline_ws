"""FastAPI application: one WebSocket endpoint, `/run`.

Everything protocol-shaped lives next door: `protocol.py` (frames),
`session.py` (routing and per-connection state), `handlers/` (methods),
`pipeline.py` + `preprocess/` (the work itself), `progress.py` (the numbers
the UI draws). This module only owns the socket lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, patches
from .gate import BusyGate
from .modeldb import CACHE_NAME, DEFAULT_URL, ModelCatalog, default_cache_dir
from .pipeline import resolve_path
from .protocol import E_ERROR, FrameError, error_payload, unpack
from .runs import RunRegistry
from .session import Connection

logger = logging.getLogger("uvicorn.error")


async def watch_loop_lag(*, interval: float = 1.0, threshold: float = 3.0) -> None:
    """Say so in the log when something blocks the event loop.

    A blocked loop answers nothing — no `echo`, no `ls`, no new connection — and
    from the outside that is indistinguishable from a dead server: clients drop
    the socket and cannot redial until it clears. The cause is always code that
    should have been in a worker thread, and without this line the only symptom
    is "the UI disconnects when a run starts".
    """
    last = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        now = time.monotonic()
        lag = now - last - interval
        if lag > threshold:
            logger.warning(
                "event loop was blocked for %.1fs: heartbeats and new connections "
                "were not served meanwhile",
                lag,
            )
        last = now


def create_app(
    *,
    root: str | None = None,
    models_dir: str | None = None,
    gate: BusyGate | None = None,
    cache_dir: str | None = None,
    model_db: ModelCatalog | None = None,
    presets_dir: str | None = None,
) -> FastAPI:
    """App factory.

    `root` and `models_dir` are the deployment's path bases: `RELINE_ROOT` /
    `RELINE_MODELS_DIR` (or the `--root` / `--models` flags of `app.py`) are how
    a machine says where its data and models live, so the UI only has to know
    the address. A run may still override either one in `start`.

    `presets_dir` (`RELINE_PRESETS_DIR`) is where the presets a user saves in the
    UI are kept on this machine; without it the runner stores none.

    `cache_dir` (`RELINE_CACHE_DIR`) is where the runner keeps what it fetched
    for clients — today the model list. `RELINE_MODELS_DB_URL` overrides the
    database address; an empty value switches the network off. `model_db` hands
    in a ready catalog (tests).
    """
    patches.apply()
    raw_cache = cache_dir if cache_dir is not None else os.environ.get("RELINE_CACHE_DIR")
    catalog = (
        model_db
        if model_db is not None
        else ModelCatalog(
            os.environ.get("RELINE_MODELS_DB_URL", DEFAULT_URL),
            os.path.join(
                os.path.abspath(raw_cache) if raw_cache else default_cache_dir(),
                CACHE_NAME,
            ),
        )
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # the last good copy is served at once; a fresh one is fetched behind it,
        # so a slow or unreachable database never delays the runner starting
        catalog.load_cache()
        refreshing = asyncio.create_task(catalog.refresh())
        watching = asyncio.create_task(watch_loop_lag())
        try:
            yield
        finally:
            refreshing.cancel()
            watching.cancel()

    app = FastAPI(title="reline_ws", version=__version__, lifespan=lifespan)
    # `/health` is read by the UI straight from the browser (is the link up, is a
    # run going): without CORS the page could not tell this server's answer from
    # a network failure. Only GET, and WebSockets are not subject to CORS at all.
    origins = [
        item.strip()
        for item in os.environ.get("RELINE_CORS_ORIGINS", "*").split(",")
        if item.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET"],
        allow_headers=["*"],
        expose_headers=["X-Krot-Link", "Retry-After"],
    )
    busy_gate = gate if gate is not None else BusyGate()
    registry = RunRegistry(busy_gate)
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
    raw_presets = presets_dir if presets_dir is not None else os.environ.get("RELINE_PRESETS_DIR")
    default_presets = resolve_path(raw_presets, default_root) if raw_presets else None

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Plain HTTP probe, for whoever is debugging a deployment.

        A reverse proxy that cannot reach this process answers 502/504 for
        every path, WebSocket or not; a 200 here proves the origin is alive
        and the WebSocket trouble is in the proxy's upgrade path instead.
        The path bases are part of the answer: "which folder does this
        deployment read from" is the first question when a run finds nothing.
        `run_id` names the run in flight, so a client that lost its socket
        can re-attach instead of guessing.
        """
        return {
            "ok": True,
            "version": __version__,
            "busy": busy_gate.busy,
            "root": default_root,
            "models": default_models,
            "model_db": catalog.describe(),
            "presets": default_presets,
            "run_id": registry.current().run_id if registry.current() is not None else None,
        }

    @app.websocket("/run")
    async def run_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        conn = Connection(
            ws,
            busy_gate,
            root=default_root,
            models_dir=default_models,
            registry=registry,
            model_db=catalog,
            presets_dir=default_presets,
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
