"""`start` / `stop`: the run lifecycle of one connection.

Path bases come from the request and are all optional: `root` (every node
path, the configs folder, a relative `models`), `models` (model files),
`configs` (presets folder). Without them the config is used as written, which
keeps old absolute-path configs running.

Validation happens before the gate is taken, so a rejected request can never
leave the server marked busy. The job always ends in a `finally` that releases
the gate — a crashed pipeline, a dead socket or a cancelled run all free the
server for the next start.

Failure contract: an exception anywhere in the run produces
`done {ok: false, error}` — never a successful `done`. The UI keeps errors in
its run log, and a lying "готово" right after an error was exactly the bug
that made error messages appear to vanish.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ..pipeline import PipelineWs
from ..preprocess import MODELS_DIR, run_preprocessors
from ..progress import ProgressTracker, pipeline_windows, preprocess_windows
from ..protocol import E_ACCEPTED, E_DONE, E_ERROR, done_payload, error_payload

if TYPE_CHECKING:
    from ..session import Connection

logger = logging.getLogger("uvicorn.error")


async def handle_start(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    if conn.phase != "idle":
        await conn.reply(request_id, E_ERROR, error_payload("запуск уже идёт"))
        return
    raw_pipeline = d.get("pipeline")
    if raw_pipeline is None:
        await conn.reply(request_id, E_ERROR, error_payload("pipeline required"))
        return
    try:
        root = conn.resolve_root(d.get("root"))
        nodes, preprocess = PipelineWs.prepare_config(raw_pipeline, root)
    except Exception as exc:
        logger.debug("invalid pipeline config: %s", exc)
        await conn.reply(request_id, E_ERROR, error_payload(f"invalid pipeline config: {exc}"))
        return
    # both path bases come from `start` and are optional: absent `root` keeps
    # absolute/relative paths exactly as the config wrote them, absent `models`
    # keeps the server-wide folder (`RELINE_MODELS_DIR`).
    models_dir = conn.resolve_models(d.get("models")) or MODELS_DIR
    if not conn.gate.acquire():
        await conn.reply(request_id, E_ERROR, error_payload("worker busy"))
        return
    try:
        conn.prepare_configs_dir(d.get("configs"))
    except OSError as exc:
        conn.gate.release()
        await conn.reply(request_id, E_ERROR, error_payload(f"cannot prepare configs dir: {exc}"))
        return
    conn.phase = "running"
    conn.run_id = request_id
    conn.cancel_event.clear()
    await conn.reply(request_id, E_ACCEPTED)
    conn.job_task = asyncio.create_task(_run_job(conn, request_id, nodes, preprocess, models_dir))


async def handle_stop(conn: "Connection", request_id: int, _d: dict[str, Any]) -> None:
    if conn.phase == "idle":
        logger.debug("stop with nothing running")
    conn.cancel_event.set()
    await conn.reply(request_id, "ok")


async def _run_job(
    conn: "Connection",
    request_id: int,
    nodes: list[dict],
    preprocess: list[dict],
    models_dir: str,
) -> None:
    tracker = ProgressTracker(conn.send_event, request_id)
    cancelled = False
    error_text: str | None = None
    try:
        windows, head = preprocess_windows(preprocess)
        if not nodes and not preprocess:
            raise ValueError("pipeline is empty: nothing to do")
        if preprocess:
            keep_going = await run_preprocessors(
                preprocess,
                nodes,
                conn.root,
                tracker,
                windows,
                conn.cancel_event,
                models_dir,
            )
            cancelled = not keep_going
        if not cancelled:
            read_window, image_window = pipeline_windows(head)
            pipeline = PipelineWs.build(nodes)
            cancelled = not await pipeline.run(tracker, conn.cancel_event, read_window, image_window)
        if not cancelled:
            await tracker.finish()
    except Exception as exc:
        logger.exception("pipeline error")
        error_text = str(exc) or type(exc).__name__
    finally:
        conn.finish_run()

    await conn.send_event(E_DONE, done_payload(ok=error_text is None and not cancelled, cancelled=cancelled, error=error_text))
