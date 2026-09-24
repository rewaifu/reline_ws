"""`start` / `stop` / `status` / `attach`: the run lifecycle.

Path bases come from the request and are all optional: `root` (every node
path, the configs folder, a relative `models`), `models` (model files),
`configs` (presets folder). Without them the config is used as written, which
keeps old absolute-path configs running.

Validation happens before the gate is taken, so a rejected request can never
leave the server marked busy. The job always ends in a `finally` that retains
the outcome and releases the gate — a crashed pipeline, a stopped run or an
abandoned socket all free the server for the next start.

Failure contract: an exception anywhere in the run produces
`done {ok: false, error}` — never a successful `done`. The UI keeps errors in
its run log, and a lying "готово" right after an error was exactly the bug
that made error messages appear to vanish.

Session contract: the run is owned by the registry, not by the socket that
started it. A dead socket only unsubscribes; the job keeps going, and a
reconnected client learns its `run_id` via `status` and resumes watching via
`attach`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ..pipeline import PipelineWs
from ..preprocess import MODELS_DIR, run_preprocessors
from ..progress import Phase, ProgressTracker, pipeline_windows, preprocess_windows
from ..protocol import (
    E_ACCEPTED,
    E_ATTACHED,
    E_DONE,
    E_ERROR,
    E_STATUS,
    done_payload,
    error_payload,
)
from ..runs import BusyError, RunRecord, RunRegistry

if TYPE_CHECKING:
    from ..session import Connection

logger = logging.getLogger("uvicorn.error")


def _busy_reply(record: RunRecord | None) -> dict[str, Any]:
    payload = error_payload("worker busy")
    if record is not None:
        payload["run_id"] = record.run_id
    return payload


async def handle_start(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    if conn.owned is not None:
        record = conn.registry.get(conn.owned)
        if record is not None and record.status == "running":
            await conn.reply(request_id, E_ERROR, error_payload("запуск уже идёт"))
            return
        conn.owned = None
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
    try:
        record = conn.registry.start(request_id)
    except BusyError as exc:
        other = conn.registry.get(exc.run_id) if exc.run_id else conn.registry.current()
        await conn.reply(request_id, E_ERROR, _busy_reply(other))
        return
    try:
        conn.prepare_configs_dir(d.get("configs"))
    except OSError as exc:
        conn.registry.abort(record)
        await conn.reply(request_id, E_ERROR, error_payload(f"cannot prepare configs dir: {exc}"))
        return
    conn.owned = record.run_id
    conn.registry.attach(conn, record, request_id)
    await conn.reply(request_id, E_ACCEPTED, {"run_id": record.run_id})
    record.task = asyncio.create_task(
        _run_job(conn.registry, record, nodes, preprocess, models_dir, root)
    )


async def handle_stop(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    raw = d.get("run_id")
    run_id = raw if isinstance(raw, str) and raw else (conn.owned or _followed(conn))
    record = conn.registry.get(run_id) if run_id else conn.registry.current()
    if record is None or record.status != "running":
        logger.debug("stop with nothing running")
        await conn.reply(request_id, "ok")
        return
    record.cancel_event.set()
    await conn.reply(request_id, "ok")


def _followed(conn: "Connection") -> str | None:
    for run_id in conn.follow:
        return run_id
    return None


async def handle_status(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    raw = d.get("run_id")
    if isinstance(raw, str) and raw:
        record = conn.registry.get(raw)
        if record is None:
            await conn.reply(request_id, E_ERROR, error_payload("unknown run_id"))
            return
        await conn.reply(request_id, E_STATUS, conn.registry.snapshot(record))
        return
    record = conn.registry.current() or conn.registry.retained()
    if record is None:
        await conn.reply(request_id, E_STATUS, {"status": "idle"})
        return
    await conn.reply(request_id, E_STATUS, conn.registry.snapshot(record))


async def handle_attach(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    raw = d.get("run_id")
    record = conn.registry.get(raw) if isinstance(raw, str) and raw else conn.registry.current()
    if record is None:
        if isinstance(raw, str) and raw:
            await conn.reply(request_id, E_ERROR, error_payload("unknown run_id"))
        else:
            await conn.reply(request_id, E_ERROR, error_payload("no active run"))
        return
    # an attach remembers the run as "ours" when we are not already busy with
    # our own: `stop` without a `run_id` then stops the watched run
    if conn.owned is None or conn.registry.get(conn.owned) is None:
        conn.owned = record.run_id
    conn.registry.attach(conn, record, request_id)
    await conn.reply(request_id, E_ATTACHED, conn.registry.snapshot(record))


async def _run_job(
    registry: RunRegistry,
    record: RunRecord,
    nodes: list[dict],
    preprocess: list[dict],
    models_dir: str,
    root: str | None,
) -> None:
    async def publish(method: str, payload: dict[str, Any] | None) -> None:
        await registry.publish(record, method, payload)

    tracker = ProgressTracker(publish, record.request_id)
    cancelled = False
    error_text: str | None = None
    try:
        if not nodes and not preprocess:
            raise ValueError("pipeline is empty: nothing to do")
        # Two phases, two bars: the preprocessors fill the bar from 0 %, and
        # `begin_phase` resets it once they are through, so a long download
        # does not eat the image loop's scale.
        if preprocess:
            await tracker.begin_phase(Phase.PREPROCESS)
            keep_going = await run_preprocessors(
                preprocess,
                nodes,
                root,
                tracker,
                preprocess_windows(preprocess),
                record.cancel_event,
                models_dir,
            )
            cancelled = not keep_going
        if not cancelled:
            await tracker.begin_phase(Phase.PROCESS)
            read_window, image_window = pipeline_windows()
            pipeline = PipelineWs.build(nodes)
            cancelled = not await pipeline.run(tracker, record.cancel_event, read_window, image_window)
        if not cancelled:
            await tracker.finish()
    except Exception as exc:
        logger.exception("pipeline error")
        error_text = str(exc) or type(exc).__name__
    status = "cancelled" if cancelled else ("failed" if error_text else "done")
    outcome = done_payload(ok=error_text is None and not cancelled, cancelled=cancelled, error=error_text)
    outcome["run_id"] = record.run_id
    # Terminal frame first: `finish` unsubscribes the watchers, so publishing
    # after it would deliver `done` to nobody — exactly the silent-run bug.
    await registry.publish(record, E_DONE, dict(outcome))
    registry.finish(record, status, outcome)
