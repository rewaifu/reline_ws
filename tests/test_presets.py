"""Presets on this machine's disk, the event-loop guard, and the run start that
must never block the loop."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import threading
import time

from _harness import Checks, finish

from reline_ws.gate import BusyGate
from reline_ws.pipeline import PipelineWs
from reline_ws.protocol import E_CONFIG_LIST, E_CONFIG_WRITE, E_ERROR
from reline_ws.server import watch_loop_lag
from reline_ws.session import Connection
from test_session import FakeSocket


class Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


async def check_presets(checks: Checks, tmp: str) -> None:
    # -- no folder configured: nothing to save to, and the client can tell ----
    bare = Connection(FakeSocket(), BusyGate())
    await bare.dispatch({"m": "config_list", "id": 1, "d": {}})
    checks.eq("without a presets dir the list says it is disabled", (bare.ws.last["m"], bare.ws.last["d"]), (E_CONFIG_LIST, {"entries": [], "enabled": False}))
    await bare.dispatch({"m": "config_write", "id": 2, "d": {"name": "a.json", "content": "{}"}})
    checks.eq("…and a write is an error, not a silent no-op", bare.ws.last["m"], E_ERROR)

    # -- a folder from the deployment, created on the first write ---------------
    folder = os.path.join(tmp, "presets")
    conn = Connection(FakeSocket(), BusyGate(), presets_dir=folder)
    await conn.dispatch({"m": "config_list", "id": 3, "d": {}})
    checks.eq("a configured but missing folder lists as enabled and empty", conn.ws.last["d"], {"entries": [], "enabled": True})
    checks.eq("…listing creates nothing", os.path.exists(folder), False)

    await conn.dispatch({"m": "config_write", "id": 4, "d": {"name": "user-1.json", "content": '{"name": "Мой пресет"}'}})
    checks.eq("a write answers under its own name", (conn.ws.last["m"], conn.ws.last["id"]), (E_CONFIG_WRITE, 4))
    checks.eq("…the file holds exactly the text (UTF-8)", open(os.path.join(folder, "user-1.json"), encoding="utf-8").read(), '{"name": "Мой пресет"}')

    checks.eq("the file is readable like any other (0644)", oct(os.stat(os.path.join(folder, "user-1.json")).st_mode & 0o777), "0o644")
    await conn.dispatch({"m": "config_write", "id": 5, "d": {"name": "user-1.json", "content": "[]"}})
    await conn.dispatch({"m": "config_read", "id": 6, "d": {"name": "user-1.json"}})
    checks.eq("a second write replaces the file", conn.ws.last["d"]["content"], "[]")
    checks.eq("…leaving no temp file behind", sorted(os.listdir(folder)), ["user-1.json"])

    # a writer's own temp file is not a preset
    with open(os.path.join(folder, ".preset-x.tmp"), "w") as handle:
        handle.write("half")
    await conn.dispatch({"m": "config_list", "id": 7, "d": {}})
    checks.eq("temp files are not listed", conn.ws.last["d"]["entries"], ["user-1.json"])

    for ident, name in (("nested", "a/b.json"), ("up", "../x.json"), ("empty", ""), ("number", 7)):
        await conn.dispatch({"m": "config_write", "id": 8, "d": {"name": name, "content": "{}"}})
        checks.eq(f"a {ident} name is refused", conn.ws.last["m"], E_ERROR)
    checks.eq("…and nothing escaped the folder", sorted(os.listdir(tmp)), ["presets"])

    await conn.dispatch({"m": "config_write", "id": 9, "d": {"name": "x.json", "content": {"not": "text"}}})
    checks.eq("content must be text", conn.ws.last["m"], E_ERROR)
    await conn.dispatch({"m": "config_write", "id": 10, "d": {"name": "big.json", "content": "x" * (2 * 1024 * 1024 + 1)}})
    checks.eq("an oversized write is refused", conn.ws.last["m"], E_ERROR)
    checks.eq("…and not written", os.path.exists(os.path.join(folder, "big.json")), False)

    await conn.dispatch({"m": "config_delete", "id": 11, "d": {"name": "user-1.json"}})
    await conn.dispatch({"m": "config_list", "id": 12, "d": {}})
    checks.eq("a deleted preset is gone from the list", conn.ws.last["d"]["entries"], [])

    # -- a relative folder lands under the launch root --------------------------
    rooted = Connection(FakeSocket(), BusyGate(), root=tmp, presets_dir="saved")
    await rooted.dispatch({"m": "config_write", "id": 13, "d": {"name": "r.json", "content": "{}"}})
    checks.true("a relative presets dir goes under --root", os.path.isfile(os.path.join(tmp, "saved", "r.json")))

    # -- a start may still name its own folder ----------------------------------
    other = os.path.join(tmp, "other")
    conn.prepare_configs_dir(other)
    await conn.dispatch({"m": "config_write", "id": 14, "d": {"name": "o.json", "content": "{}"}})
    checks.true("…and the override wins for that connection", os.path.isfile(os.path.join(other, "o.json")))


async def check_loop_guard(checks: Checks) -> None:
    handler = Records()
    log = logging.getLogger("uvicorn.error")
    log.addHandler(handler)
    try:
        watcher = asyncio.create_task(watch_loop_lag(interval=0.05, threshold=0.2))
        await asyncio.sleep(0.2)
        checks.eq("a free loop is not reported", handler.lines, [])
        time.sleep(0.5)  # exactly the bug: blocking work on the event loop
        await asyncio.sleep(0.15)
        watcher.cancel()
    finally:
        log.removeHandler(handler)
    checks.true("a blocked loop is reported with its duration", any("blocked for" in line for line in handler.lines))


async def check_start_is_off_loop(checks: Checks) -> None:
    """`build` loads the models (seconds, CUDA): it must run in a worker thread."""
    threads: list[bool] = []
    original = PipelineWs.build.__func__  # type: ignore[attr-defined]

    def spy(cls, nodes, postprocess=None):  # noqa: ANN001
        threads.append(threading.current_thread() is threading.main_thread())
        return original(cls, nodes, postprocess)

    PipelineWs.build = classmethod(spy)  # type: ignore[method-assign]
    try:
        conn = Connection(FakeSocket(), BusyGate())
        # no preprocessors: the job goes straight to `build` (which then refuses a
        # chain without a reader — what matters here is the thread it ran on)
        level = {"low_input": 0, "high_input": 255, "low_output": 0, "high_output": 255, "gamma": 1.0}
        await conn.dispatch({"m": "start", "id": 1, "d": {"pipeline": {"nodes": [{"type": "level", "options": level}]}}})
        record = conn.registry.current()
        assert record is not None and record.task is not None
        await record.task
    finally:
        PipelineWs.build = classmethod(original)  # type: ignore[method-assign]
    checks.eq("the pipeline is built once", len(threads), 1)
    checks.eq("…and not on the event loop's thread", threads, [False])
    checks.eq("…and the run still ends with a done frame", len(conn.ws.of("done")), 1)


def main() -> int:
    checks = Checks("presets")
    print("presets:", flush=True)
    tmp = tempfile.mkdtemp(prefix="reline_ws_presets_")
    try:
        asyncio.run(check_presets(checks, tmp))
        asyncio.run(check_loop_guard(checks))
        asyncio.run(check_start_is_off_loop(checks))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return finish(checks)


if __name__ == "__main__":
    raise SystemExit(main())
