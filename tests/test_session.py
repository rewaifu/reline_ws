"""Connection routing: envelope handling, ls, presets, run sessions."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from typing import Any

from _harness import Checks, finish

from reline_ws.gate import BusyGate
from reline_ws.handlers.configs import config_path
from reline_ws.handlers.fs import list_directory, parse_extensions
from reline_ws.protocol import E_ECHO, E_ERROR, E_LS, FrameError, done_payload, pack, unpack
from reline_ws.runs import BusyError
from reline_ws.session import Connection
from starlette.websockets import WebSocketDisconnect


class FakeSocket:
    """Collects the frames a connection would send."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.closed = False

    async def send_bytes(self, payload: bytes) -> None:
        self.frames.append(unpack(payload))

    async def close(self) -> None:
        self.closed = True

    def of(self, method: str) -> list[dict[str, Any]]:
        return [frame for frame in self.frames if frame["m"] == method]

    @property
    def last(self) -> dict[str, Any]:
        return self.frames[-1]


class DeadSocket(FakeSocket):
    """The client is gone: every write raises, exactly like starlette when the
    tab was closed mid-run (`ClientDisconnected` from uvicorn looks the same)."""

    def __init__(self, error: type[BaseException] = WebSocketDisconnect) -> None:
        super().__init__()
        self.error = error
        self.writes = 0

    async def send_bytes(self, payload: bytes) -> None:
        self.writes += 1
        raise self.error()


def make_connection() -> tuple[Connection, FakeSocket]:
    socket = FakeSocket()
    return Connection(socket, BusyGate()), socket


#: a config the job can actually finish (unknown preprocessors are skipped,
#: an empty image loop over zero steps is a success) without touching images
QUICK = {"nodes": [], "preprocess": [{"type": "wat", "options": {}}]}


async def scenario(checks: Checks, tmp: str) -> None:
    conn, socket = make_connection()

    # -- envelope ------------------------------------------------------
    await conn.dispatch({"m": "echo", "id": 5, "d": {"t": 1730000000000}})
    checks.eq("echo answers with the same payload", socket.last, {"m": "echo", "id": 5, "d": {"t": 1730000000000}})
    checks.eq("echo method constant", E_ECHO, "echo")

    await conn.dispatch({"m": "nope", "id": 6, "d": {}})
    checks.eq("unknown method is an error", socket.last["m"], E_ERROR)
    checks.true("…that names the method", "nope" in socket.last["d"]["message"])
    checks.eq("…and is not fatal", socket.last["d"]["fatal"], False)

    await conn.dispatch({"m": "echo"})
    checks.eq("a frame without id still answers", (socket.last["m"], socket.last["id"]), (E_ECHO, 0))

    # -- ls ------------------------------------------------------------
    await conn.dispatch({"m": "ls", "id": 7, "d": {"path": os.path.join(tmp, "fol")}})
    payload = socket.last["d"]
    checks.eq("ls completes a prefix", sorted(payload["entries"]), ["folder", "folder2"])
    checks.eq("…and marks directories", sorted(payload["dirs"]), ["folder", "folder2"])

    await conn.dispatch({"m": "ls", "id": 8, "d": {"path": os.path.join(tmp, "folder", "")}})
    entries = socket.last["d"]["entries"]
    checks.true("a closed slash lists the folder", "weights.pth" in entries and "deep" in entries)
    checks.eq("ext narrows files but keeps dirs", sorted(socket.last["d"]["dirs"]), ["deep"])

    await conn.dispatch({"m": "ls", "id": 9, "d": {"path": os.path.join(tmp, "folder", ""), "ext": ["pth"]}})
    checks.eq("…to the wanted extension", sorted(socket.last["d"]["entries"]), ["deep", "weights.pth"])

    await conn.dispatch({"m": "ls", "id": 10, "d": {"path": os.path.join(tmp, "folder", ""), "files_only": True}})
    checks.eq("files_only drops directories", sorted(socket.last["d"]["entries"]), ["note.txt", "weights.pth"])
    checks.eq("ls method constant", E_LS, "ls")

    await conn.dispatch({"m": "ls", "id": 11, "d": {}})
    checks.eq("ls without a path is an error", socket.last["m"], E_ERROR)

    # -- path bases: `root` works for ls too ---------------------------
    await conn.dispatch({"m": "ls", "id": 112, "d": {"path": "fol", "root": tmp}})
    checks.eq("ls resolves a relative path against the request root", sorted(socket.last["d"]["entries"]), ["folder", "folder2"])
    await conn.dispatch({"m": "ls", "id": 113, "d": {"path": "fol"}})
    checks.eq("…and the root sticks for the connection", sorted(socket.last["d"]["entries"]), ["folder", "folder2"])
    await conn.dispatch({"m": "ls", "id": 114, "d": {"path": "/definitely/not/there"}})
    checks.eq("an absolute path is still taken as-is", socket.last["d"]["entries"], [])
    checks.eq("…and does not disturb the base", conn.root, os.path.abspath(tmp))
    await conn.dispatch({"m": "ls", "id": 115, "d": {"path": "deep", "root": os.path.join(tmp, "folder")}})
    checks.eq("a later request may move the base", (socket.last["d"]["entries"], socket.last["d"]["dirs"]), (["deep"], ["deep"]))

    # -- path bases: `models` ------------------------------------------
    checks.eq("without `models` nothing is overridden", conn.resolve_models(None), None)
    checks.eq("a relative models path lands under root", conn.resolve_models("weights"), os.path.join(tmp, "folder", "weights"))
    checks.eq("…and is remembered for the next start", conn.resolve_models(None), os.path.join(tmp, "folder", "weights"))
    checks.eq("an absolute models path survives", conn.resolve_models(os.path.join(tmp, "m2")), os.path.join(tmp, "m2"))
    checks.eq("junk is ignored", conn.resolve_models(7), os.path.join(tmp, "m2"))
    served = Connection(socket, BusyGate(), models_dir="/srv/models")
    checks.eq("without `models` the server default applies", served.resolve_models(None), "/srv/models")
    checks.eq("…and a blank string counts as absent", served.resolve_models("  "), "/srv/models")

    # -- launch-time bases (`--root` / `--models`) ----------------------
    launched = Connection(socket, BusyGate(), root="data", models_dir="weights")
    checks.eq("a launch root is absolute from the start", launched.root, os.path.abspath("data"))
    checks.eq("…and serves a relative ls without any `root` in the frame", launched.resolve_root(None), os.path.abspath("data"))
    checks.eq("…and a relative models path lands under it", launched.resolve_models(None), os.path.join(os.path.abspath("data"), "weights"))
    checks.eq("a run may still override the launch root", launched.resolve_root("/elsewhere"), "/elsewhere")
    checks.eq("…and the models base follows", launched.resolve_models("m"), "/elsewhere/m")
    checks.eq("a blank override keeps the launch root", launched.resolve_root("  "), "/elsewhere")

    # -- presets -------------------------------------------------------
    conn.prepare_configs_dir(os.path.join(tmp, "configs"))
    await conn.dispatch({"m": "config_list", "id": 12, "d": {}})
    checks.eq("preset list", socket.last["d"]["entries"], ["a.json", "b.json"])

    await conn.dispatch({"m": "config_read", "id": 13, "d": {"name": "a.json"}})
    checks.eq("preset read", (socket.last["d"]["name"], socket.last["d"]["content"]), ("a.json", "{}"))

    await conn.dispatch({"m": "config_read", "id": 14, "d": {"name": "../secret"}})
    checks.eq("escaping names are refused", socket.last["m"], E_ERROR)

    await conn.dispatch({"m": "config_delete", "id": 15, "d": {"name": "b.json"}})
    checks.eq("preset delete answers ok", (socket.last["m"], socket.last["id"]), ("ok", 15))
    checks.true("…and the file is gone", not os.path.exists(os.path.join(tmp, "configs", "b.json")))

    # -- start validation ----------------------------------------------
    await conn.dispatch({"m": "start", "id": 16, "d": {}})
    checks.eq("start without a pipeline is refused", socket.last["d"]["message"], "pipeline required")
    checks.eq("…and no run exists", conn.registry.current(), None)

    await conn.dispatch({"m": "start", "id": 17, "d": {"pipeline": {"nodes": "nope"}}})
    checks.true("a broken config is refused", socket.last["d"]["message"].startswith("invalid pipeline config"))
    checks.eq("…and the server is still free", conn.registry.current(), None)

    # -- one run, two watchers ------------------------------------------
    await conn.dispatch({"m": "start", "id": 18, "d": {"pipeline": QUICK}})
    accepted = socket.last
    checks.eq("a run is accepted", accepted["m"], "accepted")
    run_id = accepted["d"].get("run_id")
    checks.true("…with a run_id", isinstance(run_id, str) and bool(run_id))
    checks.eq("…and the registry holds it", conn.registry.current() is not None and conn.registry.current().run_id, run_id)

    await conn.dispatch({"m": "start", "id": 19, "d": {"pipeline": QUICK}})
    checks.eq("start while owning a run is refused", socket.last["d"]["message"], "запуск уже идёт")

    # a second connection shares the registry, like two sockets of one server
    watcher = Connection(FakeSocket(), conn.gate, registry=conn.registry)
    await watcher.dispatch({"m": "start", "id": 20, "d": {"pipeline": QUICK}})
    busy = watcher.ws.last
    checks.eq("a second run is refused", busy["d"]["message"], "worker busy")
    checks.eq("…naming the run in flight", busy["d"].get("run_id"), run_id)

    await watcher.dispatch({"m": "status", "id": 21, "d": {}})
    naming = watcher.ws.last
    checks.eq("status without id names the active run", (naming["m"], naming["d"]["run_id"]), ("status", run_id))
    checks.eq("…as running", naming["d"]["status"], "running")

    await watcher.dispatch({"m": "attach", "id": 22, "d": {"run_id": run_id}})
    attached = watcher.ws.last
    checks.eq("attach answers attached", attached["m"], "attached")
    checks.eq("…to the same run", attached["d"]["run_id"], run_id)

    record = conn.registry.current()
    assert record is not None and record.task is not None
    await record.task
    starter_done = [frame for frame in socket.of("done")]
    watcher_done = [frame for frame in watcher.ws.of("done")]
    checks.eq("the starter gets its done", len(starter_done), 1)
    checks.eq("…and so does the attacher", len(watcher_done), 1)
    checks.eq("…successful", (starter_done[0]["d"]["ok"], watcher_done[0]["d"]["ok"]), (True, True))
    checks.eq("…carrying the run_id", starter_done[0]["d"].get("run_id"), run_id)
    checks.eq("every watcher keeps its own envelope id", (starter_done[0]["id"], watcher_done[0]["id"]), (18, 22))
    checks.eq("the server is free again", conn.registry.current(), None)

    await watcher.dispatch({"m": "status", "id": 23, "d": {"run_id": run_id}})
    retained = watcher.ws.last
    checks.eq("a finished run still answers status", (retained["m"], retained["d"]["status"]), ("status", "done"))
    checks.eq("…with its result", retained["d"]["result"]["ok"], True)

    await conn.dispatch({"m": "status", "id": 24, "d": {"run_id": "nope"}})
    checks.eq("an unknown run_id is an error", (socket.last["m"], "unknown run_id" in socket.last["d"]["message"]), ("error", True))
    await conn.dispatch({"m": "attach", "id": 25, "d": {"run_id": "nope"}})
    checks.eq("…for attach too", socket.last["m"], "error")
    fresh = Connection(FakeSocket(), BusyGate())
    await fresh.dispatch({"m": "attach", "id": 26, "d": {}})
    checks.eq("attach with no active run is an error", fresh.ws.last["d"]["message"], "no active run")
    await fresh.dispatch({"m": "status", "id": 27, "d": {}})
    checks.eq("…while status without runs is idle", (fresh.ws.last["m"], fresh.ws.last["d"]["status"]), ("status", "idle"))

    # -- stop ------------------------------------------------------------
    stopper, stop_socket = make_connection()
    live = stopper.registry.start(40)
    stopper.owned = live.run_id
    stopper.registry.attach(stopper, live, 40)
    await stopper.dispatch({"m": "stop", "id": 41, "d": {}})
    checks.eq("stop answers ok", (stop_socket.last["m"], stop_socket.last["id"]), ("ok", 41))
    checks.true("…and raises the run's cancel flag", live.cancel_event.is_set())
    stopper.registry.finish(live, "cancelled", done_payload(ok=False, cancelled=True))
    checks.eq("…and the server is free", stopper.registry.current(), None)

    idle_conn, idle_socket = make_connection()
    await idle_conn.dispatch({"m": "stop", "id": 42, "d": {}})
    checks.eq("stop with nothing running still answers ok", idle_socket.last["m"], "ok")

    # -- the client vanishes mid-run: the run survives --------------------
    dead = DeadSocket()
    gone = Connection(dead, BusyGate())
    checks.eq("a healthy send reports success", await Connection(FakeSocket(), BusyGate()).send({"m": "echo", "id": 30, "d": {}}), True)
    survivor = gone.registry.start(31)
    gone.owned = survivor.run_id
    gone.registry.attach(gone, survivor, 31)
    checks.eq("the first failed write reports the dead socket", await gone.send({"m": "progress", "id": 31, "d": {}}), False)
    checks.eq("…and marks the connection closed", gone.closed, True)
    checks.eq("…which forgot the run", (gone.owned, gone.follow), (None, {}))
    checks.eq("…but the run itself is still active", gone.registry.current() is survivor, True)
    checks.eq("…still marked running", survivor.status, "running")
    checks.true("…its cancel flag untouched", not survivor.cancel_event.is_set())
    await gone.shutdown()
    checks.eq("shutdown after a dead write stays quiet", gone.closed, True)
    gone.registry.finish(survivor, "cancelled", done_payload(ok=False, cancelled=True))
    checks.eq("…and the server is free once the run ends", gone.registry.current(), None)

    # a serialization bug is not a disconnect: it must still be loud
    alive = Connection(FakeSocket(), BusyGate())
    raised = ""
    try:
        await alive.send({"m": "x", "id": 32, "d": {"bad": object()}})
    except Exception as exc:  # noqa: BLE001 - the point is that it raises at all
        raised = type(exc).__name__
    checks.true("a packing bug still raises (a dead client must not hide it)", raised != "")
    checks.eq("…and does not mark the connection closed", alive.closed, False)

    shutdown_conn = Connection(FakeSocket(), BusyGate())
    await shutdown_conn.shutdown()
    checks.eq("shutdown closes the connection for good", shutdown_conn.closed, True)

    # -- broadcast: dead watchers leave, live ones stay --------------------
    pub_conn, pub_socket = make_connection()
    pub_record = pub_conn.registry.start(50)
    pub_conn.registry.attach(pub_conn, pub_record, 50)
    lurker = Connection(DeadSocket(), pub_conn.gate, registry=pub_conn.registry)
    pub_conn.registry.attach(lurker, pub_record, 51)
    await pub_conn.registry.publish(pub_record, "progress", {"percent": 10})
    checks.eq("the dead watcher was dropped", sorted(pub_record.holders), sorted([id(pub_conn)]))
    outcome = done_payload(ok=True)
    outcome["run_id"] = pub_record.run_id
    # Terminal frame first: `finish` unsubscribes the watchers, so publishing
    # after it would deliver `done` to nobody.
    await pub_conn.registry.publish(pub_record, "done", dict(outcome))
    checks.eq("…which still reaches the live watcher", pub_socket.last["m"], "done")
    pub_conn.registry.finish(pub_record, "done", outcome)
    checks.eq("finish frees the server", pub_conn.registry.current(), None)
    # -- registry races ----------------------------------------------------
    gate_conn, _ = make_connection()
    first = gate_conn.registry.start(60)
    raised_busy = ""
    try:
        gate_conn.registry.start(61)
    except BusyError as exc:
        raised_busy = str(exc)
    checks.eq("a second registry start raises BusyError", raised_busy, "worker busy")
    gate_conn.registry.abort(first)
    checks.eq("abort frees the gate without retaining", (gate_conn.registry.current(), gate_conn.registry.retained()), (None, None))

    stored = done_payload(ok=False, error="boom")
    checks.eq("done payload carries the failure", (stored["ok"], stored["error"]), (False, "boom"))


def main() -> int:
    checks = Checks("session")
    print("session:", flush=True)
    tmp = tempfile.mkdtemp(prefix="reline_ws_session_")
    try:
        os.makedirs(os.path.join(tmp, "folder", "deep"))
        os.makedirs(os.path.join(tmp, "folder2"))
        os.makedirs(os.path.join(tmp, "configs"))
        with open(os.path.join(tmp, "folder", "weights.pth"), "wb") as handle:
            handle.write(b"w")
        with open(os.path.join(tmp, "folder", "note.txt"), "wb") as handle:
            handle.write(b"t")
        with open(os.path.join(tmp, "configs", "a.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        with open(os.path.join(tmp, "configs", "b.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")

        checks.eq("parse_extensions normalizes", parse_extensions([".PTH", "pt", "."]), {"pth", "pt"})
        checks.eq("…and ignores junk", parse_extensions("pth"), None)
        checks.eq("a missing directory lists nothing", list_directory(os.path.join(tmp, "nope", "x")), ([], []))
        checks.eq("config_path rejects nested names", config_path(tmp, "a/b.json"), None)
        checks.eq("…and accepts plain ones", config_path(tmp, "a.json"), os.path.join(tmp, "a.json"))

        asyncio.run(scenario(checks, tmp))

        checks.raises("a non-map frame is a FrameError", FrameError, unpack, pack([1, 2, 3]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return finish(checks)


if __name__ == "__main__":
    sys.exit(main())
