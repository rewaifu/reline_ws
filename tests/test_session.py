"""Connection routing: envelope handling, ls, presets, start/stop contracts."""

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
from reline_ws.session import Connection


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


def make_connection() -> tuple[Connection, FakeSocket]:
    socket = FakeSocket()
    return Connection(socket, BusyGate()), socket


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

    # -- start / stop contracts ----------------------------------------
    await conn.dispatch({"m": "start", "id": 16, "d": {}})
    checks.eq("start without a pipeline is refused", socket.last["d"]["message"], "pipeline required")
    checks.eq("…and the gate stays free", conn.gate.busy, False)

    await conn.dispatch({"m": "start", "id": 17, "d": {"pipeline": {"nodes": "nope"}}})
    checks.true("a broken config is refused", socket.last["d"]["message"].startswith("invalid pipeline config"))
    checks.eq("…and the gate is still free", conn.gate.busy, False)

    conn.gate.acquire()
    await conn.dispatch({"m": "start", "id": 18, "d": {"pipeline": {"nodes": []}}})
    checks.eq("a second run is refused", socket.last["d"]["message"], "worker busy")
    conn.gate.release()

    conn.phase = "running"
    await conn.dispatch({"m": "start", "id": 19, "d": {"pipeline": {"nodes": []}}})
    checks.eq("start while running is refused", socket.last["d"]["message"], "запуск уже идёт")
    conn.phase = "idle"

    await conn.dispatch({"m": "stop", "id": 20, "d": {}})
    checks.eq("stop answers ok", (socket.last["m"], socket.last["id"]), ("ok", 20))
    checks.true("…and raises the cancel flag", conn.cancel_event.is_set())

    # -- job finish is always a released gate --------------------------
    conn.gate.acquire()
    conn.phase = "running"
    conn.finish_run()
    checks.eq("finish_run frees the gate", conn.gate.busy, False)
    checks.eq("…and the connection is idle again", conn.phase, "idle")

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
