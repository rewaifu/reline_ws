"""End-to-end: a real uvicorn server, a real folder of images, real frames.

Runs the whole stack the way the UI does — structured `start`, MessagePack
frames, a reader → level → writer pipeline — and checks the contract that the
frontend depends on: detailed progress, the final `done`, and the failure
invariant (a broken run never ends in a successful `done`).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import Any

import numpy as np
import pepeline
import websockets
from pathlib import Path

from _harness import REPO_ROOT, SRC, Checks, finish

from reline_ws.protocol import pack, unpack

LEVEL_OPTIONS = {
    "low_input": 0,
    "high_input": 255,
    "low_output": 0,
    "high_output": 255,
    "gamma": 1.0,
}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def make_images(folder: str, count: int) -> None:
    os.makedirs(folder, exist_ok=True)
    pixel = np.zeros((24, 32, 3), dtype=np.float32)
    for index in range(count):
        pixel[:, :, 0] = index / max(count - 1, 1)
        pepeline.save(pixel, os.path.join(folder, f"img_{index:03d}.png"))


def wait_for_port(port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


async def send(ws: Any, frame: dict[str, Any]) -> None:
    await ws.send(pack(frame))


async def read_frame(ws: Any, timeout: float = 120.0) -> dict[str, Any]:
    return unpack(await asyncio.wait_for(ws.recv(), timeout=timeout))


async def read_until_done(ws: Any) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while True:
        frame = await read_frame(ws)
        frames.append(frame)
        if frame["m"] == "done":
            return frames


def pipeline(nodes: list[dict[str, Any]], preprocess: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"nodes": nodes, "preprocess": preprocess or []}


async def scenario(checks: Checks, tmp: str, endpoint: str) -> None:
    source = os.path.join(tmp, "in")
    target = os.path.join(tmp, "out")

    # -- happy path ----------------------------------------------------
    async with websockets.connect(endpoint, max_size=None) as ws:
        await send(
            ws,
            {
                "m": "start",
                "id": 1,
                "d": {
                    "pipeline": pipeline(
                        [
                            {"type": "folder_reader", "options": {"path": source, "mode": "rgb", "recursive": False}},
                            {"type": "level", "options": LEVEL_OPTIONS},
                            {"type": "folder_writer", "options": {"path": target, "format": "png"}},
                        ]
                    )
                },
            },
        )
        frames = await read_until_done(ws)
        checks.eq("the run is accepted", frames[0]["m"], "accepted")
        progress = [frame for frame in frames if frame["m"] == "progress"]
        checks.true("progress is streamed", len(progress) >= 2)

        stages = {frame["d"]["stage"] for frame in progress}
        checks.true("the read stage is reported", "read" in stages)
        checks.true("the image stages are reported", bool(stages & {"process", "write"}))

        percents = [frame["d"]["percent"] for frame in progress]
        checks.eq("percent never decreases", percents, sorted(percents))
        checks.eq("the bar ends full", percents[-1], 100)

        counted = [frame["d"] for frame in progress if frame["d"].get("total") == 12]
        checks.true("counters describe the images", bool(counted) and counted[-1]["done"] == 12)
        checks.eq("frames carry the elapsed time", all("elapsed" in frame["d"] for frame in progress), True)
        checks.true("speed is measured", any(frame["d"].get("rate") for frame in progress))
        checks.true("frames name the running node", any(frame["d"].get("node") for frame in progress))

        done = frames[-1]["d"]
        checks.eq("the run is done", (done["ok"], done["cancelled"], done["error"]), (True, False, None))
        checks.eq("every image was written", len(sorted(os.listdir(target))), 12)

        await send(ws, {"m": "ls", "id": 2, "d": {"path": os.path.join(source, "")}})
        listing = await read_frame(ws)
        checks.eq("ls works on the same socket", listing["id"], 2)
        checks.eq("…and sees the images", len(listing["d"]["entries"]), 12)

        # -- failure invariant ----------------------------------------
        await send(ws, {"m": "start", "id": 3, "d": {"pipeline": pipeline([{"type": "nope", "options": {}}])}})
        broken = await read_until_done(ws)
        final = broken[-1]["d"]
        checks.eq("a broken pipeline fails", final["ok"], False)
        checks.true("…with an error message", bool(final["error"]) or final["cancelled"] is True)
        checks.eq("…and never reports success", any(frame.get("d", {}).get("ok") is True for frame in broken), False)
        checks.eq("the connection stays usable", broken[-1]["m"], "done")

    # -- preprocessors, same server, fresh socket ----------------------
    archive = os.path.join(tmp, "pack.zip")
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in sorted(os.listdir(source)):
            zipped.write(os.path.join(source, name), name)
    unpacked = os.path.join(tmp, "unpacked")
    os.makedirs(unpacked, exist_ok=True)
    shutil.move(archive, os.path.join(unpacked, "pack.zip"))

    async with websockets.connect(endpoint, max_size=None) as ws:
        await send(
            ws,
            {
                "m": "start",
                "id": 9,
                "d": {
                    "pipeline": pipeline(
                        [
                            {"type": "folder_reader", "options": {"path": os.path.join(unpacked, "pack"), "mode": "rgb", "recursive": False}},
                            {"type": "level", "options": LEVEL_OPTIONS},
                            {"type": "folder_writer", "options": {"path": os.path.join(tmp, "out2"), "format": "png"}},
                        ],
                        [{"type": "unarchive", "options": {"path": "pack.zip"}}],
                    ),
                    "root": unpacked,
                },
            },
        )
        frames = await read_until_done(ws)
        progress = [frame for frame in frames if frame["m"] == "progress"]
        unpack_frames = [frame for frame in progress if frame["d"]["stage"] == "unarchive"]
        checks.true("the unpack stage is reported", bool(unpack_frames))
        checks.eq("…labelled with the archive", unpack_frames[0]["d"]["label"], "pack.zip")
        checks.true("…and counted", unpack_frames[-1]["d"].get("total", 0) >= 1)
        checks.eq("the unpacked source was removed", os.path.exists(os.path.join(unpacked, "pack.zip")), False)
        checks.eq("the pipeline ran after the unpack", frames[-1]["d"]["ok"], True)

    # -- `root` (ls + pipeline) and `models`, all relative ---------------
    # A fresh base: node paths are written the way a portable config would,
    # relative to the folder the whole run lives in.
    base = os.path.join(tmp, "base")
    shutil.copytree(source, os.path.join(base, "src"), dirs_exist_ok=True)
    weight = os.path.join(tmp, "fake.pth")
    with open(weight, "wb") as handle:
        handle.write(b"pth")

    async with websockets.connect(endpoint, max_size=None) as ws:
        await send(ws, {"m": "ls", "id": 21, "d": {"path": "src/img_00", "root": base}})
        listing = await read_frame(ws)
        checks.eq("ls autocompletes a relative path inside root", sorted(listing["d"]["entries"]), [f"img_{index:03d}.png" for index in range(10)])

        await send(
            ws,
            {
                "m": "start",
                "id": 22,
                "d": {
                    "pipeline": pipeline(
                        [
                            {"type": "folder_reader", "options": {"path": "src", "mode": "rgb", "recursive": False}},
                            {"type": "level", "options": LEVEL_OPTIONS},
                            {"type": "folder_writer", "options": {"path": "done", "format": "png"}},
                        ],
                        [{"type": "download", "options": {"name": "fake", "url": Path(weight).as_uri()}}],
                    ),
                    "root": base,
                    "models": "weights",
                },
            },
        )
        frames = await read_until_done(ws)
        checks.eq("relative node paths run against root", frames[-1]["d"]["ok"], True)
        checks.eq("…writing into the root", len(sorted(os.listdir(os.path.join(base, "done")))), 12)
        checks.eq("a relative models path installs under root", os.path.exists(os.path.join(base, "weights", "fake.pth")), True)

        # a second run without `models` keeps the same folder: the base and the
        # model folder stick to the connection instead of being re-derived
        await send(ws, {"m": "ls", "id": 23, "d": {"path": "src/img_00"}})
        listing = await read_frame(ws)
        checks.eq("the root from `start` serves later ls", sorted(listing["d"]["entries"]), [f"img_{index:03d}.png" for index in range(10)])


def main() -> int:
    checks = Checks("e2e ws")
    print("e2e ws:", flush=True)
    tmp = tempfile.mkdtemp(prefix="reline_ws_e2e_")
    port = free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([SRC, REPO_ROOT])},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        make_images(os.path.join(tmp, "in"), 12)
        if not wait_for_port(port, deadline=time.monotonic() + 60):
            output = server.stdout.read() if server.stdout is not None else ""
            raise AssertionError(f"server did not start on port {port}:\n{output}")
        checks.ok("the server answers on the endpoint")
        asyncio.run(scenario(checks, tmp, f"ws://127.0.0.1:{port}/run"))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        if server.stdout is not None:
            server.stdout.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return finish(checks)


if __name__ == "__main__":
    sys.exit(main())
