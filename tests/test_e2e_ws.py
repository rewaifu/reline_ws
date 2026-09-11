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
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pepeline
import websockets

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


def read_log(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


#: big enough that a run which ignores the disconnect is still writing when the
#: checks below look at the output folder
ABANDON_IMAGES = 240


async def abandon(checks: Checks, tmp: str, endpoint: str, log_path: str) -> None:
    """The tab closes mid-run (the case that used to print a `pipeline error`
    traceback): the server must read it as a disconnect, stop the batch and
    free itself for the client that reconnects."""
    source = os.path.join(tmp, "abandon_in")
    target = os.path.join(tmp, "abandon_out")
    make_images(source, ABANDON_IMAGES)
    ws = await websockets.connect(endpoint, max_size=None)
    await send(
        ws,
        {
            "m": "start",
            "id": 50,
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
    checks.eq("the abandoned run starts", (await read_frame(ws))["m"], "accepted")
    await read_frame(ws)  # real work is in flight now
    transport = getattr(ws, "transport", None)
    if transport is not None:
        transport.abort()  # no close frame: the next write just fails
    else:
        await ws.close()
    checks.ok("the client vanished mid-run")

    # a fresh connection must get the server back: `busy` means the abandoned
    # run is still holding the gate
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(endpoint, max_size=None) as probe:
                await send(
                    probe,
                    {
                        "m": "start",
                        "id": 51,
                        "d": {
                            "pipeline": pipeline(
                                [
                                    {"type": "folder_reader", "options": {"path": os.path.join(tmp, "in"), "mode": "rgb", "recursive": False}},
                                    {"type": "level", "options": LEVEL_OPTIONS},
                                    {"type": "folder_writer", "options": {"path": os.path.join(tmp, "after"), "format": "png"}},
                                ]
                            )
                        },
                    },
                )
                frame = await read_frame(probe)
                if frame["m"] != "accepted":
                    await asyncio.sleep(0.3)
                    continue
                checks.ok("a reconnect takes the server back without waiting for the batch")
                try:
                    await read_until_done(probe)
                except Exception as exc:  # noqa: BLE001 - the server log says why
                    raise AssertionError(
                        f"the reconnect run failed: {exc!r}\n\n{read_log(log_path)[-2000:]}"
                    ) from exc
        except AssertionError:
            raise
        except Exception:  # noqa: BLE001 - reconnect races are expected here
            await asyncio.sleep(0.3)
            continue
        break
    else:
        raise AssertionError(f"the server stayed busy after the client vanished\n\n{read_log(log_path)[-2000:]}")

    written = len(sorted(os.listdir(target)))
    checks.true(f"…and the abandoned batch stopped early ({written} of {ABANDON_IMAGES})", written < ABANDON_IMAGES)


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
        checks.eq("a config without preprocessors has one phase", {frame["d"]["phase"] for frame in progress}, {"process"})

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

        # -- a run that would do nothing is a failure, not a green done ----
        # Reported from a deployment: reader → level with no writer, a folder
        # that is not mounted, an empty folder — all finished "ok" with no
        # output, which reads as a tool that simply does not work.
        await send(
            ws,
            {
                "m": "start",
                "id": 4,
                "d": {
                    "pipeline": pipeline(
                        [
                            {"type": "folder_reader", "options": {"path": source, "mode": "rgb"}},
                            {"type": "level", "options": LEVEL_OPTIONS},
                        ]
                    )
                },
            },
        )
        frames = await read_until_done(ws)
        checks.eq("a chain without a writer fails", frames[-1]["d"]["ok"], False)
        checks.true("…saying what is missing", "no writer" in (frames[-1]["d"]["error"] or ""))

        missing = os.path.join(tmp, "not_mounted")
        await send(
            ws,
            {
                "m": "start",
                "id": 5,
                "d": {
                    "pipeline": pipeline(
                        [
                            {"type": "folder_reader", "options": {"path": missing, "mode": "rgb"}},
                            {"type": "folder_writer", "options": {"path": target, "format": "png"}},
                        ]
                    )
                },
            },
        )
        frames = await read_until_done(ws)
        checks.eq("a reader folder that is not there fails", frames[-1]["d"]["ok"], False)
        checks.true("…naming the folder", missing in (frames[-1]["d"]["error"] or ""))

        empty_source = os.path.join(tmp, "empty")
        empty_target = os.path.join(tmp, "out_empty")
        os.makedirs(empty_source, exist_ok=True)
        await send(
            ws,
            {
                "m": "start",
                "id": 6,
                "d": {
                    "pipeline": pipeline(
                        [
                            {"type": "folder_reader", "options": {"path": empty_source, "mode": "rgb"}},
                            {"type": "folder_writer", "options": {"path": empty_target, "format": "png"}},
                        ]
                    )
                },
            },
        )
        frames = await read_until_done(ws)
        checks.eq("an empty reader folder fails", frames[-1]["d"]["ok"], False)
        checks.true("…naming the folder", empty_source in (frames[-1]["d"]["error"] or ""))
        written = sorted(os.listdir(empty_target)) if os.path.isdir(empty_target) else []
        checks.eq("…and writes no file", written, [])

        await send(ws, {"m": "start", "id": 7, "d": {"pipeline": pipeline([])}})
        frames = await read_until_done(ws)
        checks.eq("an empty pipeline fails", frames[-1]["d"]["ok"], False)

        # a refused run must not leave the server busy (the gate is released in
        # `finally`, and the next start on this very socket proves it)
        await send(
            ws,
            {
                "m": "start",
                "id": 8,
                "d": {
                    "pipeline": pipeline(
                        [
                            {"type": "folder_reader", "options": {"path": source, "mode": "rgb", "recursive": False}},
                            {"type": "folder_writer", "options": {"path": target, "format": "png"}},
                        ]
                    )
                },
            },
        )
        frames = await read_until_done(ws)
        checks.eq("the server is free after the refusals", frames[-1]["d"]["ok"], True)

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

    # -- the launch bases: the client sends a config and nothing else -----
    # A deployment sets `RELINE_ROOT`/`RELINE_MODELS_DIR` (see `main`), so the
    # frames below carry no path parameters at all — the same frames the UI
    # sends.
    base = os.path.join(tmp, "base")
    shutil.copytree(source, os.path.join(base, "src"), dirs_exist_ok=True)
    weight = os.path.join(tmp, "fake.pth")
    with open(weight, "wb") as handle:
        handle.write(b"pth")

    async with websockets.connect(endpoint, max_size=None) as ws:
        await send(ws, {"m": "ls", "id": 21, "d": {"path": "src/img_00"}})
        listing = await read_frame(ws)
        checks.eq("ls browses the launch root", sorted(listing["d"]["entries"]), [f"img_{index:03d}.png" for index in range(10)])

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
                    )
                },
            },
        )
        frames = await read_until_done(ws)
        checks.eq("relative node paths run against the launch root", frames[-1]["d"]["ok"], True)
        checks.eq("…writing under it", len(sorted(os.listdir(os.path.join(base, "done")))), 12)
        checks.eq("a relative launch models folder receives the download", os.path.exists(os.path.join(base, "weights", "fake.pth")), True)

        # two phases, two bars: the download fills the bar, then the image loop
        # starts its own scale instead of inheriting the download's sliver
        phases = [frame["d"]["phase"] for frame in frames if frame["m"] == "progress"]
        checks.eq("preprocessors run in their own phase", sorted(set(phases)), ["preprocess", "process"])
        checks.true("…which comes first", phases.index("preprocess") < phases.index("process"))
        download_percents = [frame["d"]["percent"] for frame in frames if frame["m"] == "progress" and frame["d"]["phase"] == "preprocess"]
        image_percents = [frame["d"]["percent"] for frame in frames if frame["m"] == "progress" and frame["d"]["phase"] == "process"]
        checks.eq("the preprocess phase ends full", download_percents[-1], 100)
        checks.eq("…and the process phase starts from zero", image_percents[0], 0)
        checks.true("…climbing to the end", image_percents[-1] == 100 and image_percents == sorted(image_percents))

        # no frame ever carried this: the connection starts from the launch base
        await send(ws, {"m": "ls", "id": 23, "d": {"path": "src/img_00"}})
        listing = await read_frame(ws)
        checks.eq("a later ls keeps browsing there", sorted(listing["d"]["entries"]), [f"img_{index:03d}.png" for index in range(10)])


def main() -> int:
    checks = Checks("e2e ws")
    print("e2e ws:", flush=True)
    tmp = tempfile.mkdtemp(prefix="reline_ws_e2e_")
    port = free_port()
    # the log goes to a file, not a pipe: `abandon` needs to read what the
    # server said *after* a checkpoint, and a pipe cannot be peeked at.
    log_path = os.path.join(tmp, "server.log")
    log_file = open(log_path, "w", encoding="utf-8")
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
            "info",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([SRC, REPO_ROOT]),
            # a deployment configures its bases once, at launch; the client is
            # expected to send nothing but the pipeline
            "RELINE_ROOT": os.path.join(tmp, "base"),
            "RELINE_MODELS_DIR": "weights",
        },
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        make_images(os.path.join(tmp, "in"), 12)
        if not wait_for_port(port, deadline=time.monotonic() + 60):
            raise AssertionError(f"server did not start on port {port}:\n{read_log(log_path)}")
        checks.ok("the server answers on the endpoint")
        asyncio.run(scenario(checks, tmp, f"ws://127.0.0.1:{port}/run"))
        # Deployment probes, the ones a 502 hunt needs: `/health` proves the
        # origin is alive, and a plain GET on `/run` is 404 because the route
        # only speaks the WebSocket upgrade.
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as http:
            health = http.get("/health")
            checks.eq("health answers 200", health.status_code, 200)
            checks.eq("…with the version", health.json()["ok"], True)
            checks.eq("…and the launch root it reads from", health.json()["root"], os.path.join(tmp, "base"))
            checks.eq("…including a relative models folder", health.json()["models"], os.path.join(tmp, "base", "weights"))
            checks.eq("a plain GET on /run is not a route", http.get("/run").status_code, 404)

        # Everything the deliberate failures above wrote is behind us: only what
        # the vanished client produces is read here.
        marker = len(read_log(log_path))
        asyncio.run(abandon(checks, tmp, f"ws://127.0.0.1:{port}/run", log_path))
        time.sleep(0.5)
        tail = read_log(log_path)[marker:]
        checks.true("a vanished client is logged as a disconnect", "client disconnected" in tail)
        checks.eq("…never as a pipeline error", "pipeline error" in tail, False)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return finish(checks)


if __name__ == "__main__":
    sys.exit(main())
