"""Preprocessors: model install, archives, and the progress they report."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from typing import Any

from _harness import Checks, finish

from reline_ws.preprocess import run_preprocessors
from reline_ws.preprocess.archives import count_archives, dearchive
from reline_ws.preprocess.models import find_installed, install_model, rewrite_upscale_models
from reline_ws.progress import ProgressTracker, preprocess_windows


class Recorder:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, method: str, payload: dict[str, Any]) -> None:
        if method == "progress":
            self.frames.append(payload)

    @property
    def stages(self) -> list[str]:
        return [frame["stage"] for frame in self.frames]


def make_zip(path: str, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def main() -> int:
    checks = Checks("preprocess")
    print("preprocess:", flush=True)
    tmp = tempfile.mkdtemp(prefix="reline_ws_test_")
    try:
        models = os.path.join(tmp, "models")
        os.makedirs(models)

        # -- model install ---------------------------------------------
        with open(os.path.join(models, "4x_a.pth"), "wb") as handle:
            handle.write(b"weights")
        checks.eq("installed model: stem match, no download", install_model(models, "4x_a", None), os.path.join(models, "4x_a.pth"))
        checks.eq("find_installed returns None when absent", find_installed(models, "4x_zzz"), None)

        archive = os.path.join(tmp, "4x_b.tar.xz")
        make_zip(archive, {"sub/dir/4x_b.pth": b"weights-b"})
        installed = install_model(models, "4x_b", "file://" + archive)
        checks.eq("archive url: model found inside and moved", os.path.basename(installed), "4x_b.pth")
        checks.eq("…with its bytes", open(installed, "rb").read(), b"weights-b")

        raw = os.path.join(tmp, "4x_c.safetensors")
        with open(raw, "wb") as handle:
            handle.write(b"weights-c")
        checks.eq(
            "model file url: moved as-is",
            os.path.basename(install_model(models, "4x_c", "file://" + raw)),
            "4x_c.safetensors",
        )
        checks.raises("missing model without url raises", ValueError, install_model, models, "4x_missing", None)

        nodes = [
            {"type": "upscale", "options": {"model": "4x_b", "dtype": "F32"}},
            {"type": "upscale", "options": {"model": "4x_a"}},
            {"type": "folder_reader", "options": {"path": "/x"}},
        ]
        checks.eq("upscale nodes repointed by name", rewrite_upscale_models(nodes, "4x_b", "/models/4x_b.pth"), 1)
        checks.eq("…the matching one", nodes[0]["options"]["model"], "/models/4x_b.pth")
        checks.eq("…and only it", nodes[1]["options"]["model"], "4x_a")

        # -- archives --------------------------------------------------
        room = os.path.join(tmp, "room")
        os.makedirs(room)
        inner = os.path.join(tmp, "inner.tar.gz")
        with tarfile.open(inner, "w:gz") as tar:
            tar.add(os.path.join(models, "4x_a.pth"), arcname="img.png")
        make_zip(os.path.join(room, "pack.zip"), {"data/inner.tar.gz": open(inner, "rb").read()})
        checks.eq("archives are counted before unpacking", count_archives(room), 1)
        dearchive(room)
        checks.true("the source archive is gone", not os.path.exists(os.path.join(room, "pack.zip")))
        checks.true("nested archive unpacked in place", os.path.isfile(os.path.join(room, "pack", "data", "inner", "img.png")))

        # -- orchestration + progress ----------------------------------
        base = os.path.join(tmp, "base")
        os.makedirs(base)
        make_zip(os.path.join(base, "raws.zip"), {"a.png": b"a", "b.png": b"b"})
        # a model that is NOT installed yet, otherwise the step is a no-op
        fresh = os.path.join(tmp, "4x_d.tar.xz")
        make_zip(fresh, {"4x_d.pth": b"weights-d" * 2048})
        preprocess = [
            {"type": "download", "options": {"name": "4x_d", "url": "file://" + fresh}},
            {"type": "unarchive", "options": {"path": "raws.zip"}},
        ]
        job_nodes = [{"type": "upscale", "options": {"model": "4x_d"}}]
        recorder = Recorder()
        tracker = ProgressTracker(recorder, request_id=7)
        windows, head = preprocess_windows(preprocess)
        cancel = asyncio.Event()

        async def run() -> bool:
            return await run_preprocessors(preprocess, job_nodes, base, tracker, windows, cancel, models)

        checks.true("preprocessors finish", asyncio.run(run()))
        checks.eq("both stages are reported", sorted(set(recorder.stages)), ["download", "unarchive"])
        checks.eq("the bar stops at the head share", tracker.percent, head)
        download_frames = [frame for frame in recorder.frames if frame["stage"] == "download"]
        checks.true("download reports bytes", any(frame.get("bytes_done", 0) > 0 for frame in download_frames))
        checks.eq("download is labelled with the model", download_frames[0]["label"], "4x_d")
        unpack_frames = [frame for frame in recorder.frames if frame["stage"] == "unarchive"]
        checks.eq("unpack counts archives", (unpack_frames[-1]["done"], unpack_frames[-1]["total"]), (1, 1))
        checks.eq("unpack is labelled with the archive", unpack_frames[0]["label"], "raws.zip")
        checks.true("unpacked into its folder", os.path.isfile(os.path.join(base, "raws", "a.png")))
        checks.true("source archive removed", not os.path.exists(os.path.join(base, "raws.zip")))
        checks.eq("upscale node repointed at the installed model", job_nodes[0]["options"]["model"], os.path.join(models, "4x_d.pth"))

        cancel.set()
        checks.eq("a cancelled run reports nothing new", asyncio.run(run()), False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return finish(checks)


if __name__ == "__main__":
    sys.exit(main())
