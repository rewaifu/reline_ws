"""Progress arithmetic: bar windows, speed, ETA, throttling."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from _harness import Checks, finish

from reline_ws.progress import (
    MIN_INTERVAL_S,
    ProgressTracker,
    Stage,
    Window,
    pipeline_windows,
    preprocess_windows,
    preprocess_weights,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Recorder:
    """Collects the frames a tracker would put on the wire."""

    def __init__(self) -> None:
        self.frames: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, method: str, payload: dict[str, Any]) -> None:
        self.frames.append((method, payload))

    @property
    def progress(self) -> list[dict[str, Any]]:
        return [payload for method, payload in self.frames if method == "progress"]

    @property
    def last(self) -> dict[str, Any]:
        return self.progress[-1]


async def scenario(checks: Checks) -> None:
    clock = FakeClock()
    recorder = Recorder()
    tracker = ProgressTracker(recorder, request_id=1, clock=clock)

    read_window, image_window = pipeline_windows(0.0)
    await tracker.open_stage(Stage.READ, read_window, label="Folder Reader", node="folder_reader")
    checks.eq("open_stage announces the step", recorder.last["stage"], "read")
    checks.eq("…with the label", recorder.last["label"], "Folder Reader")
    checks.eq("…from the window start", recorder.last["percent"], 0)

    clock.advance(1.0)
    await tracker.update(done=50, total=50)
    checks.eq("read window is a sliver of the bar", recorder.last["percent"], 5)

    await tracker.open_stage(Stage.PROCESS, image_window, label="Sharp", node="sharp", total=100)
    checks.eq("process window opens after read", recorder.last["percent"], 5)
    checks.eq("counters are reported", (recorder.last["done"], recorder.last["total"]), (0, 100))

    clock.advance(1.0)
    await tracker.update(done=25, total=100)
    first = recorder.last
    checks.eq("quarter of the images move a quarter of the window", first["percent"], 28)
    checks.eq("rate is units per second", first["rate"], 25.0)
    checks.true("eta is reported", first["eta"] > 0)
    checks.eq("elapsed comes from the clock", first["elapsed"], 2.0)

    # throttle: a burst inside MIN_INTERVAL_S must not add frames
    sent_after_first = len(recorder.progress)
    for done in range(26, 40):
        await tracker.update(done=done, total=100)
    checks.eq("bursts are throttled", len(recorder.progress), sent_after_first)

    clock.advance(MIN_INTERVAL_S)
    await tracker.update(done=40, total=100)
    checks.eq("next frame after the interval", len(recorder.progress), sent_after_first + 1)

    # monotone: a stale or shrinking counter never walks the bar backwards
    before = tracker.percent
    clock.advance(1.0)
    await tracker.update(done=5, total=100)
    checks.eq("percent never decreases", tracker.percent, before)

    # the hot node may switch the stage inside the same window (writer step)
    clock.advance(1.0)
    await tracker.update(done=100, total=100, stage=Stage.WRITE, label="Folder Writer", node="folder_writer")
    checks.eq("stage follows the hot node", recorder.last["stage"], "write")
    checks.eq("…and the window is complete", recorder.last["percent"], 100)

    await tracker.finish()
    checks.eq("finish fills the bar", recorder.last["percent"], 100)

    # download: bytes, not items
    clock2 = FakeClock()
    recorder2 = Recorder()
    downloader = ProgressTracker(recorder2, request_id=2, clock=clock2)
    await downloader.open_stage(Stage.DOWNLOAD, Window(0.0, 60.0), label="4x_a")
    clock2.advance(2.0)
    await downloader.update(bytes_done=4096, bytes_total=8192, force=True)
    frame = recorder2.last
    checks.eq("download reports bytes", (frame["bytes_done"], frame["bytes_total"]), (4096, 8192))
    checks.eq("download percent comes from the bytes", frame["percent"], 30)
    checks.eq("download speed is bytes per second", frame["rate"], 2048.0)
    checks.eq("no item counters while downloading", "done" in frame, False)


def run_tracker_checks(checks: Checks) -> None:
    asyncio.run(scenario(checks))


def main() -> int:
    checks = Checks("progress")
    print("progress:", flush=True)

    checks.eq("download outweighs unpack", preprocess_weights([{"type": "download"}, {"type": "unarchive"}]), [1.0, 0.25])
    windows, head = preprocess_windows([{"type": "download"}, {"type": "unarchive"}])
    checks.eq("head share with a download", round(head, 3), 60.0)
    checks.eq("download window", (round(windows[0].start, 3), round(windows[0].end, 3)), (0.0, 48.0))
    checks.eq("unarchive window", (round(windows[1].start, 3), round(windows[1].end, 3)), (48.0, 60.0))

    lone, lone_head = preprocess_windows([{"type": "unarchive"}])
    checks.eq("a lone unpack takes a small head", round(lone_head, 3), 15.0)
    checks.eq("no preprocessors -> no head", preprocess_windows([]), ([], 0.0))

    read, images = pipeline_windows(60.0)
    checks.eq("read window starts where the head ends", read.start, 60.0)
    checks.eq("images own the rest of the bar", (round(images.start, 3), images.end), (62.0, 100.0))
    checks.eq("window maps counters to percent", Window(10.0, 30.0).percent(1, 4), 15.0)
    checks.eq("unknown total keeps the window start", Window(10.0, 30.0).percent(3, 0), 10.0)

    run_tracker_checks(checks)
    return finish(checks)


if __name__ == "__main__":
    sys.exit(main())
