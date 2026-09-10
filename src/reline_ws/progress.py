"""Detailed run progress: one monotone `percent` plus stage, counters, rate
and ETA (WS_API.md, `progress`).

The bar is a single 0..100 scale shared by every stage of a run. Each stage
gets a *window* on that scale; within its window a stage reports `done/total`
and the tracker does the arithmetic. That keeps the accounting in one place —
the runner only says "stage X at n/m", it never computes percentages.

Sends are throttled (5/s) so a 10k-image run cannot flood the socket, and
`rate`/`eta` are derived from wall time, not from call counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

SendEvent = Callable[[str, dict[str, Any]], Awaitable[None]]

#: fast pipelines finish between two renders; the UI still needs the numbers
MIN_INTERVAL_S = 0.2
#: a rate measured over less than this is noise
MIN_RATE_WINDOW_S = 0.5


class Stage(str, Enum):
    DOWNLOAD = "download"
    UNARCHIVE = "unarchive"
    READ = "read"
    PROCESS = "process"
    WRITE = "write"


@dataclass(frozen=True)
class Window:
    """Slice of the 0..100 bar owned by one stage."""

    start: float
    end: float

    @property
    def span(self) -> float:
        return max(self.end - self.start, 0.0)

    def percent(self, done: int, total: int) -> float:
        if total <= 0:
            return self.start
        ratio = min(max(done, 0) / total, 1.0)
        return self.start + self.span * ratio


#: weights of the preprocessors inside the head of the bar: a network download
#: dominates an unpack, so the download branch of the same config gets 4x the
#: room of an unpack (it can be a 300 MB file, an archive is usually seconds).
PREPROCESS_WEIGHTS = {"download": 1.0, "unarchive": 0.25}
#: share of the bar handed to preprocessors when the config has any
PREPROCESS_SHARE = 0.6
#: share of the pipeline window spent listing files (scanning is cheap next to
#: decoding and filtering, but it is not instant on network storage)
READ_SHARE = 0.05


def preprocess_weights(preprocess: list[dict[str, Any]]) -> list[float]:
    """Relative cost of every preprocessor item, in config order."""
    weights: list[float] = []
    for item in preprocess:
        kind = item.get("type") if isinstance(item, dict) else None
        weights.append(PREPROCESS_WEIGHTS.get(str(kind), 0.25))
    return weights


def preprocess_windows(preprocess: list[dict[str, Any]]) -> tuple[list[Window], float]:
    """Windows for the preprocessors plus the share they consume overall.

    The head of the bar grows with the weight of the section: a model download
    deserves 60% of the run, a lone unpack about 15%, and a config without
    preprocessors gives the pipeline the whole bar."""
    if not preprocess:
        return [], 0.0
    weights = preprocess_weights(preprocess)
    total = sum(weights) or 1.0
    head = PREPROCESS_SHARE * 100.0 * min(1.0, total)
    windows: list[Window] = []
    cursor = 0.0
    for weight in weights:
        span = head * weight / total
        windows.append(Window(cursor, cursor + span))
        cursor += span
    return windows, head


def pipeline_windows(head_percent: float) -> tuple[Window, Window]:
    """(read window, per-image window) of the 0..100 bar."""
    read_end = head_percent + (100.0 - head_percent) * READ_SHARE
    return Window(head_percent, read_end), Window(read_end, 100.0)


class ProgressTracker:
    """Maps stage-local progress onto the shared bar and streams frames."""

    def __init__(
        self,
        send: SendEvent,
        request_id: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._send = send
        self._request_id = request_id
        self._clock = clock
        self._started = clock()
        self._percent = 0.0
        self._stage: Stage | None = None
        self._window = Window(0.0, 0.0)
        self._label: str | None = None
        self._node: str | None = None
        self._done = 0
        self._total = 0
        self._stage_started = self._started
        self._stage_start_done = 0
        self._stage_start_percent = 0.0
        self._last_sent = -MIN_INTERVAL_S
        self._bytes_done = 0
        self._bytes_total = 0

    # -- state -----------------------------------------------------------

    @property
    def percent(self) -> float:
        return self._percent

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    # -- public API ------------------------------------------------------

    async def open_stage(
        self,
        stage: Stage,
        window: Window,
        *,
        label: str | None = None,
        node: str | None = None,
        total: int = 0,
    ) -> None:
        """Enter a stage: resets the stage counters and announces the step
        before its first number arrives, so the UI can name what runs now."""
        now = self._clock()
        self._stage = stage
        self._window = window
        self._label = label
        self._node = node
        self._done = 0
        self._total = total
        self._stage_started = now
        self._stage_start_done = 0
        self._stage_start_percent = max(self._percent, window.start)
        self._percent = self._stage_start_percent
        self._bytes_done = 0
        self._bytes_total = 0
        await self._emit(now, force=True)

    async def update(
        self,
        *,
        done: int | None = None,
        total: int | None = None,
        label: str | None = None,
        node: str | None = None,
        stage: Stage | None = None,
        bytes_done: int | None = None,
        bytes_total: int | None = None,
        force: bool = False,
    ) -> None:
        """Report stage-local progress; the tracker decides if it is worth a
        frame (throttle + monotone percent).

        `stage` may narrow the current window without opening a new one: inside
        one image group the writer step *is* the `write` stage, and the UI says
        so while the bar keeps moving through the same window.

        `force` bypasses the throttle — used for milestone frames such as the
        last one of a step, which carry the totals and must not be swallowed by
        a fast step that ran inside one throttle interval."""
        now = self._clock()
        if stage is not None:
            self._stage = stage
        if done is not None:
            self._done = max(done, self._done)
        if total is not None:
            self._total = max(total, 0)
        if label is not None:
            self._label = label
        if node is not None:
            self._node = node
        if bytes_done is not None:
            self._bytes_done = max(bytes_done, self._bytes_done)
        if bytes_total is not None:
            self._bytes_total = max(bytes_total, 0)
        target = self._stage_percent()
        self._percent = max(self._percent, target)
        await self._emit(now, force=force)

    def _stage_percent(self) -> float:
        """Where the current stage is inside its window.

        Downloads measure in bytes (an item counter would jump 0 -> 1 and tell
        the user nothing), everything else in items."""
        if self._stage is Stage.DOWNLOAD and self._bytes_total > 0:
            return self._window.percent(self._bytes_done, self._bytes_total)
        return self._window.percent(self._done, self._total)

    async def finish(self) -> None:
        """Terminal frame: the bar is full and the run has nothing left."""
        self._percent = 100.0
        await self._emit(self._clock(), force=True)
        self._stage = None

    # -- internals -------------------------------------------------------

    def _unit_rate(self, now: float) -> float | None:
        """Units per second of the current stage.

        The divisor is floored at `MIN_RATE_WINDOW_S`: a stage that is faster
        than that would otherwise never report a speed at all, and a slightly
        understated rate that stabilizes within two frames beats silence."""
        elapsed = max(now - self._stage_started, MIN_RATE_WINDOW_S)
        if self._stage is Stage.DOWNLOAD:
            moved = self._bytes_done
        else:
            moved = self._done - self._stage_start_done
        if moved <= 0:
            return None
        return moved / elapsed

    def _percent_rate(self, now: float) -> float | None:
        """Percent per second, blended: the local slope reacts to a slow node,
        the global average keeps the estimate sane right after a stage opens."""
        elapsed = now - self._started
        local_elapsed = now - self._stage_started
        local = None
        if local_elapsed >= MIN_RATE_WINDOW_S:
            moved = self._percent - self._stage_start_percent
            if moved > 0:
                local = moved / local_elapsed
        if local is not None:
            return local
        if elapsed >= MIN_RATE_WINDOW_S and self._percent > 0:
            return self._percent / elapsed
        return None

    async def _emit(self, now: float, *, force: bool = False) -> None:
        if not force and now - self._last_sent < MIN_INTERVAL_S:
            return
        self._last_sent = now
        percent = int(min(max(self._percent, 0.0), 100.0))
        # monotone on the wire even when a stage reports a rounded step back
        self._percent = max(self._percent, float(percent))
        payload: dict[str, Any] = {
            "percent": percent,
            "stage": (self._stage or Stage.PROCESS).value,
            "elapsed": round(now - self._started, 1),
        }
        if self._label is not None:
            payload["label"] = self._label
        if self._node is not None:
            payload["node"] = self._node
        if self._total > 0 or self._done > 0:
            payload["done"] = self._done
            payload["total"] = self._total
        if self._bytes_done > 0:
            payload["bytes_done"] = self._bytes_done
        if self._bytes_total > 0:
            payload["bytes_total"] = self._bytes_total
        rate = self._unit_rate(now)
        if rate is not None:
            payload["rate"] = round(rate, 1)
        percent_rate = self._percent_rate(now)
        if percent_rate is not None and percent_rate > 0 and percent < 100:
            payload["eta"] = round((100.0 - percent) / percent_rate, 1)
        await self._send("progress", payload)
