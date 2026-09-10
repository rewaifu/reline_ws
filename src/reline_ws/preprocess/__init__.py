"""Preprocessors: the part of a config that runs before the pipeline.

Two kinds, both real (WS_API.md, «Препроцессоры»):

* `download` — installs an upscale model into the run's model folder (the
  `models` base from `start`, `RELINE_MODELS_DIR` when absent) and repoints every
  upscale node that asked for it by name (see `models.py`);
* `unarchive` — unpacks an archive, or a folder of archives, recursively
  (see `archives.py`).

Both are blocking file work, so each item runs in a worker thread while the
event loop polls a small shared state for progress and cancellation. The
thread cannot be killed; a cancelled step therefore stops being reported and
the run ends, while the worker finishes what it was doing in the background.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from ..pipeline import resolve_path
from ..progress import ProgressTracker, Stage, Window
from .archives import count_archives, dearchive
from .models import install_model, normalize_options, rewrite_upscale_models

logger = logging.getLogger("uvicorn.error")

MODELS_DIR = os.environ.get("RELINE_MODELS_DIR") or "/content/models"
#: how often the event loop checks the worker for news
POLL_S = 0.2


@dataclass
class StepState:
    """Progress of one item: written by the worker thread, read by the loop."""

    done: int = 0
    total: int = 0
    label: str | None = None
    bytes_done: int = 0
    bytes_total: int = 0


def _stage_of(item: dict[str, Any]) -> tuple[Stage, str]:
    kind = str(item.get("type"))
    if kind == "download":
        options = normalize_options(item.get("options"))
        return Stage.DOWNLOAD, str(options.get("name") or "download")
    return Stage.UNARCHIVE, os.path.basename(str(normalize_options(item.get("options")).get("path") or "unarchive"))


def _run_download(state: StepState, models_dir: str, nodes: list[dict], options: dict[str, Any]) -> None:
    name = str(options.get("name") or "").strip()
    if not name:
        logger.warning("download: empty name, skipped")
        return
    url = options.get("url")
    state.label = name

    def on_bytes(done: int, total: int) -> None:
        # bytes only: the tracker turns them into percent and speed itself
        state.bytes_done = done
        state.bytes_total = total

    path = install_model(models_dir, name, url if isinstance(url, str) else None, on_bytes)
    rewritten = rewrite_upscale_models(nodes, name, path)
    if rewritten:
        logger.info("download: %d upscale node(s) repointed to %s", rewritten, path)


def _run_unarchive(state: StepState, target: str) -> None:
    state.total = count_archives(target)

    def on_archive(name: str) -> None:
        state.done += 1
        state.label = name

    dearchive(target, on_archive)
    # nested archives can push the total up while we unpack
    state.total = max(state.total, state.done)


async def _run_step(
    tracker: ProgressTracker,
    cancel_event: asyncio.Event,
    window: Window,
    stage: Stage,
    label: str,
    worker: Any,
) -> bool:
    """Run one blocking worker in a thread, streaming its progress."""
    state = StepState()
    await tracker.open_stage(stage, window, label=label)
    task = asyncio.create_task(asyncio.to_thread(worker, state))
    while not task.done():
        await tracker.update(
            done=state.done,
            total=state.total,
            label=state.label,
            bytes_done=state.bytes_done,
            bytes_total=state.bytes_total,
        )
        if cancel_event.is_set():
            logger.info("preprocess cancelled while %s was running", label)
            return False
        await asyncio.sleep(POLL_S)
    await task
    # closing frame of the step: forced, because a step faster than the
    # throttle interval would otherwise never report what it did
    await tracker.update(
        done=state.done or 1,
        total=state.total or max(state.done, 1),
        label=state.label,
        bytes_done=state.bytes_done,
        bytes_total=state.bytes_total,
        force=True,
    )
    return True


async def run_preprocessors(
    preprocess: list[dict[str, Any]],
    nodes: list[dict],
    root: str | None,
    tracker: ProgressTracker,
    windows: list[Window],
    cancel_event: asyncio.Event,
    models_dir: str = MODELS_DIR,
) -> bool:
    """Run the preprocess section. Returns False when cancelled.

    `nodes` is mutated in place (upscale model paths are repointed at the
    installed files), so the pipeline must be built after this returns.
    """
    for index, item in enumerate(preprocess):
        if cancel_event.is_set():
            return False
        if not isinstance(item, dict):
            continue
        stage, label = _stage_of(item)
        options = normalize_options(item.get("options"))
        window = windows[index] if index < len(windows) else Window(0.0, 0.0)
        if stage is Stage.DOWNLOAD:
            worker = lambda state, _o=options: _run_download(state, models_dir, nodes, _o)
        elif stage is Stage.UNARCHIVE:
            target = resolve_path(str(options.get("path") or ""), root)
            worker = lambda state, _t=target: _run_unarchive(state, _t)
        else:
            logger.warning("unknown preprocess type %r, skipped", item.get("type"))
            continue
        keep_going = await _run_step(tracker, cancel_event, window, stage, label, worker)
        if not keep_going:
            return False
    return True
