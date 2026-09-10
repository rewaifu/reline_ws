"""Pipeline runner: config in, images through the nodes, progress out.

Layout of a run (see WS_API.md): the config is split into `preprocess` and
`nodes`; preprocessors run first (they may rewrite node options, e.g. install a
model and repoint the upscale node at the file), then the pipeline streams
images through the node chain between a reader and the next writer.

Performance notes (why it looks like this):

* a whole image group runs in **one** worker-thread hop, not one per node — a
  6-node pipeline over 1000 images saves 5000 thread handoffs, and the image
  never bounces between threads;
* the worker measures each node, the loop reports the *hot* node, so the UI can
  name what actually costs time without a round trip per node;
* the reader's iterator is lazy, so `len()` gives the total without decoding
  a single file (and without holding the images in memory).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from reline.nodes import (
    FileReaderNode,
    FileWriterNode,
    FolderReaderNode,
    FolderWriterNode,
)
from reline.pipeline import Pipeline
from reline.static import Node

from .progress import ProgressTracker, Stage, Window

logger = logging.getLogger("uvicorn.error")

READER_NODES = (FileReaderNode, FolderReaderNode)
WRITER_NODES = (FileWriterNode, FolderWriterNode)


def label_of(node_type: str) -> str:
    """`folder_reader` -> `Folder Reader` (same wording as the UI registry)."""
    return " ".join(part.capitalize() for part in node_type.split("_") if part)


@dataclass(frozen=True)
class PlanStep:
    """One configured node: what it is, what to call it, and the node itself."""

    type: str
    label: str
    node: Node

    @property
    def is_reader(self) -> bool:
        return isinstance(self.node, READER_NODES)

    @property
    def is_writer(self) -> bool:
        return isinstance(self.node, WRITER_NODES)


def resolve_path(path: str, root: str | None) -> str:
    """Absolute paths are kept as-is; anything else is joined onto root."""
    if not root:
        return path
    if path == root or path.startswith(root.rstrip(os.sep) + os.sep):
        return path
    return os.path.join(root, path)


def reader_path(step: "PlanStep") -> str | None:
    """Folder a reader walks, when it has one (file readers have a file)."""
    options = getattr(step.node, "options", None)
    path = getattr(options, "path", None)
    return path if isinstance(path, str) and path else None


def check_steps(steps: list["PlanStep"]) -> None:
    """Refuse a chain that cannot write, before it reads a single file.

    A run whose writer is missing — or switched off in the UI, which drops it
    the same way — reads and processes images and then reports success while
    nothing lands on disk: indistinguishable from "I pressed start and nothing
    happened". A reader folder that is not there is worse: upstream `_scandir`
    swallows the error and returns an empty list, so the run finishes green
    over zero images. Both shapes become `done {ok: false, error}` here, which
    the run log shows; a preprocess-only run has no steps and is left alone.
    """
    if not steps:
        return
    if not any(step.is_reader for step in steps):
        raise ValueError("pipeline has no reader: nothing would be read")
    if not any(step.is_writer for step in steps):
        raise ValueError("pipeline has no writer: nothing would be written")
    for step in steps:
        if not isinstance(step.node, FolderReaderNode):
            continue
        path = reader_path(step)
        # checked after the preprocessors ran: `unarchive` creates the folder
        # it unpacks into, and that folder is a normal reader path
        if path is not None and not os.path.isdir(path):
            raise ValueError(f"reader folder does not exist: {path}")


#: what an installed model file looks like
MODEL_SUFFIXES = (".pth", ".pt", ".safetensors")


def looks_like_model_path(value: str) -> bool:
    """Does an upscale node's `model` name a file, or a download request?

    The wire format has no `is_own_model` flag: a bare name (`4x_fake`) asks
    for the `download` preprocessor to install it, while a mount path from an
    old config or a path typed by hand is a file the runner has to resolve
    itself (the UI says exactly that when it serializes an own model).
    """
    if not value or "://" in value:
        return False
    return (
        value.startswith(os.sep)
        or os.sep in value
        or os.path.splitext(value)[1].lower() in MODEL_SUFFIXES
    )


def split_config(data: Any) -> tuple[list[dict], list[dict]]:
    """Config arrives as `{nodes, preprocess}` or as a legacy flat list."""
    if isinstance(data, list):
        return data, []
    if not isinstance(data, dict):
        raise ValueError("pipeline must be a map or a list")
    nodes = data.get("nodes") or []
    preprocess = data.get("preprocess") or []
    if not isinstance(nodes, list) or not isinstance(preprocess, list):
        raise ValueError("nodes and preprocess must be lists")
    return nodes, preprocess


def _is_disabled(item: Any) -> bool:
    """The UI marks a node it switched off with `meta.disabled`; a run must not
    execute it (this is what the enable switch in the node list means)."""
    if not isinstance(item, dict):
        return True
    meta = item.get("meta")
    return isinstance(meta, dict) and meta.get("disabled") is True


class PipelineWs:
    """A built pipeline plus the step plan the runner reports progress from."""

    def __init__(self, steps: list[PlanStep]) -> None:
        self.steps = steps

    @classmethod
    def prepare_config(cls, data: Any, root: str | None) -> tuple[list[dict], list[dict]]:
        """Split the config and resolve node paths against root. The pipeline
        itself is built later, AFTER the preprocessors ran — they rewrite
        options (upscale model paths) in place."""
        nodes, preprocess = split_config(data)
        nodes = [item for item in nodes if not _is_disabled(item)]
        preprocess = [item for item in preprocess if not _is_disabled(item)]
        for item in nodes:
            options = item.get("options") if isinstance(item, dict) else None
            if not isinstance(options, dict):
                continue
            if isinstance(options.get("path"), str):
                options["path"] = resolve_path(options["path"], root)
            # an upscale pointed at a file (mount path from an old config, or a
            # path typed by hand) is resolved here; a bare model name is left
            # alone so the download preprocessor can install it
            model = options.get("model")
            if isinstance(model, str) and looks_like_model_path(model):
                options["model"] = resolve_path(model, root)
        return nodes, preprocess

    @classmethod
    def build(cls, nodes: list[dict]) -> PipelineWs:
        """Construct after the preprocessors ran (they may rewrite options)."""
        built = Pipeline.from_json(nodes).nodes
        steps = [
            PlanStep(type=str(item.get("type")), label=label_of(str(item.get("type"))), node=node)
            for item, node in zip(nodes, built)
        ]
        check_steps(steps)
        return cls(steps)

    async def run(
        self,
        tracker: ProgressTracker,
        cancel_event: asyncio.Event,
        read_window: Window,
        image_window: Window,
    ) -> bool:
        """Stream every reader group, emitting `progress`. Returns False when
        the run was cancelled."""
        index = 0
        total_steps = len(self.steps)
        empty: list[str] = []
        while index < total_steps:
            if cancel_event.is_set():
                return False
            step = self.steps[index]

            if not step.is_reader:
                index += 1
                continue

            await tracker.open_stage(Stage.READ, read_window, label=step.label, node=step.type)
            data = await asyncio.to_thread(step.node.single_process, None)
            total = len(data)
            if total == 0:
                empty.append(reader_path(step) or step.label)

            end = total_steps
            for position in range(index + 1, total_steps):
                if self.steps[position].is_writer:
                    end = position + 1
                    break
            chain = self.steps[index + 1 : end]

            if chain:
                finished = await self._run_chain_group(
                    tracker, cancel_event, image_window, chain, data, total
                )
                if not finished:
                    return False

            index = end

        # every reader empty: the run "succeeded" over nothing, which reads as
        # a broken tool — say so instead of writing a green done. A stop that
        # landed meanwhile stays a stop.
        readers = sum(1 for step in self.steps if step.is_reader)
        if not cancel_event.is_set() and empty and len(empty) == readers:
            raise ValueError(f"no images found in: {', '.join(empty)}")
        return True

    async def _run_chain_group(
        self,
        tracker: ProgressTracker,
        cancel_event: asyncio.Event,
        window: Window,
        chain: list[PlanStep],
        data: Any,
        total: int,
    ) -> bool:
        """Feed one reader's images through one node chain."""
        timings = [0.0] * len(chain)
        first = chain[0]
        await tracker.open_stage(
            Stage.PROCESS, window, label=first.label, node=first.type, total=total
        )
        for position, image in enumerate(data, start=1):
            if cancel_event.is_set():
                return False
            if image is not None:
                hot = await asyncio.to_thread(_run_chain, chain, image, timings, cancel_event)
            else:
                hot = None
                # skipped images do no work: yield so echo/stop still answer
                await asyncio.sleep(0)
            if hot is not None:
                await tracker.update(
                    done=position,
                    total=total,
                    label=hot.label,
                    node=hot.type,
                    stage=Stage.WRITE if hot.is_writer else Stage.PROCESS,
                )
            else:
                await tracker.update(done=position, total=total)
        return True


def _run_chain(
    chain: list[PlanStep],
    image: Any,
    timings: list[float],
    cancel_event: asyncio.Event,
) -> PlanStep | None:
    """Run one image through the chain in the worker thread; returns the node
    that spent the most time on it."""
    for position in range(len(timings)):
        timings[position] = 0.0
    for position, step in enumerate(chain):
        if cancel_event.is_set():
            break
        started = time.perf_counter()
        image = step.node.single_process(image)
        timings[position] = time.perf_counter() - started
        if step.is_writer:
            break
    if not any(timings):
        return None
    return chain[max(range(len(timings)), key=timings.__getitem__)]
