from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable

from reline.nodes import (
    FileReaderNode,
    FolderReaderNode,
    FileWriterNode,
    FolderWriterNode,
)
from reline.pipeline import Pipeline

READER_NODES = (FileReaderNode, FolderReaderNode)
WRITER_NODES = (FileWriterNode, FolderWriterNode)
from reline.nodes.folder_reader.node import ImageIterator
from reline.static import ImageFile
from pepeline import read, ImgFormat

# Progress split across phases: preprocessors occupy the first
# PREPROCESS_SHARE percent of the bar, the pipeline images take the rest.
PREPROCESS_SHARE = 50


def _patched_next(self) -> "ImageFile | None":
    """Upstream ImageIterator forgets to advance `current` when a file fails
    to decode, so one broken file makes `next()` return None forever and the
    processing loop spins. This copy advances before returning the miss."""
    if self.current >= self.end:
        raise StopIteration
    file_path = self.image_paths[self.current]
    commonprefix = os.path.commonprefix([self.dir_path, file_path])
    dirpath = os.path.dirname(os.path.relpath(file_path, commonprefix))
    basename, _ = os.path.splitext(os.path.basename(file_path))
    self.current += 1
    try:
        data = read(file_path, self.mode, ImgFormat.F32)
    except Exception as e:
        logging.warning(f"image {basename} not decoded due to error: {e}")
        return None
    return ImageFile(data, basename, dirpath)


ImageIterator.__next__ = _patched_next
# PREPROCESS_SHARE percent of the bar, the pipeline images take the rest.
PREPROCESS_SHARE = 50

SendEvent = Callable[[str, dict[str, Any]], Awaitable[None]]


def resolve_path(path: str, root: str | None) -> str:
    """Absolute paths are kept as-is; anything else is joined onto root."""
    if not root:
        return path
    if path == root or path.startswith(root.rstrip(os.sep) + os.sep):
        return path
    return os.path.join(root, path)


def split_config(data: Any) -> tuple[list[dict], list[dict]]:
    """Config arrives as {nodes, preprocess} or as a legacy flat list.

    Returns (nodes, preprocessors) — preprocessors run before the pipeline.
    """
    if isinstance(data, list):
        return data, []
    nodes = data.get("nodes") or []
    preprocess = data.get("preprocess") or []
    return nodes, preprocess



class PipelineWs(Pipeline):
    """Pipeline runner speaking the WS_API.md envelope protocol."""

    @classmethod
    def prepare_config(cls, data: Any, root: str | None) -> tuple[list[dict], list[dict]]:
        """Split the config into (nodes, preprocess) and resolve node paths
        against root. The pipeline itself is built later, AFTER the
        preprocessors ran — download rewrites upscale model paths in place."""
        nodes, preprocess = split_config(data)
        for item in nodes:
            options = item.get("options") if isinstance(item, dict) else None
            if isinstance(options, dict) and isinstance(options.get("path"), str):
                options["path"] = resolve_path(options["path"], root)
        return nodes, preprocess

    @classmethod
    def build(cls, nodes: list[dict]) -> "PipelineWs":
        """Construct after preprocessors ran (they may rewrite options)."""
        return cls(Pipeline.from_json(nodes).nodes)

    async def process_ws(
        self,
        send_event: SendEvent,
        cancel_event: asyncio.Event,
        has_preprocess: bool,
    ) -> bool:
        """Run reader→per-image→writer groups, emitting `progress` events.

        Returns True when finished on its own, False when cancelled.
        """
        data: list = []
        data_len = 0
        last_percent = -1

        async def send_progress(percent: int) -> None:
            nonlocal last_percent
            percent = max(percent, last_percent)
            if percent != last_percent:
                last_percent = percent
                await send_event("progress", {"percent": min(percent, 100)})

        def image_percent(img_index: int) -> int:
            base = PREPROCESS_SHARE if has_preprocess else 0
            return base + round((100 - base) * img_index / max(data_len, 1))

        index = 0
        while index < len(self.nodes):
            if cancel_event.is_set():
                return False

            node = self.nodes[index]

            if isinstance(node, READER_NODES):
                data = await asyncio.to_thread(node.single_process, data)
                data_len = len(data)

                writer_index: int | None = None
                for i, n in enumerate(self.nodes[index + 1 :], start=index + 1):
                    if isinstance(n, WRITER_NODES):
                        writer_index = i
                        break

                for img_index, img in enumerate(data):
                    if cancel_event.is_set():
                        return False
                    if img is None:
                        continue

                    await send_progress(image_percent(img_index))
                    await asyncio.sleep(0)

                    for inner_index in range(index + 1, len(self.nodes)):
                        if cancel_event.is_set():
                            return False

                        inner_node = self.nodes[inner_index]
                        img = await asyncio.to_thread(inner_node.single_process, img)

                        if isinstance(inner_node, WRITER_NODES):
                            break

                index = (writer_index + 1) if writer_index is not None else len(self.nodes)
            else:
                index += 1

        await send_progress(100)
        return True
