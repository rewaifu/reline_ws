"""Runtime fixes for upstream `reline` bugs.

`ImageIterator.__next__` forgets to advance `current` when a file fails to
decode: it returns None, the caller skips the image, and the next call decodes
the very same broken file again — `next()` then returns None forever and the
processing loop spins without ever reaching the end of the folder. The
replacement advances the cursor *before* returning the miss.

`apply()` is idempotent and called once from `server.create_app()`. Keep the
patch here, next to its explanation and its test, so it can be dropped the day
upstream fixes it.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pepeline import ImgFormat, read
from reline.nodes.folder_reader.node import ImageIterator
from reline.static import ImageFile

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger("uvicorn.error")

_PATCHED_MARKER = "__reline_ws_patched__"


def _next_skipping_broken(self: ImageIterator) -> ImageFile | None:
    if self.current >= self.end:
        raise StopIteration
    file_path = self.image_paths[self.current]
    commonprefix = os.path.commonprefix([self.dir_path, file_path])
    dirpath = os.path.dirname(os.path.relpath(file_path, commonprefix))
    basename, _ = os.path.splitext(os.path.basename(file_path))
    self.current += 1
    try:
        data = read(file_path, self.mode, ImgFormat.F32)
    except Exception as exc:
        logger.warning("image %s not decoded due to error: %s", basename, exc)
        return None
    return ImageFile(data, basename, dirpath)


def apply() -> bool:
    """Install the fix once; returns True when it was actually applied."""
    if getattr(ImageIterator.__next__, _PATCHED_MARKER, False):
        return False
    setattr(_next_skipping_broken, _PATCHED_MARKER, True)
    ImageIterator.__next__ = _next_skipping_broken
    return True


def iter_images(iterator: ImageIterator) -> "Iterator[ImageFile | None]":
    """Typed view over the iterator for callers that need the item type."""
    return iter(iterator)
