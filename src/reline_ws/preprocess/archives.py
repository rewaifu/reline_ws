"""Archive helpers for the `unarchive` preprocessor (patool based).

Extraction is recursive: an archive inside an archive is unpacked where it
landed, and an archive is removed only after its content has been fully
extracted, so an interrupted run never loses data it has not written yet.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import patoolib

logger = logging.getLogger("uvicorn.error")

ARCHIVE_SUFFIXES = {
    ".zip",
    ".tar",
    ".7z",
    ".rar",
    ".gz",
    ".bz2",
    ".xz",
    ".zst",
    ".lz4",
    ".lzma",
}

#: called with the archive that is about to be unpacked
StepCallback = Callable[[str], None]


def is_archive(path: str) -> bool:
    try:
        return os.path.isfile(path) and patoolib.is_archive(path)
    except Exception:
        return False


def strip_archive_suffixes(name: str) -> str:
    """`4x_a.tar.xz` -> `4x_a` (patool names multi-suffix archives too)."""
    stem = name
    while Path(stem).suffix.lower() in ARCHIVE_SUFFIXES:
        stem = Path(stem).stem
    return stem


def count_archives(target: str) -> int:
    """How many archives an `unarchive` step starts with — a cheap scandir
    walk, used to size the progress of that step. Nested archives are only
    discovered while unpacking, so the count grows as the step runs and the
    caller keeps reporting the latest value."""
    if os.path.isfile(target):
        return 1
    if not os.path.isdir(target):
        return 0
    return sum(
        1
        for _dirpath, _dirnames, filenames in os.walk(target)
        for name in filenames
        if Path(name).suffix.lower() in ARCHIVE_SUFFIXES
    )


def extract(archive: str, outdir: str) -> None:
    """Unpack one archive into `outdir` (created if needed)."""
    os.makedirs(outdir, exist_ok=True)
    patoolib.extract_archive(archive, outdir=outdir, verbosity=-1)


def _dearchive_folder(folder: str, on_step: StepCallback | None) -> None:
    for entry in sorted(os.listdir(folder)):
        obj = os.path.join(folder, entry)
        if os.path.isdir(obj):
            _dearchive_folder(obj, on_step)
        elif is_archive(obj):
            base = strip_archive_suffixes(entry)
            if not base:
                logger.warning("unarchive: cannot derive a folder name for %s, skipped", obj)
                continue
            outdir = os.path.join(folder, base)
            if on_step is not None:
                on_step(entry)
            extract(obj, outdir)
            _dearchive_folder(outdir, on_step)
            os.remove(obj)


def dearchive(target: str, on_step: StepCallback | None = None) -> None:
    """Unpack `target` (an archive, or a folder of archives) in place."""
    if os.path.isdir(target):
        _dearchive_folder(target, on_step)
        return
    parent = os.path.dirname(target) or "."
    base = strip_archive_suffixes(os.path.basename(target))
    if not base:
        raise ValueError(f"unarchive: cannot derive a folder name from {target!r}")
    outdir = os.path.join(parent, base)
    if on_step is not None:
        on_step(os.path.basename(target))
    extract(target, outdir)
    _dearchive_folder(outdir, on_step)
    os.remove(target)
