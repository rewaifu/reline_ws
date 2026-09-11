"""Archive helpers for the `unarchive` preprocessor (patool based).

Extraction is recursive: an archive inside an archive is unpacked where it
landed, and an archive is removed only after its content has been fully
extracted, so an interrupted run never loses data it has not written yet.
"""

from __future__ import annotations

import logging
import os
import shutil
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

#: `on_step(name, finished)`: the archive that is being unpacked right now, and
#: whether it is through. Announcing the name at the start is what labels a long
#: extraction; counting it at the end is what keeps the bar honest — a counter
#: that ticks before `extract` returns parks the bar at 100 % for as long as the
#: unpack really takes.
StepCallback = Callable[[str, bool], None]


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


def _merge(source: str, target: str) -> None:
    """Move everything under `source` into `target`, overwriting by name."""
    for entry in os.listdir(source):
        src = os.path.join(source, entry)
        dst = os.path.join(target, entry)
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            _merge(src, dst)
        else:
            os.replace(src, dst)


def extract(archive: str, outdir: str) -> None:
    """Unpack one archive into `outdir`, replacing what is already there.

    Extracting straight into a folder that already holds the same names is not
    overwriting: the unpacker appends `_1` to every collision, so a second run
    of the same config doubled its own input (220 images became 440, then 880)
    and the reader happily processed the copies. A staging folder plus a
    name-for-name merge makes the step idempotent — a re-run, or the replayed
    `start` of a reconnected client, lands the same files in the same place.
    """
    staging = f"{outdir}.part"
    shutil.rmtree(staging, ignore_errors=True)
    patoolib.extract_archive(archive, outdir=staging, verbosity=-1)
    # the merge starts only once the archive is fully out, so `outdir` never
    # shows a half-extracted tree
    os.makedirs(outdir, exist_ok=True)
    _merge(staging, outdir)
    shutil.rmtree(staging, ignore_errors=True)


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
                on_step(entry, False)
            extract(obj, outdir)
            if on_step is not None:
                on_step(entry, True)
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
        on_step(os.path.basename(target), False)
    extract(target, outdir)
    if on_step is not None:
        on_step(os.path.basename(target), True)
    _dearchive_folder(outdir, on_step)
    os.remove(target)
