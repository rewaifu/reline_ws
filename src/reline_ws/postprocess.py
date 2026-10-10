"""PSD postprocess: pair processed images with their sources, save layered PSDs.

Runs after the main chain, inside the same `process` phase and bar: for every
image the chain wrote, the matching source file is re-read as the background
and the written file as the foreground, then `pepeline.compose_psd` saves a
two-layer PSD into the `psd/` subfolder of the writer output.

Pairing is by file stem (name without extension):

1. images whose stem exists on both sides pair up first;
2. leftovers pair in OS-like (natural) order — `img2` before `img10` — until
   either side runs out: a stem missing on either side is skipped quietly.

The canvas comes from the source file, so the PSD keeps its size; the
processed image is fitted onto it per the node's `fit` method.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from .progress import ProgressTracker, Stage, Window

logger = logging.getLogger("uvicorn.error")

def _pepeline() -> Any:
    """pepeline with PSD compose (1.3+): imported late so the pairing logic
    stays importable (and testable) on older installs."""
    import pepeline

    try:
        compose = pepeline.compose_psd
        overlay_fit = pepeline.OverlayFit
    except AttributeError:
        from pepeline import pepeline as native

        compose = native.compose_psd
        overlay_fit = native.OverlayFit
    return compose, overlay_fit, pepeline.ImgColor, pepeline.ImgFormat, pepeline.read


#: image extensions the matcher walks, same set the folder reader accepts
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def natural_key(name: str) -> list[int | str]:
    """OS-like sort key: `img2` before `img10` (numeric runs compare as numbers)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _collect_stems(folder: str) -> dict[str, str]:
    """File stem -> absolute path for every image directly inside `folder`.

    Only files `pepeline.read` can actually decode: a broken file is skipped
    here, not mid-run — `compose_psd` would otherwise raise on it (or worse,
    the read panics out of the worker thread).
    """
    _, _, ImgColor, ImgFormat, read = _pepeline()
    found: dict[str, str] = {}
    try:
        entries = sorted(os.listdir(folder), key=natural_key)
    except OSError as exc:
        logger.warning("postprocess: cannot list %s: %s", folder, exc)
        return found
    for entry in entries:
        stem, ext = os.path.splitext(entry)
        if ext.lower() not in IMAGE_EXTENSIONS or stem in found:
            continue
        path = os.path.join(folder, entry)
        try:
            read(path, ImgColor.RGB, ImgFormat.U8)
        except Exception as exc:
            logger.warning("postprocess: %s unreadable, skipped: %s", entry, exc)
            continue
        found[stem] = path
    return found


def pair_stems(processed: dict[str, str], sources: dict[str, str]) -> list[tuple[str, str, str]]:
    """(stem, processed path, source path): matched stems first, then leftovers
    pair in natural order until either side runs out. A stem missing on either
    side is skipped quietly — pairing never fails, it just yields fewer pairs."""
    matched = sorted(set(processed) & set(sources), key=natural_key)
    pairs = [(stem, processed[stem], sources[stem]) for stem in matched]
    left_processed = sorted(set(processed) - set(sources), key=natural_key)
    left_sources = sorted(set(sources) - set(processed), key=natural_key)
    pairs.extend(
        (proc, processed[proc], sources[src])
        for proc, src in zip(left_processed, left_sources)
    )
    return pairs


FIT_NAMES = ("original", "fit", "cover", "stretch")


@dataclass
class PostState:
    """Progress of the PSD pass: written by the worker thread, read by the loop."""

    done: int = 0
    total: int = 0


def _run_postprocess(
    state: PostState, processed_dir: str, source_dir: str, fit: Any, delete_originals: bool = False
) -> None:
    compose_psd, _, ImgColor, ImgFormat, read = _pepeline()
    processed = _collect_stems(processed_dir)
    if not processed:
        return
    pairs = pair_stems(processed, _collect_stems(source_dir))
    state.total = len(pairs)
    if not pairs:
        return
    psd_dir = os.path.join(processed_dir, "psd")
    os.makedirs(psd_dir, exist_ok=True)
    for stem, foreground_path, background_path in pairs:
        try:
            background = read(background_path, ImgColor.RGB, ImgFormat.U8)
            foreground = read(foreground_path, ImgColor.RGB, ImgFormat.U8)
            compose_psd(
                background,
                foreground,
                os.path.join(psd_dir, f"{stem}.psd"),
                fit=fit,
            )
        except Exception as exc:
            logger.warning("postprocess: %s skipped: %s", stem, exc)
            continue
        # the PSD now holds the layer: the flat file it was composed from can
        # go when the node asked for it
        if delete_originals:
            try:
                os.remove(foreground_path)
            except OSError as exc:
                logger.warning("postprocess: cannot remove %s: %s", foreground_path, exc)
        state.done += 1


async def run_postprocess(
    tracker: ProgressTracker,
    cancel_event: asyncio.Event,
    window: Window,
    label: str,
    processed_dir: str,
    source_dir: str,
    fit_name: str,
    delete_originals: bool = False,
) -> bool:
    """Compose PSDs for one writer's output. Returns False when cancelled."""
    state = PostState()
    await tracker.open_stage(Stage.PROCESS, window, label=label, node="psd_compose")
    _, overlay_fit, _, _, _ = _pepeline()
    fit = getattr(overlay_fit, fit_name, overlay_fit.Original)
    task = asyncio.create_task(
        asyncio.to_thread(_run_postprocess, state, processed_dir, source_dir, fit, delete_originals)
    )
    while not task.done():
        await tracker.update(done=state.done, total=state.total)
        if cancel_event.is_set():
            return False
        await asyncio.sleep(0.2)
    await task
    await tracker.update(done=max(state.done, 0), total=max(state.total, 0), force=True)
    return True
