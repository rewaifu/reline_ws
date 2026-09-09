"""Real preprocessors for the WS pipeline: `download` and `unarchive`.

`download` — resolves an upscale model into MODELS_DIR:
  * already installed (stem match) -> no download, just link the path;
  * url points at a model file (.pth/.pt/.safetensors) -> moved there as-is;
  * url points at an archive -> extracted, the model file with the matching
    name is found inside and moved to MODELS_DIR.
  Every upscale node whose `options.model` equals the download name is
  rewritten to the installed absolute path, so the pipeline never sees a
  bare mdb name.

`unarchive` — extracts `options.path` (archive file, or a folder containing
archives) with patool, recursing into results; each archive is removed only
after its content (and any nested archives) has been fully extracted.

MODELS_DIR comes from the RELINE_MODELS_DIR env var at startup
(default /content/models).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable

import patoolib

from rebind.pipeline import resolve_path

logger = logging.getLogger("uvicorn.error")

MODELS_DIR = os.environ.get("RELINE_MODELS_DIR") or "/content/models"
MODEL_SUFFIXES = {".pth", ".pt", ".safetensors"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".7z", ".rar", ".gz", ".bz2", ".xz", ".zst", ".lz4", ".lzma"}

SendProgress = Callable[[int], Awaitable[None]]


def _is_archive(path: str) -> bool:
    try:
        return os.path.isfile(path) and patoolib.is_archive(path)
    except Exception:
        return False


def _strip_archive_suffixes(name: str) -> str:
    """`4x_a.tar.xz` -> `4x_a` (patool names multi-suffix archives too)."""
    stem = name
    while Path(stem).suffix.lower() in ARCHIVE_SUFFIXES:
        stem = Path(stem).stem
    return stem


def _extract(archive: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    patoolib.extract_archive(archive, outdir=outdir, verbosity=-1)


def _dearchive_folder(folder: str) -> None:
    for entry in sorted(os.listdir(folder)):
        obj = os.path.join(folder, entry)
        if os.path.isdir(obj):
            _dearchive_folder(obj)
        elif _is_archive(obj):
            base = _strip_archive_suffixes(entry)
            if not base:
                logger.warning("unarchive: cannot derive a folder name for %s, skipped", obj)
                continue
            outdir = os.path.join(folder, base)
            _extract(obj, outdir)
            _dearchive_folder(outdir)
            os.remove(obj)


def _dearchive(target: str) -> None:
    if os.path.isdir(target):
        _dearchive_folder(target)
        return
    parent = os.path.dirname(target) or "."
    base = _strip_archive_suffixes(os.path.basename(target))
    if not base:
        raise ValueError(f"unarchive: cannot derive a folder name from {target!r}")
    outdir = os.path.join(parent, base)
    _extract(target, outdir)
    _dearchive_folder(outdir)
    os.remove(target)


def _is_model_file(path: str) -> bool:
    return os.path.isfile(path) and Path(path).suffix.lower() in MODEL_SUFFIXES


def _find_installed(models_dir: str, name: str) -> str | None:
    """Exact file name first, then a stem match (4x_a -> 4x_a.pth)."""
    stem = Path(name).stem
    for dirpath, _dirnames, filenames in os.walk(models_dir):
        for fn in filenames:
            if fn == name:
                return os.path.join(dirpath, fn)
        for fn in filenames:
            if Path(fn).stem == stem and Path(fn).suffix.lower() in MODEL_SUFFIXES:
                return os.path.join(dirpath, fn)
    return None


def _search_model(root_dir: str, name: str) -> str | None:
    """Inside an extracted archive: the model file whose name (or stem)
    matches the requested model name."""
    stem = Path(name).stem
    exact: str | None = None
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if fn == name:
                return full
            if exact is None and Path(fn).stem == stem and Path(fn).suffix.lower() in MODEL_SUFFIXES:
                exact = full
    return exact


def _download(url: str, dest: str) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=120) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)

def _ensure_model(models_dir: str, name: str, url: str | None) -> str:
    installed = _find_installed(models_dir, name)
    if installed is not None:
        logger.info("download: %s already installed at %s, skipping", name, installed)
        return installed
    if not url:
        raise ValueError(f"модель {name!r} не найдена в {models_dir} и у download-ноды нет ссылки")
    os.makedirs(models_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reline_dl_") as tmp:
        dest = os.path.join(tmp, Path(url).name or name)
        _download(url, dest)
        if Path(dest).suffix.lower() in MODEL_SUFFIXES:
            final = os.path.join(models_dir, os.path.basename(dest))
            shutil.move(dest, final)
            return final
        # archive (or unknown payload): extract and hunt for the model file
        outdir = os.path.join(tmp, "extracted")
        _extract(dest, outdir)
        found = _search_model(outdir, name)
        if found is None:
            raise ValueError(f"модель {name!r} не найдена внутри {url}")
        final = os.path.join(models_dir, os.path.basename(found))
        shutil.move(found, final)
        logger.info("download: %s extracted from %s -> %s", name, url, final)
        return final


def _rewrite_upscale_models(nodes: list[dict], name: str, path: str) -> int:
    """Point every upscale node asking for `name` at the installed file."""
    count = 0
    for item in nodes:
        if not isinstance(item, dict):
            continue
        options = item.get("options")
        if isinstance(options, dict) and item.get("type") == "upscale" and options.get("model") == name:
            options["model"] = path
            count += 1
    return count


async def run_preprocessors(
    preprocess: list[dict],
    nodes: list[dict],
    root: str | None,
    send_progress: SendProgress,
    cancel_event: asyncio.Event,
    models_dir: str = MODELS_DIR,
) -> bool:
    """Run the preprocess section. Returns False when cancelled.

    Heavy file work (network, extraction) runs in threads; the preprocess
    list and `nodes` are mutated in place (upscale model paths).
    """
    total = len(preprocess)
    for index, item in enumerate(preprocess):
        if cancel_event.is_set():
            return False
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        options = item.get("options")
        options = options if isinstance(options, dict) else {}
        if item_type == "download":
            name = str(options.get("name") or "").strip()
            url = options.get("url")
            if not name:
                logger.warning("download: empty name, skipped")
            else:
                path = await asyncio.to_thread(_ensure_model, models_dir, name, url if isinstance(url, str) else None)
                rewritten = _rewrite_upscale_models(nodes, name, path)
                if rewritten:
                    logger.info("download: %d upscale node(s) repointed to %s", rewritten, path)
        elif item_type == "unarchive":
            target = resolve_path(str(options.get("path") or ""), root)
            await asyncio.to_thread(_dearchive, target)
        else:
            logger.warning("unknown preprocess type %r, skipped", item_type)
        await send_progress(round(50 * (index + 1) / max(total, 1)))
    return True
