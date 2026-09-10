"""Model installation for the `download` preprocessor.

`download` resolves an upscale model into `MODELS_DIR`:

* already installed (exact name, then stem match) -> no download, and the
  upscale node is simply repointed at the installed file;
* url points at a model file (`.pth`/`.pt`/`.safetensors`) -> moved there as-is;
* url points at an archive -> extracted, the model file with the matching name
  is found inside and moved to `MODELS_DIR`.

Every upscale node whose `options.model` equals the download name is rewritten
to the installed absolute path, so the pipeline never sees a bare name.

Progress: the streaming copy reports bytes, so the UI can show a real download
bar with speed instead of a frozen stage.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .archives import extract

logger = logging.getLogger("uvicorn.error")

MODEL_SUFFIXES = {".pth", ".pt", ".safetensors"}
#: 1 MiB: big enough for a fast link, small enough to keep `on_bytes` lively
CHUNK_BYTES = 1024 * 1024

#: called as (bytes_done, bytes_total | 0) while the file is on the wire
ByteCallback = Callable[[int, int], None]


def find_installed(models_dir: str, name: str) -> str | None:
    """Exact file name first, then a stem match (`4x_a` -> `4x_a.pth`)."""
    if not os.path.isdir(models_dir):
        return None
    stem = Path(name).stem
    fallback: str | None = None
    for dirpath, _dirnames, filenames in os.walk(models_dir):
        for filename in filenames:
            if Path(filename).suffix.lower() not in MODEL_SUFFIXES:
                continue
            if filename == name:
                return os.path.join(dirpath, filename)
            if fallback is None and Path(filename).stem == stem:
                fallback = os.path.join(dirpath, filename)
    return fallback


def find_model_in(root_dir: str, name: str) -> str | None:
    """Inside an extracted archive: the model file whose name (or stem)
    matches the requested model name."""
    stem = Path(name).stem
    fallback: str | None = None
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if filename == name:
                return full
            if (
                fallback is None
                and Path(filename).stem == stem
                and Path(filename).suffix.lower() in MODEL_SUFFIXES
            ):
                fallback = full
    return fallback


def download(url: str, dest: str, on_bytes: ByteCallback | None = None) -> int:
    """Stream `url` into `dest`, reporting bytes; returns the file size."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    written = 0
    with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        if on_bytes is not None:
            on_bytes(0, total)
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if on_bytes is not None:
                on_bytes(written, total)
    return written


def install_model(
    models_dir: str,
    name: str,
    url: str | None,
    on_bytes: ByteCallback | None = None,
) -> str:
    """Make sure `name` exists in `models_dir`; returns the installed path."""
    installed = find_installed(models_dir, name)
    if installed is not None:
        logger.info("download: %s already installed at %s, skipping", name, installed)
        return installed
    if not url:
        raise ValueError(f"модель {name!r} не найдена в {models_dir} и у download-ноды нет ссылки")
    os.makedirs(models_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reline_dl_") as tmp:
        dest = os.path.join(tmp, Path(url).name or name)
        download(url, dest, on_bytes)
        if Path(dest).suffix.lower() in MODEL_SUFFIXES:
            final = os.path.join(models_dir, os.path.basename(dest))
            shutil.move(dest, final)
            return final
        # archive (or unknown payload): extract and hunt for the model file
        outdir = os.path.join(tmp, "extracted")
        extract(dest, outdir)
        found = find_model_in(outdir, name)
        if found is None:
            raise ValueError(f"модель {name!r} не найдена внутри {url}")
        final = os.path.join(models_dir, os.path.basename(found))
        shutil.move(found, final)
        logger.info("download: %s extracted from %s -> %s", name, url, final)
        return final


def rewrite_upscale_models(nodes: list[dict], name: str, path: str) -> int:
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


def normalize_options(options: Any) -> dict[str, Any]:
    return options if isinstance(options, dict) else {}
