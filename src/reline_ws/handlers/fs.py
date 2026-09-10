"""Path lookups for the `ls` method (path autocompletion in the UI).

One `os.scandir` pass per lookup: entry type comes from the directory entry
(no extra stat per name), the whole thing runs in a worker thread because a
network path can block for a while, and the reply is already sorted so the
socket never stalls behind a slow filesystem.

The request may carry its own `root` (the UI's paths base), which then serves
the rest of the connection too; without it the base from `start` applies.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from ..pipeline import resolve_path
from ..protocol import E_ERROR, E_LS, error_payload

if TYPE_CHECKING:
    from ..session import Connection

logger = logging.getLogger("uvicorn.error")


def parse_extensions(raw: object) -> set[str] | None:
    """`["pth", ".PT"]` -> `{"pth", "pt"}`; anything else -> no filter."""
    if not isinstance(raw, list):
        return None
    ext = {str(item).lower().lstrip(".") for item in raw if str(item).strip(".")}
    return ext or None


def list_directory(
    resolved: str,
    *,
    files_only: bool = False,
    ext: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Names in the parent directory of `resolved` that match its last part.

    Returns `(entries, dirs)` where `dirs` is the subset of `entries` that are
    directories. `ext` narrows files only: directories always stay, so the
    client can keep walking down to the file it wants.
    """
    directory, prefix = os.path.split(resolved)
    directory = directory or "."
    entries: list[str] = []
    dirs: list[str] = []
    try:
        with os.scandir(directory) as scan:
            for entry in scan:
                if not entry.name.startswith(prefix):
                    continue
                is_dir = entry.is_dir()
                if is_dir:
                    if files_only:
                        continue
                elif ext is not None and os.path.splitext(entry.name)[1].lstrip(".").lower() not in ext:
                    continue
                entries.append(entry.name)
                if is_dir:
                    dirs.append(entry.name)
    except OSError as exc:
        logger.debug("ls %s: %s", directory, exc)
        return [], []
    entries.sort()
    dirs.sort()
    return entries, dirs


async def handle_ls(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    query = d.get("path")
    if not isinstance(query, str) or not query:
        await conn.reply(request_id, E_ERROR, error_payload("path required"))
        return
    resolved = resolve_path(query, conn.resolve_root(d.get("root")))
    entries, dirs = await asyncio.to_thread(
        list_directory,
        resolved,
        files_only=d.get("files_only") is True,
        ext=parse_extensions(d.get("ext")),
    )
    await conn.reply(request_id, E_LS, {"entries": entries, "dirs": dirs})
