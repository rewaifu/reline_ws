"""Config presets: `config_list` / `config_read` / `config_write` / `config_delete`.

Presets live flat inside one folder: the deployment's `RELINE_PRESETS_DIR`
(`--presets-dir`), or the `configs` folder a `start` names. A name is accepted
only when it is a plain file name: anything nested or escaping the folder
(`../`, `/`) is rejected before touching the filesystem. Without a folder
there is nothing to save to: `config_list` says `enabled: false` and the other
methods answer an error, so a client can tell "no disk here" from "no presets".
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

from ..protocol import E_CONFIG_LIST, E_CONFIG_READ, E_CONFIG_WRITE, E_ERROR, error_payload

if TYPE_CHECKING:
    from ..session import Connection

logger = logging.getLogger("uvicorn.error")

#: a preset is a few KB of JSON; anything near this is not one
MAX_CONFIG_BYTES = 2 * 1024 * 1024


def config_path(configs_dir: str | None, name: Any) -> str | None:
    """Absolute path of a preset, or None when the request is not acceptable."""
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or ".." in name:
        return None
    if configs_dir is None:
        return None
    return os.path.join(configs_dir, name)


def list_names(configs_dir: str | None) -> list[str]:
    if not configs_dir or not os.path.isdir(configs_dir):
        return []
    with os.scandir(configs_dir) as scan:
        # dotfiles are the writer's own temp files, never presets
        return sorted(
            entry.name for entry in scan if entry.is_file() and not entry.name.startswith(".")
        )


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write_text(path: str, content: str) -> None:
    """Atomic write: the folder is created on demand, the file appears whole or
    not at all (a crash mid-write must not leave half a preset)."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".preset-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        # mkstemp is private (0600); a preset is a plain file other tools may read
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


async def handle_list(conn: "Connection", request_id: int, _d: dict[str, Any]) -> None:
    names = await asyncio.to_thread(list_names, conn.configs_dir)
    await conn.reply(
        request_id,
        E_CONFIG_LIST,
        {"entries": names, "enabled": conn.configs_dir is not None},
    )


async def handle_read(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    path = config_path(conn.configs_dir, d.get("name"))
    if path is None:
        await conn.reply(request_id, E_ERROR, error_payload("name required (and configs dir set)"))
        return
    try:
        content = await asyncio.to_thread(read_text, path)
    except OSError as exc:
        await conn.reply(request_id, E_ERROR, error_payload(str(exc)))
        return
    await conn.reply(request_id, E_CONFIG_READ, {"name": d["name"], "content": content})


async def handle_write(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    path = config_path(conn.configs_dir, d.get("name"))
    content = d.get("content")
    if path is None:
        await conn.reply(request_id, E_ERROR, error_payload("name required (and presets dir set)"))
        return
    if not isinstance(content, str):
        await conn.reply(request_id, E_ERROR, error_payload("content must be a string"))
        return
    if len(content.encode("utf-8")) > MAX_CONFIG_BYTES:
        await conn.reply(request_id, E_ERROR, error_payload("content is too large"))
        return
    try:
        await asyncio.to_thread(write_text, path, content)
    except OSError as exc:
        await conn.reply(request_id, E_ERROR, error_payload(str(exc)))
        return
    await conn.reply(request_id, E_CONFIG_WRITE, {"name": d["name"]})


async def handle_delete(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    path = config_path(conn.configs_dir, d.get("name"))
    if path is None:
        await conn.reply(request_id, E_ERROR, error_payload("name required (and configs dir set)"))
        return
    try:
        await asyncio.to_thread(os.remove, path)
    except OSError as exc:
        await conn.reply(request_id, E_ERROR, error_payload(str(exc)))
        return
    await conn.reply(request_id, "ok")
