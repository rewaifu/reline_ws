"""Config presets: the `config_list` / `config_read` / `config_delete` trio.

Presets live flat inside the `configs` folder passed to `start`. A name is
accepted only when it is a plain file name: anything nested or escaping the
folder (`../`, `/`) is rejected before touching the filesystem.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from ..protocol import E_CONFIG_LIST, E_CONFIG_READ, E_ERROR, error_payload

if TYPE_CHECKING:
    from ..session import Connection

logger = logging.getLogger("uvicorn.error")


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
        return sorted(entry.name for entry in scan if entry.is_file())


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


async def handle_list(conn: "Connection", request_id: int, _d: dict[str, Any]) -> None:
    names = await asyncio.to_thread(list_names, conn.configs_dir)
    await conn.reply(request_id, E_CONFIG_LIST, {"entries": names})


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
