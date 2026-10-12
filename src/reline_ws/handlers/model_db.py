"""`models`: the list of downloadable upscale models.

The runner fetches the model database itself (see `modeldb.py`) so that its
reachability is a property of the machine the runner lives on, not of the
browser that happens to open the UI.

Request `d`: `{refresh?: bool}` — ask for a fresh copy instead of waiting for
the TTL (honoured at most every 30 s). Answer: `{files, fetched_at, stale}`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..modeldb import ModelDbError
from ..protocol import E_ERROR, E_MODELS, error_payload

if TYPE_CHECKING:
    from ..session import Connection


async def handle_models(conn: "Connection", request_id: int, d: dict[str, Any]) -> None:
    catalog = conn.model_db
    if catalog is None:
        await conn.reply(request_id, E_ERROR, error_payload("model database is not configured"))
        return
    try:
        answer = await catalog.snapshot(refresh=d.get("refresh") is True)
    except ModelDbError as exc:
        await conn.reply(request_id, E_ERROR, error_payload(f"model database unavailable: {exc}"))
        return
    await conn.reply(request_id, E_MODELS, answer)
