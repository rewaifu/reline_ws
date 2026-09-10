"""Envelope protocol for the `/run` WebSocket (see WS_API.md).

One frame = one MessagePack map `{m, id, d}`. This module owns the vocabulary
and the (de)serialization, so no other module needs to know msgpack details.

Payloads are real MessagePack structures: nested maps/arrays/strings/ints go
over the wire natively, nothing is pre-serialized to a JSON string.
"""

from __future__ import annotations

from typing import Any

import msgpack

# client -> server
M_START = "start"
M_STOP = "stop"
M_ECHO = "echo"
M_LS = "ls"
M_CONFIG_LIST = "config_list"
M_CONFIG_READ = "config_read"
M_CONFIG_DELETE = "config_delete"

# server -> client
E_ACCEPTED = "accepted"
E_PROGRESS = "progress"
E_DONE = "done"
E_ERROR = "error"
E_OK = "ok"
E_ECHO = "echo"
E_LS = "ls"
E_CONFIG_LIST = "config_list"
E_CONFIG_READ = "config_read"


class FrameError(ValueError):
    """A frame that could not be decoded as an envelope."""


def pack(message: dict[str, Any]) -> bytes:
    """Serialize one envelope. `use_bin_type` keeps str as msgpack str (not
    the pre-1.0 raw type), matching `unpack`'s `raw=False`."""
    return msgpack.packb(message, use_bin_type=True)


def unpack(frame: bytes) -> dict[str, Any]:
    """Decode one envelope; raises FrameError on anything that is not a map."""
    try:
        message = msgpack.unpackb(frame, raw=False, strict_map_key=False)
    except Exception as exc:
        raise FrameError(str(exc) or "not a msgpack frame") from exc
    if not isinstance(message, dict):
        raise FrameError("frame must be a map")
    return message


def event(method: str, request_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Server -> client frame. `d` is omitted when there is nothing to say."""
    message: dict[str, Any] = {"m": method, "id": request_id}
    if payload is not None:
        message["d"] = payload
    return message


def error_payload(message: str, *, fatal: bool = False) -> dict[str, Any]:
    return {"message": message, "fatal": fatal}


def done_payload(
    *,
    ok: bool,
    cancelled: bool = False,
    output: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": ok, "cancelled": cancelled, "error": error}
    if output is not None:
        payload["output"] = output
    return payload


def frame_size(message: dict[str, Any]) -> int:
    """Wire cost of one frame — used by the tests and the size budget."""
    return len(pack(message))
