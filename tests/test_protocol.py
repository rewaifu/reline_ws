"""Envelope encoding, and what the MessagePack switch to structures saves."""

from __future__ import annotations

import json
import sys
from typing import Any

from _harness import Checks, finish

from reline_ws.protocol import (
    E_PROGRESS,
    FrameError,
    M_START,
    done_payload,
    error_payload,
    event,
    frame_size,
    pack,
    unpack,
)

CONFIG: dict[str, Any] = {
    "nodes": [
        {
            "type": "folder_reader",
            "options": {"path": "/data/in", "mode": "rgb", "recursive": False},
            "meta": {"name": "Читалка входных файлов", "parents": ["a1b2c3", "d4e5f6"]},
        },
        {"type": "upscale", "options": {"model": "4x_DWTP_DS_ATDl2", "dtype": "F32", "size": 4}},
        {"type": "sharp", "options": {"blur": 1.2, "sigma": 1.0, "amount": 1.5}},
        {"type": "folder_writer", "options": {"path": "/data/out", "format": "png"}},
    ],
    "preprocess": [{"type": "download", "options": {"name": "4x_DWTP_DS_ATDl2", "url": "https://bucket/models/4x.tar.xz"}}],
}


def main() -> int:
    checks = Checks("protocol")
    print("protocol:", flush=True)

    # -- envelope ------------------------------------------------------
    frame = event(M_START, 7, {"pipeline": CONFIG})
    checks.eq("a frame round-trips", unpack(pack(frame)), frame)
    checks.eq("str stays str", isinstance(unpack(pack({"m": "progress"}))["m"], str), True)
    blob = unpack(pack({"m": "x", "d": {"bin": b"\x00\x01\xff"}}))["d"]["bin"]
    checks.eq("bytes travel as msgpack bin", (type(blob), blob), (bytes, b"\x00\x01\xff"))
    checks.eq("floats survive", unpack(pack({"d": {"eta": 12.5}}))["d"]["eta"], 12.5)
    checks.eq("nothing is stringified", unpack(pack({"d": {"done": 42}}))["d"]["done"], 42)
    checks.raises("a list is not an envelope", FrameError, unpack, pack([1, 2, 3]))
    checks.raises("junk is not an envelope", FrameError, unpack, b"\xc1")

    # -- payloads ------------------------------------------------------
    ok_payload = done_payload(ok=True)
    checks.eq("success carries no error text", (ok_payload["ok"], ok_payload["cancelled"], ok_payload["error"]), (True, False, None))
    bad_payload = done_payload(ok=False, error="boom")
    checks.eq("failure carries the reason", (bad_payload["ok"], bad_payload["error"]), (False, "boom"))
    cancelled = done_payload(ok=False, cancelled=True)
    checks.eq("cancel is its own outcome", (cancelled["ok"], cancelled["cancelled"]), (False, True))
    checks.eq("errors default to non-fatal", error_payload("x")["fatal"], False)

    # -- the size budget ------------------------------------------------
    structured = frame_size(event(M_START, 1, {"pipeline": CONFIG}))
    legacy = frame_size(event(M_START, 1, {"pipeline": json.dumps(CONFIG, ensure_ascii=False)}))
    checks.true(
        f"a structured pipeline is smaller than a JSON string ({structured} vs {legacy} bytes)",
        structured < legacy,
    )
    progress_frame = frame_size(
        event(
            E_PROGRESS,
            1,
            {
                "percent": 42,
                "stage": "process",
                "label": "Gaussian Blur",
                "node": "sharp",
                "done": 42,
                "total": 120,
                "elapsed": 12.5,
                "rate": 8.4,
                "eta": 9.3,
            },
        )
    )
    checks.true(f"a full progress frame fits in 150 bytes ({progress_frame})", progress_frame <= 150)
    checks.true("a throttle of 5 frames/s costs under 1 KiB/s", progress_frame * 5 <= 1024)

    return finish(checks)


if __name__ == "__main__":
    sys.exit(main())
