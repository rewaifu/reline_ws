"""Run every test file: `uv run python tests/run.py` (or plain python)."""

from __future__ import annotations

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "test_protocol",
    "test_progress",
    "test_pipeline",
    "test_preprocess",
    "test_session",
    "test_e2e_ws",
]


def main() -> int:
    failed: list[str] = []
    for name in MODULES:
        started = time.monotonic()
        module = importlib.import_module(name)
        try:
            code = module.main()
        except Exception as exc:  # noqa: BLE001 - a failing test must not hide the rest
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(name)
            continue
        if code != 0:
            failed.append(name)
        print(f"  ({name} took {time.monotonic() - started:.1f}s)\n", flush=True)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
