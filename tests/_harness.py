"""Shared harness for the script-style tests in this folder.

No pytest in this project (and none needed): each `test_*.py` exposes
`main() -> int`, prints one line per check, and `run.py` calls them all.

    uv run python tests/run.py            # everything
    uv run python tests/test_progress.py  # one file
"""

from __future__ import annotations

import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class Checks:
    """Collects check names; failing assertions abort the test file."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.passed: list[str] = []

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  PASS {name}", flush=True)

    def eq(self, name: str, got: Any, want: Any) -> None:
        assert got == want, f"{name}: got {got!r}, want {want!r}"
        self.ok(name)

    def true(self, name: str, value: Any) -> None:
        assert value, f"{name}: expected a truthy value, got {value!r}"
        self.ok(name)

    def near(self, name: str, got: float, want: float, tol: float = 1e-6) -> None:
        assert abs(got - want) <= tol, f"{name}: got {got!r}, want {want!r} ±{tol}"
        self.ok(name)

    def raises(self, name: str, exc: type[BaseException], fn: Any, *args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except exc:
            self.ok(name)
            return
        raise AssertionError(f"{name}: expected {exc.__name__}")


def finish(checks: Checks) -> int:
    print(f"{checks.title}: {len(checks.passed)} checks passed\n", flush=True)
    return 0
