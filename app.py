"""uvicorn entry point — `uvicorn app:app --port 8000`.

The application itself lives in the `reline_ws` package (`src/` layout):
this shim exists so the documented command keeps working.
"""

from __future__ import annotations

from reline_ws.server import app, create_app

__all__ = ["app", "create_app"]
