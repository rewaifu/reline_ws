"""Entry point: `uvicorn app:app` or `python app.py [flags]`.

The application itself lives in the `reline_ws` package (`src/` layout). This
module is where a deployment is configured: the path bases a machine reads and
writes (`--root`, `--models`) are launch parameters, not something a UI has to
know, and an old config full of absolute paths keeps working when they are
absent.

    uvicorn app:app --host 0.0.0.0 --port 8000          # env only
    python app.py --root /data --models weights         # flags
    RELINE_ROOT=/data python app.py                     # same thing
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

# A src/ layout is not importable by path alone; running this file from the
# repo root should work whether or not the package was installed first.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import uvicorn  # noqa: E402

from reline_ws.server import app, create_app  # noqa: E402

__all__ = ["app", "create_app", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reline_ws",
        description="WebSocket runner for reline pipelines (WS_API.md)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("RELINE_HOST", "0.0.0.0"),
        help="interface to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RELINE_PORT", "8000")),
        help="port to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("RELINE_ROOT"),
        help="base for every relative path in a config; absent means paths are "
        "used exactly as written (default: RELINE_ROOT)",
    )
    parser.add_argument(
        "--models",
        default=os.environ.get("RELINE_MODELS_DIR"),
        help="folder `download` installs models into and existing ones are "
        "looked up in; a relative value goes under --root (default: "
        "RELINE_MODELS_DIR, then /content/models)",
    )
    parser.add_argument(
        "--no-proxy-headers",
        action="store_true",
        help="ignore X-Forwarded-* (the default trusts a local reverse proxy)",
    )
    args = parser.parse_args(argv)

    uvicorn.run(
        create_app(root=args.root, models_dir=args.models),
        host=args.host,
        port=args.port,
        proxy_headers=not args.no_proxy_headers,
        forwarded_allow_ips="*",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
