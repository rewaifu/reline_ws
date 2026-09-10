"""reline_ws — WebSocket runner for reline pipelines.

Serves one endpoint, `/run`, speaking the MessagePack protocol documented in
the frontend repo (`WS_API.md`). Import `reline_ws.server` (or run the root
`app.py`) to start it; nothing here is imported by the tests that only need
frames and progress arithmetic.
"""
