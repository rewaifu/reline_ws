"""Run registry: runs that outlive the connection that started them.

One run at a time (GPU-bound, `BusyGate`). The old contract cancelled the job
when its socket died; the new one keeps it running detached, so a client that
lost the network (sleeping tab, flapping tunnel, restarted proxy) can
`attach` by `run_id` and keep watching instead of repeating `start` — a
repeat is not a resume, it is a second job the gate would refuse anyway.

A finished run is retained for `RETAIN_S`, so a client that dropped at 99 %
can still ask `status` and read its `done`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .gate import BusyGate
from .protocol import event

if TYPE_CHECKING:
    from .session import Connection

#: how long a finished run answers `status`/`attach` after its `done`
RETAIN_S = 300.0


class BusyError(Exception):
    """A run is already in flight; `run_id` names it (None when unknown)."""

    def __init__(self, run_id: str | None) -> None:
        super().__init__("worker busy")
        self.run_id = run_id


@dataclass
class RunRecord:
    """One run: the work, its latest numbers, and who watches it."""

    run_id: str
    request_id: int
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    status: str = "running"  # running | done | failed | cancelled
    progress: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    finished_at: float | None = None
    #: key `id(conn)` -> the connection watching this run
    holders: dict[int, "Connection"] = field(default_factory=dict)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class RunRegistry:
    """The server-wide runs: at most one active, plus the last finished one."""

    def __init__(self, gate: BusyGate) -> None:
        self._gate = gate
        self._active: RunRecord | None = None
        self._last: RunRecord | None = None

    # -- lookup ----------------------------------------------------------

    def current(self) -> RunRecord | None:
        """The run in flight, if any."""
        return self._active

    def retained(self) -> RunRecord | None:
        """The last finished run, while its retention window lasts."""
        self._prune()
        return self._last

    def get(self, run_id: str) -> RunRecord | None:
        """Active or retained run by id; None when unknown or expired."""
        if self._active is not None and self._active.run_id == run_id:
            return self._active
        if self._last is not None and self._last.run_id == run_id:
            self._prune()
            return self._last
        return None

    def _prune(self) -> None:
        if self._last is not None and self._last.finished_at is not None:
            if time.monotonic() - self._last.finished_at > RETAIN_S:
                self._last = None

    # -- lifecycle -------------------------------------------------------

    def start(self, request_id: int) -> RunRecord:
        """Take the gate for a new run; raises BusyError while one runs."""
        self._prune()
        if self._active is not None:
            raise BusyError(self._active.run_id)
        if not self._gate.acquire():
            raise BusyError(None)
        record = RunRecord(run_id=new_run_id(), request_id=request_id)
        self._active = record
        return record

    def abort(self, record: RunRecord) -> None:
        """Drop a run that never started (bad configs dir): no retain, no done."""
        if self._active is record:
            self._active = None
        record.holders.clear()
        self._gate.release()

    def finish(self, record: RunRecord, status: str, result: dict[str, Any]) -> None:
        """The job is over: retain the outcome, free the gate, let go of watchers."""
        record.status = status
        record.result = result
        record.finished_at = time.monotonic()
        record.task = None
        for conn in list(record.holders.values()):
            conn.follow.pop(record.run_id, None)
        record.holders.clear()
        if self._active is record:
            self._active = None
        self._last = record
        self._gate.release()

    # -- watching ---------------------------------------------------------

    def attach(self, conn: "Connection", record: RunRecord, request_id: int) -> None:
        """Subscribe a connection to a run's frames (id maps to its request)."""
        conn.follow[record.run_id] = request_id
        record.holders[id(conn)] = conn

    def detach(self, conn: "Connection", run_id: str | None = None) -> None:
        """Unsubscribe a connection from one run, or from every run it follows."""
        names = [run_id] if run_id is not None else list(conn.follow)
        for name in names:
            conn.follow.pop(name, None)
            record = self._active if self._active is not None else None
            candidates = [record, self._last]
            for candidate in candidates:
                if candidate is not None and candidate.run_id == name:
                    candidate.holders.pop(id(conn), None)

    def snapshot(self, record: RunRecord) -> dict[str, Any]:
        """What `status`/`attached` report: phase, last numbers, outcome."""
        data: dict[str, Any] = {"run_id": record.run_id, "status": record.status}
        if record.progress is not None:
            data["progress"] = record.progress
        if record.result is not None:
            data["result"] = record.result
        return data

    def note_progress(self, record: RunRecord, payload: dict[str, Any]) -> None:
        """Remember the latest numbers, so a late attacher sees where the run is."""
        record.progress = dict(payload)
        record.progress["run_id"] = record.run_id

    async def publish(self, record: RunRecord, method: str, payload: dict[str, Any] | None = None) -> None:
        """One frame to every watcher; a dead watcher just stops watching.

        A failed write detaches that connection only — the run itself is
        unaffected. Cancelling the job because one of several watchers went
        away would make `attach` useless the moment the first tab sleeps.
        """
        frame_payload = dict(payload or {})
        frame_payload["run_id"] = record.run_id
        if method == "progress":
            self.note_progress(record, frame_payload)
        for conn in list(record.holders.values()):
            reply_id = conn.follow.get(record.run_id, record.request_id)
            if not await conn.send(event(method, reply_id, frame_payload)):
                self.detach(conn, record.run_id)
