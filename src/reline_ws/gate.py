"""One run at a time across every connection.

The pipeline is GPU bound, so the runner is exclusive: `start` while another
connection (or an earlier run of this one) holds the gate answers
`error {"worker busy"}`. Acquire and release are synchronous and never await,
which makes the check-and-set atomic on the event loop — no lock needed.
"""

from __future__ import annotations


class BusyGate:
    def __init__(self) -> None:
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def acquire(self) -> bool:
        """True when the caller got the gate; False when a run is in flight."""
        if self._busy:
            return False
        self._busy = True
        return True

    def release(self) -> None:
        self._busy = False
