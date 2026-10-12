"""The model database: the list of upscale models a client may pick and download.

The list lives at a public URL (`https://mdb.yor.ovh/v1/files`). Whether that
URL is reachable depends on *where* a machine is — a Colab VM, a home server
and a browser behind a school firewall each see a different internet — so the
runner fetches it and the client asks the runner. The deployment decides the
access, not whoever opens the page.

* at launch the last good copy is read from disk (instant) and a refresh is
  started in the background;
* every good answer is written to disk atomically, so a restart without a
  network still has a list;
* a `models` request is answered from memory; a copy older than the TTL (or
  one the client asked to refresh, at most every `MIN_REFRESH_INTERVAL`) is
  refreshed first, and when that fails the old copy is served marked `stale`.

Entries are passed through as the database sends them (`name`, `url`, `ext`,
`size`, `uploaded`, …); only `name` and `url` are required and checked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import urllib.request
from typing import Any, Callable

logger = logging.getLogger("uvicorn.error")

DEFAULT_URL = "https://mdb.yor.ovh/v1/files"
#: a copy older than this is refreshed before it is served
TTL_SECONDS = 10 * 60
#: a client-requested refresh is ignored while the copy is younger than this:
#: the runner must not become a way to hammer the database
MIN_REFRESH_INTERVAL = 30.0
FETCH_TIMEOUT = 15.0
#: the real list is ~20 KB; anything near this is not the database
MAX_BYTES = 8 * 1024 * 1024
CACHE_NAME = "models-db.json"

#: blocking `url -> body`; swapped out in tests
Fetcher = Callable[[str], bytes]


class ModelDbError(RuntimeError):
    """The list could not be obtained and there is no earlier copy to serve."""


def fetch_bytes(url: str) -> bytes:
    """GET `url`, bounded in time and size. Blocking: call it from a thread."""
    request = urllib.request.Request(url, headers={"User-Agent": "reline_ws"})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ModelDbError(f"answer is larger than {MAX_BYTES} bytes")
    return body


def parse_files(body: bytes) -> list[dict[str, Any]]:
    """The database answer as a list of entries; anything else is an error.

    Every entry must carry a non-empty string `name` and `url`. An entry that
    does not is a database bug worth failing loudly for — a half-valid list
    would make the UI offer models nobody can download.
    """
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise ModelDbError(f"answer is not JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ModelDbError("answer is not a list of files")
    for index, item in enumerate(data):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("url"), str)
            or not item["url"]
        ):
            raise ModelDbError(f"entry {index} has no name or url")
    return data


def default_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "reline_ws")


class ModelCatalog:
    """The model list: in memory, on disk, and refreshed from `url`.

    `url=""` switches the network off: the catalog then only serves what the
    disk copy holds (an air-gapped machine that was seeded by hand).
    """

    def __init__(
        self,
        url: str,
        cache_path: str,
        *,
        fetch: Fetcher = fetch_bytes,
        clock: Callable[[], float] = time.time,
        ttl: float = TTL_SECONDS,
        min_refresh_interval: float = MIN_REFRESH_INTERVAL,
    ) -> None:
        self.url = url
        self.cache_path = cache_path
        self._fetch = fetch
        self._clock = clock
        self._ttl = ttl
        self._min_interval = min_refresh_interval
        self.files: list[dict[str, Any]] = []
        #: when `files` was obtained from the database (also across restarts)
        self.fetched_at: float | None = None
        #: why the last refresh failed; None after a success
        self.last_error: str | None = None
        #: when the database was last asked, answer or not: failures are not
        #: retried on every request while the network is down
        self._attempted_at: float | None = None
        self._inflight: asyncio.Future[bool] | None = None

    # -- disk ------------------------------------------------------------

    def load_cache(self) -> bool:
        """Read the last good copy from disk. A missing, unreadable or foreign
        file (another URL) is simply no copy."""
        try:
            with open(self.cache_path, encoding="utf-8") as handle:
                stored = json.load(handle)
            if stored.get("url") != self.url:
                return False
            files = parse_files(json.dumps(stored.get("files")).encode())
            fetched_at = float(stored["fetched_at"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ModelDbError):
            return False
        self.files = files
        self.fetched_at = fetched_at
        return True

    def _save(self) -> None:
        """Atomic write: a crash mid-write must never leave half a file."""
        directory = os.path.dirname(self.cache_path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".models-db-", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(
                        {"url": self.url, "fetched_at": self.fetched_at, "files": self.files},
                        handle,
                        ensure_ascii=False,
                    )
                os.replace(tmp, self.cache_path)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
        except OSError as exc:
            # no cache is a loss of resilience, not of function
            logger.warning("model db: cannot write %s: %s", self.cache_path, exc)

    # -- network ---------------------------------------------------------

    async def _download(self) -> bool:
        if not self.url:
            self.last_error = "the model database is switched off"
            return False
        self._attempted_at = self._clock()
        try:
            body = await asyncio.to_thread(self._fetch, self.url)
            files = parse_files(body)
        except (ModelDbError, OSError) as exc:
            # urllib's errors are OSErrors: URLError, HTTPError, timeouts
            self.last_error = str(exc) or type(exc).__name__
            logger.warning("model db: %s unavailable: %s", self.url, self.last_error)
            return False
        self.files = files
        self.fetched_at = self._clock()
        self.last_error = None
        await asyncio.to_thread(self._save)
        logger.info("model db: %d models from %s", len(files), self.url)
        return True

    async def refresh(self) -> bool:
        """Fetch the list now. Concurrent callers share one request. Returns
        whether the database answered; never raises for a network problem."""
        if self._inflight is None:
            self._inflight = asyncio.ensure_future(self._download())
            self._inflight.add_done_callback(self._release)
        return await asyncio.shield(self._inflight)

    def _release(self, _future: asyncio.Future[bool]) -> None:
        self._inflight = None

    # -- serving ---------------------------------------------------------

    def _due(self, force: bool) -> bool:
        now = self._clock()
        if self._attempted_at is not None and now - self._attempted_at < self._min_interval:
            return False
        if self.fetched_at is None:
            return True
        age = now - self.fetched_at
        return age > self._ttl or (force and age >= self._min_interval)

    async def snapshot(self, *, refresh: bool = False) -> dict[str, Any]:
        """The answer to a `models` request: `{files, fetched_at, stale}`.

        `stale` is true when the copy being served is older than it should be
        because the database could not be reached. Raises `ModelDbError` only
        when there is nothing at all to serve.
        """
        if self._due(refresh):
            await self.refresh()
        if self.fetched_at is None:
            raise ModelDbError(self.last_error or "no model list yet")
        return {
            "files": self.files,
            "fetched_at": self.fetched_at,
            "stale": self.last_error is not None,
        }

    def describe(self) -> dict[str, Any]:
        """What `/health` says about the list."""
        return {
            "count": len(self.files),
            "fetched_at": self.fetched_at,
            "error": self.last_error,
            "url": self.url or None,
        }
