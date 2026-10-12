"""The model database: parsing, the disk copy, refresh rules, the `models` method."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from typing import Any

from _harness import Checks, finish

from reline_ws.gate import BusyGate
from reline_ws.modeldb import ModelCatalog, ModelDbError, parse_files
from reline_ws.protocol import E_ERROR, E_MODELS
from reline_ws.session import Connection
from test_session import FakeSocket

URL = "https://db.example/v1/files"

FILES = [
    {"filename": "4x_a.tar.xz", "name": "4x_a", "ext": "tar.xz", "size": 1, "url": "https://b.example/4x_a.tar.xz"},
    {"filename": "2x_b.tar.xz", "name": "2x_b", "ext": "tar.xz", "size": 2, "url": "https://b.example/2x_b.tar.xz"},
]


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Network:
    """A fake database: counts requests, answers or fails on demand."""

    def __init__(self, body: Any = FILES) -> None:
        self.body = body
        self.calls = 0
        self.down = False
        self.delay = 0.0

    def __call__(self, url: str) -> bytes:
        import time

        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.down:
            raise OSError("network is unreachable")
        return self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()


def make_catalog(tmp: str, network: Network, clock: Clock, *, url: str = URL) -> ModelCatalog:
    return ModelCatalog(
        url,
        os.path.join(tmp, "cache", "models-db.json"),
        fetch=network,
        clock=clock,
        ttl=600,
        min_refresh_interval=30,
    )


def check_parsing(checks: Checks) -> None:
    checks.eq("a list of entries parses", len(parse_files(json.dumps(FILES).encode())), 2)
    checks.eq("an empty list is a valid list", parse_files(b"[]"), [])
    checks.raises("not JSON is an error", ModelDbError, parse_files, b"<html>")
    checks.raises("an object is not a list", ModelDbError, parse_files, b'{"files": []}')
    checks.raises("an entry without a url is an error", ModelDbError, parse_files, b'[{"name": "a"}]')
    checks.raises("an empty name is an error", ModelDbError, parse_files, b'[{"name": "", "url": "u"}]')
    checks.raises("a non-object entry is an error", ModelDbError, parse_files, b'["a"]')


async def check_catalog(checks: Checks, tmp: str) -> None:
    clock, network = Clock(), Network()
    catalog = make_catalog(tmp, network, clock)

    checks.eq("nothing is loaded before the first answer", catalog.load_cache(), False)
    answer = await catalog.snapshot()
    checks.eq("the first request fetches", network.calls, 1)
    checks.eq("…and serves the entries as the database sent them", answer["files"], FILES)
    checks.eq("…stamped with the time of the fetch", answer["fetched_at"], clock.now)
    checks.eq("…and not stale", answer["stale"], False)
    checks.true("the answer is on disk", os.path.isfile(catalog.cache_path))
    checks.eq("no temp file is left behind", [n for n in os.listdir(os.path.dirname(catalog.cache_path)) if n.endswith(".tmp")], [])

    clock.now += 100
    await catalog.snapshot()
    checks.eq("a young copy is served without a request", network.calls, 1)

    clock.now += 600
    await catalog.snapshot()
    checks.eq("a copy past the TTL is refreshed first", network.calls, 2)

    # -- forced refresh ------------------------------------------------
    clock.now += 10
    await catalog.snapshot(refresh=True)
    checks.eq("a forced refresh is ignored while the copy is younger than the floor", network.calls, 2)
    clock.now += 40
    await catalog.snapshot(refresh=True)
    checks.eq("…and honoured once it is older", network.calls, 3)

    # -- the network goes away ----------------------------------------
    clock.now += 700
    network.down = True
    answer = await catalog.snapshot()
    checks.eq("a failed refresh still serves the old copy", answer["files"], FILES)
    checks.eq("…marked stale", answer["stale"], True)
    checks.true("…with the reason on record", "unreachable" in (catalog.last_error or ""))
    calls = network.calls
    clock.now += 5
    await catalog.snapshot()
    checks.eq("a failure is not retried on every request", network.calls, calls)
    clock.now += 40
    network.down = False
    answer = await catalog.snapshot()
    checks.eq("it is retried after the floor, and recovers", (answer["stale"], network.calls), (False, calls + 1))

    # -- a restart without a network -----------------------------------
    offline = Network()
    offline.down = True
    restarted = make_catalog(tmp, offline, clock)
    checks.eq("the disk copy is loaded at launch", restarted.load_cache(), True)
    checks.eq("…with its original timestamp", restarted.fetched_at, catalog.fetched_at)
    answer = await restarted.snapshot()
    checks.eq("…and served while the database is down (fresh enough)", (answer["files"], answer["stale"]), (FILES, False))

    other = make_catalog(tmp, Network(), clock, url="https://other.example/files")
    checks.eq("a copy of another URL is not used", other.load_cache(), False)

    with open(catalog.cache_path, "w", encoding="utf-8") as handle:
        handle.write("{broken")
    checks.eq("a corrupt copy is no copy", make_catalog(tmp, Network(), clock).load_cache(), False)


async def check_failures(checks: Checks, tmp: str) -> None:
    clock = Clock()

    down = Network()
    down.down = True
    catalog = make_catalog(os.path.join(tmp, "a"), down, clock)
    try:
        await catalog.snapshot()
    except ModelDbError as exc:
        checks.true("no copy and no network is an error naming the cause", "unreachable" in str(exc))
    else:
        raise AssertionError("expected ModelDbError")

    garbage = Network(b"<html>nope</html>")
    catalog = make_catalog(os.path.join(tmp, "b"), garbage, clock)
    try:
        await catalog.snapshot()
    except ModelDbError as exc:
        checks.true("an answer that is not the list is an error", "not JSON" in str(exc))
    else:
        raise AssertionError("expected ModelDbError")
    checks.eq("…and nothing is written to disk", os.path.exists(catalog.cache_path), False)

    off = make_catalog(os.path.join(tmp, "c"), Network(), clock, url="")
    try:
        await off.snapshot()
    except ModelDbError as exc:
        checks.true("a switched-off database with no copy says so", "switched off" in str(exc))
    else:
        raise AssertionError("expected ModelDbError")
    checks.eq("…and never asks the network", off._fetch.calls, 0)  # type: ignore[attr-defined] # noqa: SLF001

    # -- concurrent requests share one fetch ----------------------------
    slow = Network()
    slow.delay = 0.2
    catalog = make_catalog(os.path.join(tmp, "d"), slow, clock)
    answers = await asyncio.gather(*(catalog.snapshot() for _ in range(5)))
    checks.eq("five simultaneous requests make one fetch", slow.calls, 1)
    checks.eq("…and all get the list", [len(a["files"]) for a in answers], [2] * 5)


async def check_method(checks: Checks, tmp: str) -> None:
    clock, network = Clock(), Network()
    catalog = make_catalog(os.path.join(tmp, "m"), network, clock)
    socket = FakeSocket()
    conn = Connection(socket, BusyGate(), model_db=catalog)

    await conn.dispatch({"m": "models", "id": 21, "d": {}})
    checks.eq("`models` answers under its own name and id", (socket.last["m"], socket.last["id"]), (E_MODELS, 21))
    checks.eq("…with the entries", socket.last["d"]["files"], FILES)
    checks.eq("…and the freshness", (socket.last["d"]["fetched_at"], socket.last["d"]["stale"]), (clock.now, False))

    await conn.dispatch({"m": "models", "id": 22})
    checks.eq("a request without a payload works", socket.last["m"], E_MODELS)

    clock.now += 60
    await conn.dispatch({"m": "models", "id": 23, "d": {"refresh": True}})
    checks.eq("`refresh: true` refetches a copy older than the floor", network.calls, 2)
    await conn.dispatch({"m": "models", "id": 24, "d": {"refresh": "yes"}})
    checks.eq("…and only a real true counts", network.calls, 2)

    network.down = True
    clock.now += 700
    await conn.dispatch({"m": "models", "id": 25, "d": {}})
    checks.eq("a stale copy is still an answer", (socket.last["m"], socket.last["d"]["stale"]), (E_MODELS, True))

    empty = Connection(FakeSocket(), BusyGate(), model_db=make_catalog(os.path.join(tmp, "e"), network, clock))
    await empty.dispatch({"m": "models", "id": 26, "d": {}})
    checks.eq("no copy and no network is an error frame", empty.ws.last["m"], E_ERROR)  # type: ignore[attr-defined]
    checks.true("…that says the database is unavailable", "unavailable" in empty.ws.last["d"]["message"])  # type: ignore[attr-defined]

    bare = Connection(FakeSocket(), BusyGate())
    await bare.dispatch({"m": "models", "id": 27, "d": {}})
    checks.eq("a runner without a catalog says so", bare.ws.last["m"], E_ERROR)  # type: ignore[attr-defined]


def main() -> int:
    checks = Checks("modeldb")
    print("modeldb:", flush=True)
    tmp = tempfile.mkdtemp(prefix="reline_ws_modeldb_")
    try:
        check_parsing(checks)
        asyncio.run(check_catalog(checks, os.path.join(tmp, "catalog")))
        asyncio.run(check_failures(checks, tmp))
        asyncio.run(check_method(checks, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return finish(checks)


if __name__ == "__main__":
    raise SystemExit(main())
