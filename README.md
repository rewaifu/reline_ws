# reline_ws

WebSocket runner for [reline](https://github.com/rewaifu/reline) image pipelines.
A client (the [reline-web](https://github.com/rewaifu/reline_web) UI) sends a
pipeline config and receives a stream of detailed progress events until the run
is done, cancelled or failed.

The wire protocol is specified in the frontend repository: **`WS_API.md`**.

## Install and run

```bash
uv venv
uv pip install -e .
.venv/bin/python app.py --root /data --models weights   # or plain uvicorn:
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

Launch parameters — where this machine reads and writes:

| Flag        | Environment          | Default           | Meaning                                                          |
| ----------- | -------------------- | ----------------- | ---------------------------------------------------------------- |
| `--root`    | `RELINE_ROOT`        | *(unset)*         | base for every relative path in a config                         |
| `--models`  | `RELINE_MODELS_DIR`  | `/content/models` | where `download` installs models and installed ones are searched |
| `--host`    | `RELINE_HOST`        | `0.0.0.0`         | interface to bind                                                |
| `--port`    | `RELINE_PORT`        | `8000`            | port to bind                                                     |
| `--no-proxy-headers` | —           | off               | ignore `X-Forwarded-*` instead of trusting a local proxy         |

`GET /health` reports them back (`"root"`, `"models"`), so which folder a
deployment is using is one curl away.

One run at a time across all connections (GPU-bound work): a second `start`
answers `error {"worker busy"}`.

## Path bases

A deployment sets its bases once, at launch; the UI only has to know the
address. A client *may* still override either one per run.

| Field     | Sent by       | Meaning                                                                    |
| --------- | ------------- | -------------------------------------------------------------------------- |
| `root`    | `start`, `ls` | base for every node path, the configs folder and a relative `models`; remembered per connection |
| `models`  | `start`       | folder downloads land in and installed models are searched in               |
| `configs` | `start`       | folder behind `config_list/read/delete`                                     |

Rules, in one place (`pipeline.resolve_path`, `session.py`):

- absent → paths are used exactly as written, which keeps every existing config
  (absolute paths) working;
- absolute, or already under `root` → left untouched, so re-running a resolved
  config never doubles the base;
- otherwise → `join(root, path)`, so a portable config can say `"path": "src"`
  and run from any folder.

`models` is the second base: a relative value (`--models weights` under
`--root /data`) means `/data/weights`, and the per-run `models` field behaves the
same way. Nothing about paths is mandatory.

An upscale node is resolved the same way, with one distinction: the wire format
has no `is_own_model` flag, so `pipeline.looks_like_model_path` decides. A bare
name (`4x_fake`) is a download request and waits for its `download`
preprocessor; a path or a file name with a model extension is resolved against
`root` (an absolute mount path from an old config is left as it is).

## Behind a proxy (the 502 hunt)

The client speaks WebSocket; a proxy in front must pass the upgrade through and
must not buffer or time out mid-run. In order:

1. **Is the origin alive?** `curl -i http://127.0.0.1:8000/health` on the
   server. `200 {"ok": true, "version": …}` means the app is up. A `502` (or a
   refused connection) for *every* path — `/health` included — cannot come from
   this app: the process is down, or the proxy dials the wrong port/host. A
   plain `GET /run` answers `404`: the route accepts only the WebSocket upgrade.
2. **Why did it die?** the uvicorn log is the only witness (a missing
   `reline`/`torch`, a `download` preprocessor with a dead URL, an OOM kill, or
   a notebook/container session that ended — that last one looks exactly like a
   502 on every path).
3. **Is it listening where the proxy dials?** `ss -ltnp | grep 8000`.
   `--host 127.0.0.1` is invisible from another container; use `0.0.0.0` and
   start with `--proxy-headers --forwarded-allow-ips='*'` when a proxy is in
   front.
4. **Does the proxy upgrade?** nginx:
   ```nginx
   location /run {
       proxy_pass http://127.0.0.1:8000;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_read_timeout 3600s;   # a run may be longer than the 60 s default
       proxy_buffering off;
   }
   ```

   Apache 2.4.47+:

   ```apache
   ProxyPass /run ws://127.0.0.1:8000/run upgrade=websocket
   ProxyPassReverse /run ws://127.0.0.1:8000/run
   ```

   Without the upgrade bits the handshake fails while `/health` keeps working —
   the two probes together say whether to look at the process or at the proxy.

The client sends `echo` every 5 s and gives up after 15 s of silence, so any
idle timeout the proxy applies is reset by traffic.

### When the tab goes away

A client that vanishes mid-run (closed tab, dropped tunnel, restarted proxy) is
an ordinary event, not a crash. The first failed write marks the connection
dead, wakes the cancel event so the pipeline stops at its next checkpoint, and
turns every later frame into a no-op. The log gets one line and nothing else:

```
INFO:     client disconnected: cancelling the run (write failed: WebSocketDisconnect)
```

No `pipeline error` traceback — a dead socket is not a broken pipeline — and no
`done` frame to a socket nobody is reading. The busy gate is released as soon as
the run unwinds, so the client that reconnects can press start again instead of
collecting `worker busy` for the rest of the abandoned batch. A *real* failure
on a live socket is still loud: traceback in the log, `done {ok: false, error}`
on the wire.

### Reading the public answer

Two commands, one on the server and one outside, place the fault:

```bash
# on the server
curl -s -o /dev/null -w 'local %{http_code}\n' http://127.0.0.1:8000/health
# from anywhere
curl -s -o /dev/null -w 'public %{http_code}\n' https://<host>/health
```

| local | public | meaning |
| --- | --- | --- |
| refused | — | uvicorn is not running; read its log (import error, OOM, dead session) |
| 200 | `404` with a short text body from the *tunnel* itself | the tunnel is no longer registered for that host — it was restarted or its session ended, so the hostname is gone. Start the tunnel again (a quick tunnel hands out a **new** hostname: update the address in the UI) |
| 200 | `502` / `504` | the tunnel is up, its origin is not: wrong port, `127.0.0.1` instead of `0.0.0.0`, or a dead process on the other side |
| 200 | `200` on `/health`, WebSocket still fails | only the upgrade path is broken — see the proxy config above |

Note that a `404` from the tunnel is not the app's `404`: FastAPI answers a
plain `GET /run` with its own `{"detail":"Not Found"}` JSON, while a dead
hostname answers a short `text/plain` line without JSON.

## Layout

```
app.py                     uvicorn entry point (thin shim over the package)
src/reline_ws/
  server.py                FastAPI app, the single `/run` endpoint
  session.py               per-connection state + method routing table
  protocol.py              envelope `{m, id, d}`: pack/unpack, payload builders
  progress.py              bar windows, stage/rate/ETA accounting, throttling
  pipeline.py              config split, node plan, the image loop
  preprocess/
    __init__.py            orchestration + progress of the preprocess section
    models.py              model install / download / archive search
    archives.py            patool extraction (recursive, safe ordering)
  handlers/
    run.py                 start / stop and the run job
    fs.py                  ls (path autocompletion)
    configs.py             config_list / config_read / config_delete
  gate.py                  the exclusive-run gate
  patches.py               fix for an upstream reline bug (see below)
tests/                     script tests, `python tests/run.py`
```

Every module owns one concern: `protocol` never touches a socket, `session`
never does filesystem work, handlers never compute percentages, and the
pipeline never knows how the bar is drawn.

## Tests

```bash
.venv/bin/python tests/run.py            # everything (no pytest needed)
.venv/bin/python tests/test_progress.py  # one file
```

`test_e2e_ws.py` starts a real uvicorn server, runs a real
`folder_reader → level → folder_writer` pipeline over a WebSocket and checks the
frames the UI consumes — including the invariant that a failed run never ends
in a successful `done`.

## Efficiency notes

Findings of the review that shaped the code:

* **One thread hop per image, not per node.** The old loop called
  `asyncio.to_thread` for every (image, node) pair; a 6-node pipeline over 1000
  images paid 5000 handoffs. A group now runs in one worker call, with the
  worker measuring each node so the progress frame can still name the node that
  actually costs time.
* **The reader stays lazy.** `len(iterator)` gives the image count without
  decoding a file, so `total` (and therefore the ETA) is known before the first
  image is processed.
* **`ls` uses one `os.scandir` pass** (entry type comes from the dirent, no
  stat per name) and runs in a worker thread, so a slow network path cannot
  stall the socket — `echo` is answered on time even during a lookup.
* **Progress frames are throttled** to 5/s with a monotone percent, which caps
  the stream at well under 1 KiB/s no matter how many images a run has.
* **Binary payloads keep their types.** The pipeline config travels as a
  MessagePack structure (it used to be a JSON string inside the frame, i.e.
  double-encoded); the config is ~30% smaller on the wire and the server no
  longer parses JSON for every start.
* **The busy flag is released in a `finally`.** It used to leak: an exception
  in the job (or a client that vanished mid-run) left the server permanently
  "busy" until a restart.
* **A failed run can no longer report success.** The old job sent `error` and
  then `done {ok: true}`, which made the UI replace the error it had just shown.
* **Disabled nodes are not executed.** The UI marks them with `meta.disabled`;
  the runner now honours it (before, a switched-off node still ran).
* **A run that cannot write is refused, not reported green.** A chain with no
  writer (or with its only writer switched off) used to read and process every
  image and then send `done {ok: true}` while nothing reached the disk — the
  report behind this fix was "I press start and nothing happens". Now the run
  fails up front with `pipeline has no writer: nothing would be written`, the
  same way for a chain with no reader and for an entirely empty config.
* **A reader folder that is not mounted is named.** Upstream `_scandir` logs
  the `OSError` and returns an empty list, so a path visible in the UI but
  missing on the runner finished with zero images and a green `done`. The
  folder is checked after the preprocessors ran (an `unarchive` legitimately
  creates its own) and the error carries the resolved path.
* **Zero images is a failure.** If every reader of a pipeline comes up empty,
  the run ends with `no images found in: <paths>` instead of a success over
  nothing.
* **Downloads report bytes and speed** through a streaming copy, so a 300 MB
  model shows real progress instead of a frozen stage.

## Upstream patch

`patches.py` replaces `ImageIterator.__next__`: upstream does not advance the
cursor when a file fails to decode, so `next()` returns `None` forever and the
loop spins on one broken image. The patch is applied once by `server.create_app()`
and covered by the tests; it can go away when upstream fixes it.
