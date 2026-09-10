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
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

Environment:

| Variable            | Default           | Meaning                                              |
| ------------------- | ----------------- | ---------------------------------------------------- |
| `RELINE_MODELS_DIR` | `/content/models` | where `download` preprocessors install upscale models |

One run at a time across all connections (GPU-bound work): a second `start`
answers `error {"worker busy"}`.

## Path bases (optional)

A `start` may carry two optional bases, and `ls` one:

| Field         | Sent by | Meaning                                                                 |
| ------------- | ------- | ----------------------------------------------------------------------- |
| `root`        | `start`, `ls` | base for every node path, the configs folder and a relative `models`; remembered per connection |
| `models`      | `start` | folder downloads land in and installed models are searched in           |
| `configs`     | `start` | folder behind `config_list/read/delete`                                 |

Rules, in one place (`pipeline.resolve_path`, `session.py`):

- absent → paths are used exactly as written, which keeps every existing config
  (absolute paths) working;
- absolute, or already under `root` → left untouched, so re-running a resolved
  config never doubles the base;
- otherwise → `join(root, path)`, so a portable config can say `"path": "src"`
  and run from any folder.

`models` is the second base: `RELINE_MODELS_DIR` is only the server-wide
default, a run may point it anywhere (`models: "weights"` under
`root: "/data"` means `/data/weights`). Nothing about paths is mandatory.

An upscale node is resolved the same way, with one distinction: the wire format
has no `is_own_model` flag, so `pipeline.looks_like_model_path` decides. A bare
name (`4x_fake`) is a download request and waits for its `download`
preprocessor; a path or a file name with a model extension is resolved against
`root` (an absolute mount path from an old config is left as it is).

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
* **Downloads report bytes and speed** through a streaming copy, so a 300 MB
  model shows real progress instead of a frozen stage.

## Upstream patch

`patches.py` replaces `ImageIterator.__next__`: upstream does not advance the
cursor when a file fails to decode, so `next()` returns `None` forever and the
loop spins on one broken image. The patch is applied once by `server.create_app()`
and covered by the tests; it can go away when upstream fixes it.
