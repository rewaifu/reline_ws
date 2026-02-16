# Pipeline WebSocket Server

A FastAPI WebSocket server for running image processing pipelines with real-time progress updates.

## Installation

```bash
git clone https://github.com/rewaifu/reline_ws
cd reline_ws
uv venv
uv pip install -e .
```

## Running the server

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

## How it works

1. Client connects to `ws://localhost:8000/ws`
2. Client sends a pipeline config as JSON
3. Server streams progress updates back
4. Client can cancel at any time

Only **one pipeline can run at a time**. If a client tries to connect while a pipeline is already running, it receives `{"error": "worker busy"}` and the connection closes.

---

## Protocol

### 1. Send pipeline config (immediately after connecting)

```json
[
  {
    "type": "folder_reader",
    "options": {
      "path": "input",
      "recursive": true,
      "mode": "rgb"
    }
  },
  {
    "type": "folder_writer",
    "options": {
      "path": "output",
      "format": "png"
    }
  }
]

```

### 2. Receive progress updates

```json
{ "status": "running", "progress": 3, "data_len": 20 }
```

| Field      | Type   | Description                      |
|------------|--------|----------------------------------|
| `status`   | string | Current state (see table below)  |
| `progress` | int    | Number of items processed so far |
| `data_len` | int    | Total number of items            |

#### Status values

| Status      | Meaning                                  |
|-------------|------------------------------------------|
| `running`   | Pipeline is actively processing          |
| `done`      | All items processed successfully         |
| `cancelled` | Stopped by client cancel request         |
| `error`     | Something went wrong (see `error` field) |

### 3. Cancel (optional)

Send at any time to stop processing:

```json
{ "action": "cancel" }
```

The server finishes the current item, then stops and sends a final `cancelled` message.

---

## Error responses

Errors are sent as JSON before the connection closes:

```json
{ "error": "worker busy" }
```
```json
{ "error": "invalid json" }
```
```json
{ "error": "invalid pipeline config: <details>" }
```
