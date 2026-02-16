from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

from rebind.pipeline import PipelineWs

app = FastAPI()

_worker_lock = asyncio.Lock()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    if _worker_lock.locked():
        await ws.send_json({"error": "worker busy"})
        await ws.close()
        return

    try:
        init_data = await ws.receive_json()
    except Exception:
        await ws.send_json({"error": "invalid json"})
        await ws.close()
        return

    try:
        pipe = PipelineWs.from_json(init_data)
    except Exception as e:
        await ws.send_json({"error": f"invalid pipeline config: {e}"})
        await ws.close()
        return

    cancel_event = asyncio.Event()

    async with _worker_lock:
        pipeline_task = asyncio.create_task(pipe.process_linear_ws(ws, cancel_event))

        async def listen_for_commands() -> None:
            try:
                while True:
                    msg = await ws.receive_json()
                    if msg.get("action") == "cancel":
                        cancel_event.set()
                        return
            except WebSocketDisconnect:
                cancel_event.set()

        cancel_task = asyncio.create_task(listen_for_commands())

        try:
            await pipeline_task
        except Exception as e:
            await ws.send_json({"error": str(e)})
        finally:
            cancel_event.set()
            cancel_task.cancel()
            try:
                await cancel_task
            except asyncio.CancelledError:
                pass
