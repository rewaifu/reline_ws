from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import logging
from typing import Any

from rebind.pipeline import PipelineWs

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

_worker_lock = asyncio.Lock()


async def safe_send(ws: WebSocket, obj: Any) -> bool:
    try:
        await ws.send_json(obj)
        return True
    except (WebSocketDisconnect, RuntimeError, ConnectionResetError) as e:
        logger.debug("safe_send: client disconnected or send error: %s", e)
        return False
    except Exception as e:
        logger.exception("safe_send unexpected error: %s", e)
        return False


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    logger.debug("ws: accepted connection")

    try:
        init_data = await ws.receive_json()
    except WebSocketDisconnect:
        logger.debug("ws: client disconnected before init")
        return
    except Exception:
        await safe_send(ws, {"error": "invalid json"})
        try:
            await ws.close()
        except Exception:
            pass
        return

    try:
        pipe = PipelineWs.from_json(init_data)
    except Exception as e:
        await safe_send(ws, {"error": f"invalid pipeline config: {e}"})
        try:
            await ws.close()
        except Exception:
            pass
        return

    cancel_event = asyncio.Event()

    try:
        if _worker_lock.locked():
            ok = await safe_send(
                ws, {"status": "queued", "message": "worker busy, waiting..."}
            )
            if not ok:
                logger.debug("ws: client disconnected while queued (send failed)")
                return
        await _worker_lock.acquire()
    except WebSocketDisconnect:
        logger.debug("ws: client disconnected while waiting for lock")
        return
    except Exception as e:
        logger.exception("ws: error while acquiring lock: %s", e)
        return
    logger.debug("ws: acquired worker lock, starting pipeline")
    try:
        if not await safe_send(
            ws, {"status": "running", "message": "pipeline started"}
        ):
            logger.debug("ws: client disconnected immediately after lock acquire")
            cancel_event.set()

        pipeline_task = asyncio.create_task(pipe.process_linear_ws(ws, cancel_event))

        async def listen_for_commands() -> None:
            try:
                while True:
                    try:
                        msg = await ws.receive_json()
                    except WebSocketDisconnect:
                        logger.debug("listen_for_commands: client disconnected")
                        cancel_event.set()
                        return
                    except Exception as e:
                        logger.debug("listen_for_commands: invalid json: %s", e)
                        await safe_send(ws, {"warning": "invalid command json"})
                        continue
                    action = msg.get("action")
                    if action == "cancel":
                        logger.debug("listen_for_commands: received cancel")
                        cancel_event.set()
                        return
                    if action == "ping":
                        await safe_send(ws, {"status": "pong"})
            except asyncio.CancelledError:
                logger.debug("listen_for_commands: cancelled")
                raise
            except Exception as e:
                logger.exception("listen_for_commands unexpected error: %s", e)
                cancel_event.set()
                return

        cancel_task = asyncio.create_task(listen_for_commands())

        async def heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(20)
                    ok = await safe_send(ws, {"status": "ping"})
                    if not ok:
                        logger.debug(
                            "heartbeat: client seems disconnected, setting cancel_event"
                        )
                        cancel_event.set()
                        return
            except asyncio.CancelledError:
                logger.debug("heartbeat: cancelled")
                raise
            except Exception as e:
                logger.exception("heartbeat unexpected error: %s", e)
                cancel_event.set()
                return

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            await pipeline_task
        except asyncio.CancelledError:
            logger.debug("pipeline_task cancelled")
            raise
        except Exception as e:
            logger.exception("pipeline_task error: %s", e)
            await safe_send(ws, {"status": "error", "error": str(e)})
        finally:
            cancel_event.set()
            for t in (cancel_task, heartbeat_task):
                if not t.done():
                    t.cancel()
            try:
                await cancel_task
            except asyncio.CancelledError:
                pass
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    finally:
        try:
            if _worker_lock.locked():
                _worker_lock.release()
                logger.debug("ws: released worker lock")
        except RuntimeError:
            logger.exception("ws: failed to release worker lock")
        try:
            await ws.close()
        except Exception:
            pass

    logger.debug("ws: handler finished")
