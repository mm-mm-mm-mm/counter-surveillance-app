import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect

# Populated in main.py after models are loaded
_active_session = None
_session_lock = asyncio.Lock()


async def websocket_handler(websocket: WebSocket):
    global _active_session
    await websocket.accept()
    async with _session_lock:
        if _active_session is not None:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "A session is already in progress."
            }))
            await websocket.close()
            return

        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=30)
            msg = json.loads(raw)
        except (asyncio.TimeoutError, json.JSONDecodeError):
            await websocket.close()
            return

        if msg.get("type") != "start" or not msg.get("filename"):
            await websocket.close()
            return

        filename = msg["filename"]

    # Import here to avoid circular imports at module load time
    from cs_app.session.manager import ProcessingSession
    from cs_app.config import VIDEO_INPUT_DIR

    video_path = VIDEO_INPUT_DIR / filename
    if not video_path.exists():
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"File not found: {filename}"
        }))
        await websocket.close()
        return

    session = ProcessingSession(video_path)
    _active_session = session

    try:
        await session.run(websocket)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except Exception:
            pass
    finally:
        _active_session = None
