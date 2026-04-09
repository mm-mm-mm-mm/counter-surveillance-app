from fastapi import APIRouter
from fastapi.responses import JSONResponse
from cs_app.config import VIDEO_INPUT_DIR

router = APIRouter(prefix="/api")

_status = {"ready": False}


def set_ready(value: bool):
    _status["ready"] = value


@router.get("/videos")
async def list_videos():
    extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    files = [
        f.name
        for f in sorted(VIDEO_INPUT_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return JSONResponse({"videos": files})


@router.get("/status")
async def get_status():
    return JSONResponse(_status)
