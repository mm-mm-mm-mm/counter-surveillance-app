import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from cs_app.api.routes import router, set_ready
from cs_app.api.websocket import websocket_handler
from cs_app.db.session import drop_and_recreate_db
from cs_app.config import FRONTEND_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database...")
    await drop_and_recreate_db()

    logger.info("Pre-warming ML models (this may take a minute on first run)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _preload_models)

    set_ready(True)
    logger.info("Ready. Open http://127.0.0.1:8000 in your browser.")
    yield


def _preload_models():
    from cs_app.pipeline.detector import VehicleDetector
    from cs_app.pipeline.anpr import ANPRProcessor
    app.state.detector = VehicleDetector()
    app.state.anpr = ANPRProcessor()
    logger.info("ML models loaded.")


app = FastAPI(title="cs_app", lifespan=lifespan)

app.include_router(router)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket_handler(websocket)


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
