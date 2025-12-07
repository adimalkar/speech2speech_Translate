import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from src.pipeline.orchestrator import LiveTranslator
from src.utils.logger import logger


app = FastAPI(title="S2ST Control Panel", version="1.0.0")

# Allow local browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals for pipeline control and event fan-out
pipeline: LiveTranslator | None = None
clients: set[WebSocket] = set()
event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
pipeline_lock = asyncio.Lock()
event_loop = None


def _event_sink(event: Dict[str, Any]):
    """Called from pipeline threads; forwards to asyncio queue."""
    global event_loop
    if not event_loop:
        return
    try:
        asyncio.run_coroutine_threadsafe(event_queue.put(event), event_loop)
    except Exception as e:
        logger.debug(f"Event sink enqueue error: {e}")


async def _dispatch_events():
    """Broadcast events to all connected websockets."""
    while True:
        event = await event_queue.get()
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


@app.on_event("startup")
async def _startup():
    global event_loop
    event_loop = asyncio.get_running_loop()
    asyncio.create_task(_dispatch_events())
    logger.info("Control server started.")


@app.post("/start")
async def start_pipeline(cfg: Dict[str, Any]):
    """Start the pipeline with given source/target languages."""
    global pipeline
    async with pipeline_lock:
        if pipeline is not None:
            return JSONResponse({"status": "already_running"}, status_code=200)

        source = cfg.get("source", "en")
        target = cfg.get("target", "fr")
        device_index = cfg.get("device_index", None)
        mic_rate = cfg.get("mic_sample_rate", None)

        pipeline = LiveTranslator(
            source_lang=source,
            target_lang=target,
            event_sink=_event_sink,
            device_index=device_index,
            mic_sample_rate=mic_rate,
        )
        pipeline.start()

        return {
            "status": "started",
            "source": source,
            "target": target,
            "device_index": device_index,
            "mic_sample_rate": mic_rate,
        }


@app.post("/stop")
async def stop_pipeline():
    """Stop the pipeline if running."""
    global pipeline
    async with pipeline_lock:
        if pipeline is not None:
            pipeline.stop()
            pipeline = None
        return {"status": "stopped"}


@app.get("/status")
async def status():
    """Return pipeline status."""
    return {"running": pipeline is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # We don't expect incoming messages; keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)


# Serve static UI under /static and root /
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/")
async def root_page():
    index_path = static_dir / "index.html"
    return FileResponse(index_path)


def run():
    uvicorn.run("ui.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()

