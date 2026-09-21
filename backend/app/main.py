from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from . import config
from .engine_worker import worker
from .prefs import SPECS

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


def validate_image(data: bytes, content_type: str | None) -> None:
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ValueError(f"Image exceeds {config.MAX_UPLOAD_BYTES} bytes")
    if content_type and content_type not in ALLOWED_TYPES and not content_type.startswith("image/"):
        raise ValueError("Unsupported content type")
    try:
        img = Image.open(__import__("io").BytesIO(data))
        img.verify()
    except Exception as exc:
        raise ValueError(f"Invalid image: {exc}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, worker.load)
    yield
    worker.stop_session()


app = FastAPI(title="Psychomantium", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8790",
        "http://localhost:8790",
        "http://127.0.0.1:5173",
        "http://10.1.1.100:8790",
        "http://10.1.1.100:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return worker.health()


@app.get("/ready")
def ready():
    body = worker.readiness()
    if not worker.ready:
        return JSONResponse(body, status_code=503)
    return body


@app.get("/api/status")
def status():
    return worker.diagnostics()


@app.get("/api/models")
def list_models():
    return {"models": worker.models_public(), "model": worker.model_id, "loading": worker.loading, "ready": worker.ready}


@app.put("/api/model")
async def put_model(payload: dict[str, Any]):
    mid = str(payload.get("model") or payload.get("id") or "").strip()
    if not mid:
        return JSONResponse({"error": "missing model"}, status_code=400)
    result = await asyncio.get_event_loop().run_in_executor(None, worker.select_model, mid)
    if not result.get("ok"):
        status = 409 if "progress" in str(result.get("error") or "") else 500
        if "unknown" in str(result.get("error") or ""):
            status = 400
        return JSONResponse(result, status_code=status)
    return result


@app.get("/api/preferences")
def get_preferences():
    return {"prefs": worker.prefs, "specs": SPECS, "path": "/data/preferences/preferences.json"}


@app.put("/api/preferences")
async def put_preferences(payload: dict[str, Any]):
    persist = bool(payload.get("persist", True))
    body = payload.get("prefs") if isinstance(payload.get("prefs"), dict) else payload
    prefs = worker.apply_prefs(body, persist=persist)
    return {"ok": True, "persisted": persist, "prefs": prefs}


@app.post("/api/session/reset-view")
def reset_view():
    return worker.reset_orientation()


@app.post("/api/session/start")
async def start_session(
    image: UploadFile = File(...),
    prompt: str = Form(""),
):
    data = await image.read()
    try:
        validate_image(data, image.content_type)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not worker.ready:
        return JSONResponse({"error": worker.load_error or "model not ready"}, status_code=503)
    result = await asyncio.get_event_loop().run_in_executor(None, worker.start_session, data, prompt)
    if not result.get("ok", False):
        return JSONResponse(result, status_code=500)
    return result


@app.post("/api/session/stop")
def stop_session():
    worker.stop_session()
    return {"ok": True, "session_active": worker.session_active}


@app.post("/api/session/snapshot")
def snapshot():
    path = worker.snapshot()
    if not path:
        return JSONResponse({"error": "no frame"}, status_code=400)
    return {"ok": True, "path": path}


@app.post("/api/intention")
async def intention(payload: dict[str, Any]):
    text = str(payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    if len(text) > 400:
        return JSONResponse({"error": "too long"}, status_code=400)
    intent = worker.add_intention(text)
    return worker._intent_public(intent)


@app.get("/api/snapshots/{name}")
def get_snapshot(name: str):
    path = config.OUTPUT_DIR / "snapshots" / name
    if not path.is_file():
        return JSONResponse({"error": "missing"}, status_code=404)
    return FileResponse(path)


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    worker.note_client(1)
    last_sent = None
    try:
        await ws.send_text(json.dumps({"type": "hello", **worker.diagnostics()}))
        seed_payload, seed_meta = worker.peek_latest()
        if seed_meta:
            await ws.send_text(json.dumps(seed_meta))
        if seed_payload is not None:
            await ws.send_bytes(seed_payload)
            last_sent = seed_payload

        async def reader():
            while True:
                msg = await ws.receive_json()
                typ = msg.get("type")
                if typ == "controls":
                    buttons = msg.get("buttons") or []
                    mouse = msg.get("mouse") or [0, 0]
                    analog = msg.get("analog") or [0, 0]
                    arrows = msg.get("arrows") or []
                    seq = worker.set_controls(
                        buttons,
                        (float(mouse[0]), float(mouse[1])),
                        0,
                        (float(analog[0]), float(analog[1])),
                        list(arrows),
                    )
                    await ws.send_text(json.dumps({"type": "ack", "seq": seq}))
                elif typ == "prefs":
                    prefs = worker.apply_prefs(msg.get("prefs") or msg, persist=False)
                    await ws.send_text(json.dumps({"type": "prefs", "prefs": prefs, "persisted": False}))
                elif typ == "reset_orientation":
                    result = worker.reset_orientation()
                    await ws.send_text(json.dumps({"type": "view_reset", **result}))
                elif typ == "intention":
                    intent = worker.add_intention(str(msg.get("text") or ""))
                    await ws.send_text(json.dumps({"type": "intention", **worker._intent_public(intent)}))
                elif typ == "stop":
                    worker.stop_session()
                elif typ == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

        async def writer():
            nonlocal last_sent
            while True:
                payload, meta = await asyncio.get_event_loop().run_in_executor(None, worker.wait_latest, 0.75)
                if meta:
                    await ws.send_text(json.dumps(meta))
                if payload is not None and payload is not last_sent:
                    last_sent = payload
                    await ws.send_bytes(payload)

        await asyncio.gather(reader(), writer())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        worker.note_client(-1)
