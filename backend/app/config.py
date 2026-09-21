import os
from pathlib import Path
from typing import Any

MODEL_ID = os.environ.get("WAYPOINT_MODEL", "Overworld/Waypoint-1.5-1B-360P")
QUANT = os.environ.get("WAYPOINT_QUANT") or None
DEVICE = os.environ.get("WAYPOINT_DEVICE", "cuda")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/outputs"))
PREFS_DIR = Path(os.environ.get("PREFS_DIR", "/data/preferences"))
SEEDS_DIR = Path(os.environ.get("SEEDS_DIR", "/data/seeds"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "78"))
DISCONNECT_GRACE_SEC = float(os.environ.get("DISCONNECT_GRACE_SEC", "8"))
MAX_FPS = float(os.environ.get("MAX_FPS", "30"))
HOST_BIND = os.environ.get("HOST_BIND", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8791"))
ENGINE_SHA = "b3f1e725b222679a517632918cc78bba0c9fa433"
SCENE_AUTHORING = os.environ.get("SCENE_AUTHORING", "1").strip().lower() not in {"0", "false", "no", "off"}

MODELS: list[dict[str, Any]] = [
    {
        "id": "Overworld/Waypoint-1.5-1B-360P",
        "label": "Waypoint 1B · 360p",
        "width": 640,
        "height": 360,
    },
    {
        "id": "Overworld/Waypoint-1.5-1B",
        "label": "Waypoint 1B · 720p",
        "width": 1280,
        "height": 720,
    },
]
MODEL_IDS = {m["id"] for m in MODELS}


def resolve_model(model_id: str | None) -> str:
    if model_id in MODEL_IDS:
        return str(model_id)
    if MODEL_ID in MODEL_IDS:
        return MODEL_ID
    return MODELS[0]["id"]


def frame_size_for(model_id: str | None) -> tuple[int, int]:
    mid = resolve_model(model_id)
    for m in MODELS:
        if m["id"] == mid:
            return int(m["width"]), int(m["height"])
    return (640, 360)


FRAME_SIZE = frame_size_for(MODEL_ID)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREFS_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "snapshots").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "samples").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "probe").mkdir(parents=True, exist_ok=True)
