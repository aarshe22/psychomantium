import os
from pathlib import Path

MODEL_ID = os.environ.get("WAYPOINT_MODEL", "Overworld/Waypoint-1.5-1B-360P")
QUANT = os.environ.get("WAYPOINT_QUANT") or None
DEVICE = os.environ.get("WAYPOINT_DEVICE", "cuda")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/outputs"))
PREFS_DIR = Path(os.environ.get("PREFS_DIR", "/data/preferences"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "78"))
DISCONNECT_GRACE_SEC = float(os.environ.get("DISCONNECT_GRACE_SEC", "8"))
HOST_BIND = os.environ.get("HOST_BIND", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8791"))
ENGINE_SHA = "b3f1e725b222679a517632918cc78bba0c9fa433"

# 360P checkpoint expects 640x360; 720p checkpoint expects 1280x720.
if "360" in MODEL_ID:
    FRAME_SIZE = (640, 360)  # (W, H)
else:
    FRAME_SIZE = (1280, 720)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREFS_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "snapshots").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "samples").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "probe").mkdir(parents=True, exist_ok=True)
