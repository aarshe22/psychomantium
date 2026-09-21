"""Host-persisted experience knobs. Stored under /data/preferences (compose bind-mount)."""

from __future__ import annotations

import json
from typing import Any

from . import config
from .dream_scene import DREAM_WORLD_PROMPT

PREFS_DIR = config.PREFS_DIR
PREFS_PATH = PREFS_DIR / "preferences.json"

DEFAULTS: dict[str, Any] = {
    "resolution": 360,
    "temperature": 0.4,
    "look_sensitivity": 1.75,
    "jpeg_quality": 86,
    "wander": 0.0,
    "motion_smoothing": 0.15,
    "dream_sharpness": 0.45,
    "drift_delay": 5.0,
    "drift_interval": 0.0,
    "drift_strength": 0.55,
    "drift_steps": 4,
    "steer_move": True,
    "initial_note": "An explorable dream",
    "world_prompt": DREAM_WORLD_PROMPT,
    "model_id": "Overworld/Waypoint-1.5-1B-360P",
    "fps_lock": False,
    "inpaint": True,
}

SPECS: dict[str, dict[str, float | str]] = {
    "resolution": {"min": 360, "max": 720, "step": 60, "label": "Output resolution"},
    "temperature": {"min": 0.4, "max": 1.8, "step": 0.05, "label": "Inference temperature"},
    "look_sensitivity": {"min": 0.25, "max": 2.5, "step": 0.05, "label": "Look sensitivity"},
    "jpeg_quality": {"min": 55, "max": 95, "step": 1, "label": "Stream JPEG quality"},
    "wander": {"min": 0.0, "max": 0.3, "step": 0.01, "label": "Idle wander"},
    "motion_smoothing": {"min": 0.0, "max": 0.85, "step": 0.05, "label": "Motion smoothing"},
    "dream_sharpness": {"min": 0.0, "max": 1.0, "step": 0.05, "label": "Dream sharpness"},
    "drift_delay": {"min": 1.0, "max": 20.0, "step": 1.0, "label": "Dream drift delay"},
    "drift_interval": {"min": 0.0, "max": 45.0, "step": 1.0, "label": "Dream drift interval"},
    "drift_strength": {"min": 0.1, "max": 1.0, "step": 0.05, "label": "Dream drift strength"},
    "drift_steps": {"min": 2.0, "max": 8.0, "step": 1.0, "label": "Dream drift steps"},
}


def clamp_prefs(raw: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(DEFAULTS)
    if raw:
        src.update(raw)
    out = dict(DEFAULTS)
    out["resolution"] = int(max(360, min(720, round(float(src["resolution"]) / 60) * 60)))
    out["temperature"] = float(max(0.4, min(1.8, float(src["temperature"]))))
    out["look_sensitivity"] = float(max(0.25, min(2.5, float(src["look_sensitivity"]))))
    jpeg = int(max(40, min(95, round(float(src["jpeg_quality"])))))
    if jpeg <= 50:
        jpeg = int(DEFAULTS["jpeg_quality"])
    out["jpeg_quality"] = int(max(55, min(95, jpeg)))
    out["wander"] = float(max(0.0, min(0.3, float(src["wander"]))))
    out["motion_smoothing"] = float(max(0.0, min(0.85, float(src["motion_smoothing"]))))
    out["dream_sharpness"] = float(max(0.0, min(1.0, float(src["dream_sharpness"]))))
    out["drift_delay"] = float(max(1.0, min(20.0, round(float(src.get("drift_delay", 5.0))))))
    out["drift_interval"] = float(max(0.0, min(45.0, round(float(src.get("drift_interval", 0.0))))))
    out["drift_strength"] = float(max(0.1, min(1.0, float(src.get("drift_strength", 0.55)))))
    out["drift_steps"] = int(max(2, min(8, round(float(src.get("drift_steps", 4))))))
    out["steer_move"] = bool(src.get("steer_move", True))
    note = str(src.get("initial_note") or DEFAULTS["initial_note"])[:400]
    out["initial_note"] = note
    world = str(src.get("world_prompt") if src.get("world_prompt") is not None else DEFAULTS["world_prompt"])[:500]
    world = world.strip()
    migrated_road = world.startswith("There is a standard road grid") or world in {
        "",
        "There is a standard road grid, and buildings.",
        "There is a standard road grid, and buildings",
    }
    low = world.lower()
    migrated_body = any(
        token in low
        for token in (
            "empty unarmed hands",
            "empty hands",
            "unarmed hands",
            "no weapons",
        )
    )
    if migrated_road:
        world = DEFAULTS["world_prompt"]
        out["inpaint"] = True
    elif migrated_body:
        world = DEFAULTS["world_prompt"]
    out["world_prompt"] = world
    out["model_id"] = config.resolve_model(str(src.get("model_id") or DEFAULTS["model_id"]))
    native_h = config.frame_size_for(out["model_id"])[1]
    if out["resolution"] > native_h:
        out["resolution"] = native_h
    out["fps_lock"] = bool(src.get("fps_lock", False))
    if not migrated_road:
        out["inpaint"] = bool(src.get("inpaint", DEFAULTS["inpaint"]))
    return out


def load_prefs() -> dict[str, Any]:
    PREFS_DIR.mkdir(parents=True, exist_ok=True)
    if PREFS_PATH.is_file():
        try:
            data = json.loads(PREFS_PATH.read_text())
            prefs = clamp_prefs(data)
            if (
                str(data.get("world_prompt") or "") != prefs["world_prompt"]
                or int(data.get("jpeg_quality") or 0) != int(prefs["jpeg_quality"])
                or int(data.get("resolution") or 0) != int(prefs["resolution"])
                or bool(data.get("inpaint", False)) != bool(prefs["inpaint"])
            ):
                save_prefs(prefs)
            return prefs
        except Exception:
            return dict(DEFAULTS)
    return dict(DEFAULTS)


def save_prefs(raw: dict[str, Any]) -> dict[str, Any]:
    PREFS_DIR.mkdir(parents=True, exist_ok=True)
    prefs = clamp_prefs(raw)
    tmp = PREFS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prefs, indent=2) + "\n")
    tmp.replace(PREFS_PATH)
    return prefs


def output_size(resolution: int) -> tuple[int, int]:
    h = int(resolution)
    w = int(round(h * 16 / 9))
    w -= w % 2
    h -= h % 2
    return w, h
