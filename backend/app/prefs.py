"""Host-persisted experience knobs. Stored under /data/preferences (compose bind-mount)."""

from __future__ import annotations

import json
from typing import Any

from . import config

PREFS_DIR = config.PREFS_DIR
PREFS_PATH = PREFS_DIR / "preferences.json"

DEFAULTS: dict[str, Any] = {
    "resolution": 360,
    "temperature": 1.0,
    "look_sensitivity": 1.0,
    "jpeg_quality": 78,
    "wander": 0.0,
    "motion_smoothing": 0.15,
    "dream_sharpness": 0.45,
    "steer_move": True,
    "initial_note": "An explorable dream",
    "world_prompt": "There is a standard road grid, and buildings.",
}

SPECS: dict[str, dict[str, float | str]] = {
    "resolution": {"min": 360, "max": 720, "step": 60, "label": "Output resolution"},
    "temperature": {"min": 0.4, "max": 1.8, "step": 0.05, "label": "Inference temperature"},
    "look_sensitivity": {"min": 0.25, "max": 2.5, "step": 0.05, "label": "Look sensitivity"},
    "jpeg_quality": {"min": 40, "max": 95, "step": 1, "label": "Stream JPEG quality"},
    "wander": {"min": 0.0, "max": 0.3, "step": 0.01, "label": "Idle wander"},
    "motion_smoothing": {"min": 0.0, "max": 0.85, "step": 0.05, "label": "Motion smoothing"},
    "dream_sharpness": {"min": 0.0, "max": 1.0, "step": 0.05, "label": "Dream sharpness"},
}


def clamp_prefs(raw: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(DEFAULTS)
    if raw:
        src.update(raw)
    out = dict(DEFAULTS)
    out["resolution"] = int(max(360, min(720, round(float(src["resolution"]) / 60) * 60)))
    out["temperature"] = float(max(0.4, min(1.8, float(src["temperature"]))))
    out["look_sensitivity"] = float(max(0.25, min(2.5, float(src["look_sensitivity"]))))
    out["jpeg_quality"] = int(max(40, min(95, round(float(src["jpeg_quality"])))))
    out["wander"] = float(max(0.0, min(0.3, float(src["wander"]))))
    out["motion_smoothing"] = float(max(0.0, min(0.85, float(src["motion_smoothing"]))))
    out["dream_sharpness"] = float(max(0.0, min(1.0, float(src["dream_sharpness"]))))
    out["steer_move"] = bool(src.get("steer_move", True))
    note = str(src.get("initial_note") or DEFAULTS["initial_note"])[:400]
    out["initial_note"] = note
    world = str(src.get("world_prompt") if src.get("world_prompt") is not None else DEFAULTS["world_prompt"])[:500]
    out["world_prompt"] = world.strip() or DEFAULTS["world_prompt"]
    return out


def load_prefs() -> dict[str, Any]:
    PREFS_DIR.mkdir(parents=True, exist_ok=True)
    if PREFS_PATH.is_file():
        try:
            data = json.loads(PREFS_PATH.read_text())
            return clamp_prefs(data)
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
