"""Open-scene vs texture-lock metrics and Klein prompts for explorable dreams.

Waypoint continues from pixels. A close-up repeating surface (brick, dirt, siding)
becomes the entire world unless we cut back to an open first-person view.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DREAM_WORLD_PROMPT = (
    "An imaginary first-person dream you can walk through. Eye-level, a path or "
    "clearing ahead, open sky when looking out. Empty unarmed hands. No weapons, "
    "tools, HUD, or text overlay."
)

DREAM_BREATH_PROMPT = (
    "Pull the camera back to a stable eye-level first-person view of this same "
    "kind of place. Restore distance: a path, clearing, or ground plane ahead and "
    "open sky or weather above. Do not fill the frame with one repeating wall, "
    "brick, dirt patch, or corrugated surface. Keep photoreal materials. "
    "No people, hands, weapons, tools, HUD, or text."
)

DREAM_LOOKOUT_PROMPT = (
    "The viewer is looking out from too close to a surface. Open the view: "
    "horizon, sky, and a walkable place ahead in the same dream. Do not tile "
    "the nearby texture across the sky or sides. Photoreal first-person, "
    "eye-level. No people, hands, weapons, tools, HUD, or text."
)

DREAM_DRIFT_PROMPT = (
    "Continue this as an imaginary dream: a slight adjacent place the viewer "
    "could walk into, same first-person eye-level. Keep a path or opening ahead "
    "and sky if the view looks out. Soft change of thought, not a hard cut. "
    "Do not fill the frame with one repeating surface. Do not add people, "
    "hands, weapons, tools, HUD, or text."
)

LOCK_THRESHOLD = 0.62
OPEN_THRESHOLD = 0.48
LOCK_STREAK = 3


@dataclass(frozen=True)
class SceneMetrics:
    lock: float
    openness: float
    tile_sim: float
    row_sim: float
    lum_std: float
    skyish: bool
    locked: bool
    open: bool


def analyze_frame(rgb: np.ndarray) -> SceneMetrics:
    """Score whether a first-person still is an open place or a sticky surface."""
    if rgb is None or rgb.size == 0:
        return SceneMetrics(1.0, 0.0, 1.0, 1.0, 0.0, False, True, False)
    img = np.asarray(rgb)
    if img.ndim == 4:
        img = img[-1]
    if img.ndim != 3 or img.shape[2] < 3:
        return SceneMetrics(1.0, 0.0, 1.0, 1.0, 0.0, False, True, False)
    small = cv2.resize(img[..., :3], (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lum_std = float(gray.std())
    tile_vecs: list[np.ndarray] = []
    tile_means: list[float] = []
    for y in range(0, 54, 18):
        for x in range(0, 96, 24):
            patch = gray[y : y + 18, x : x + 24]
            tile_means.append(float(patch.mean()))
            flat = patch.ravel()
            n = float(np.linalg.norm(flat)) + 1e-6
            tile_vecs.append(flat / n)
    sims = [
        float(tile_vecs[i] @ tile_vecs[j])
        for i in range(len(tile_vecs))
        for j in range(i + 1, len(tile_vecs))
    ]
    tile_sim = float(np.mean(sims)) if sims else 0.0
    tile_mean_std = float(np.std(np.array(tile_means, dtype=np.float32))) if tile_means else 0.0
    norms = np.linalg.norm(gray, axis=1, keepdims=True) + 1e-6
    rows = gray / norms
    row_sim = float((rows[:-1] * rows[1:]).sum(axis=1).mean())
    upper = gray[:18]
    lower = gray[36:]
    upper_mean = float(upper.mean())
    lower_mean = float(lower.mean())
    split = abs(upper_mean - lower_mean)
    skyish = upper_mean > lower_mean + 10.0
    rp = gray.mean(axis=1) - float(gray.mean())
    spec = np.abs(np.fft.rfft(rp))
    body = spec[2:]
    periodic = bool(body.size) and float(body.max()) / (float(body.mean()) + 1e-6) > 5.0
    lock = (
        0.42 * (1.0 - min(split / 36.0, 1.0))
        + 0.33 * (1.0 - min(tile_mean_std / 28.0, 1.0))
        + 0.25 * (1.0 - min(lum_std / 48.0, 1.0))
    )
    if periodic and split < 18.0:
        lock = max(lock, 0.78)
    if lum_std < 16.0 and split < 12.0:
        lock = max(lock, 0.72)
    lock = float(max(0.0, min(1.0, lock)))
    openness = (
        min(split / 36.0, 1.0) * 0.40
        + min(lum_std / 42.0, 1.0) * 0.22
        + min(tile_mean_std / 28.0, 1.0) * 0.20
        + (0.18 if skyish else 0.0)
    )
    openness = float(max(0.0, min(1.0, openness)))
    locked = lock >= LOCK_THRESHOLD and openness < OPEN_THRESHOLD + 0.06
    open_scene = openness >= OPEN_THRESHOLD and not locked
    return SceneMetrics(
        lock=lock,
        openness=openness,
        tile_sim=tile_sim,
        row_sim=row_sim,
        lum_std=lum_std,
        skyish=skyish,
        locked=locked,
        open=open_scene,
    )


def rgb_to_seed(rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb)
    if frame.ndim == 4:
        frame = frame[-1]
    stacked = np.repeat(np.ascontiguousarray(frame)[None, ...], 4, axis=0)
    return stacked


def synthetic_wall() -> np.ndarray:
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        band = 40 + (y // 18) % 2 * 70
        img[y, :] = (band + 20, band, band - 10)
        if y % 18 < 2:
            img[y, :] = (30, 28, 26)
    for x in range(0, w, 28):
        img[:, x : x + 2] = (35, 32, 30)
    return img


def synthetic_open_place() -> np.ndarray:
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:140, :] = (120, 170, 220)
    img[140:180, :] = (210, 200, 170)
    img[180:, :] = (70, 110, 55)
    rng = np.random.default_rng(0)
    noise = rng.integers(-18, 18, size=img.shape, dtype=np.int16)
    path = np.linspace(220, 420, h - 180).astype(int)
    for i, x in enumerate(path):
        img[180 + i, max(0, x - 40) : min(w, x + 40)] = (150, 130, 90)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def synthetic_dirt_closeup() -> np.ndarray:
    h, w = 360, 640
    rng = np.random.default_rng(3)
    base = rng.integers(70, 120, size=(h, w, 3), dtype=np.uint8)
    return base


if __name__ == "__main__":
    wall = analyze_frame(synthetic_wall())
    place = analyze_frame(synthetic_open_place())
    dirt = analyze_frame(synthetic_dirt_closeup())
    print("wall", wall)
    print("open", place)
    print("dirt", dirt)
    assert wall.locked, wall
    assert place.open, place
    assert not place.locked, place
    assert dirt.locked or dirt.openness < place.openness, dirt
    assert place.openness > wall.openness
    print("ok")

