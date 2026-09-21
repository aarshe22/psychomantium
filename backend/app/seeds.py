"""Original first-person start frames. Not Biome's gallery.

These are procedural stills (horizon-level, readable ground) used as
Waypoint seeds. The 1B DiT continues from pixels, not from captions.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import config

SEEDS_DIR = Path(__file__).resolve().parent.parent / "seeds"

GALLERY: list[dict[str, str]] = [
    {
        "id": "grid-street",
        "label": "Grid street",
        "caption": "First-person on a paved block with a road grid and building fronts.",
    },
    {
        "id": "dusk-avenue",
        "label": "Dusk avenue",
        "caption": "Same street language at dusk; skyline and asphalt still readable.",
    },
    {
        "id": "forest-path",
        "label": "Forest path",
        "caption": "A dirt path under a canopy; ground plane leads forward.",
    },
    {
        "id": "stone-hall",
        "label": "Stone hall",
        "caption": "Interior corridor with a vanishing point down the hall.",
    },
    {
        "id": "plaza",
        "label": "Open plaza",
        "caption": "An open paved square with distant facades on the horizon.",
    },
]


def _fill_sky_ground(w: int, h: int, sky: tuple[int, int, int], ground: tuple[int, int, int], horizon: float = 0.48) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    hy = int(h * horizon)
    for y in range(hy):
        t = y / max(hy, 1)
        img[y] = (
            int(sky[0] * (1 - t) + sky[0] * 0.55 * t),
            int(sky[1] * (1 - t) + sky[1] * 0.7 * t),
            int(sky[2] * (1 - t) + min(255, sky[2] + 20) * t),
        )
    for y in range(hy, h):
        t = (y - hy) / max(h - hy, 1)
        img[y] = (
            int(ground[0] * (1 - 0.35 * t)),
            int(ground[1] * (1 - 0.25 * t)),
            int(ground[2] * (1 - 0.2 * t)),
        )
    return img


def _perspective_road(img: np.ndarray, color: tuple[int, int, int], lane: tuple[int, int, int] | None = None) -> None:
    h, w, _ = img.shape
    hy = int(h * 0.48)
    cx = w // 2
    top_half = max(8, w // 28)
    bot_half = w // 3
    for y in range(hy, h):
        t = (y - hy) / max(h - hy, 1)
        half = int(top_half + (bot_half - top_half) * (t**1.35))
        x0, x1 = max(0, cx - half), min(w, cx + half)
        img[y, x0:x1] = color
        if lane and t > 0.08:
            dash = int(y / (6 + t * 18)) % 2 == 0
            if dash:
                mid = 1 + int(1 + t * 2)
                img[y, cx - mid : cx + mid] = lane


def _blocks(draw: ImageDraw.ImageDraw, w: int, h: int, color: tuple[int, int, int], dusk: bool = False) -> None:
    hy = int(h * 0.48)
    rng = np.random.default_rng(7)
    for side in (-1, 1):
        x = w // 2 + side * 28
        for i in range(5):
            bw = 18 + int(rng.integers(10, 28))
            bh = 40 + int(rng.integers(20, 70))
            if side < 0:
                box = [x - bw - i * 22, hy - bh + i * 4, x - i * 22, hy + 6]
            else:
                box = [x + i * 22, hy - bh + i * 4, x + bw + i * 22, hy + 6]
            c = tuple(max(0, min(255, ch + int(rng.integers(-18, 18)))) for ch in color)
            if dusk:
                c = (int(c[0] * 0.7), int(c[1] * 0.55), int(c[2] * 0.5))
            draw.rectangle(box, fill=c)


def _trees(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    hy = int(h * 0.5)
    rng = np.random.default_rng(3)
    for side in (-1, 1):
        for i in range(7):
            x = w // 2 + side * (36 + i * 26)
            bh = 50 + int(rng.integers(10, 40))
            draw.rectangle([x - 3, hy - bh, x + 3, hy + 8], fill=(52, 36, 22))
            draw.ellipse([x - 16, hy - bh - 22, x + 16, hy - bh + 18], fill=(28, 72, 34))


def _corridor(img: np.ndarray) -> None:
    h, w, _ = img.shape
    hy = int(h * 0.42)
    cx = w // 2
    img[:] = (38, 34, 30)
    for y in range(h):
        t = abs(y - hy) / h
        half = int(18 + (w * 0.42) * (t**0.9))
        x0, x1 = max(0, cx - half), min(w, cx + half)
        if y < hy:
            img[y, x0:x1] = (72, 64, 52)
        else:
            img[y, x0:x1] = (48, 42, 36)
    img[hy - 2 : hy + 2, cx - 10 : cx + 10] = (180, 160, 90)


def render_seed(seed_id: str, size: tuple[int, int] = (640, 360)) -> bytes:
    w, h = size
    if seed_id == "dusk-avenue":
        arr = _fill_sky_ground(w, h, (48, 32, 58), (42, 36, 38), 0.47)
        _perspective_road(arr, (28, 26, 28), (90, 70, 40))
        im = Image.fromarray(arr)
        _blocks(ImageDraw.Draw(im), w, h, (70, 58, 62), dusk=True)
    elif seed_id == "forest-path":
        arr = _fill_sky_ground(w, h, (110, 140, 160), (46, 58, 32), 0.5)
        _perspective_road(arr, (78, 58, 32), None)
        im = Image.fromarray(arr)
        _trees(ImageDraw.Draw(im), w, h)
    elif seed_id == "stone-hall":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        _corridor(arr)
        im = Image.fromarray(arr)
    elif seed_id == "plaza":
        arr = _fill_sky_ground(w, h, (168, 186, 210), (130, 124, 112), 0.52)
        _perspective_road(arr, (150, 144, 132), None)
        im = Image.fromarray(arr)
        _blocks(ImageDraw.Draw(im), w, h, (150, 142, 130))
    else:
        arr = _fill_sky_ground(w, h, (142, 178, 214), (64, 66, 62), 0.48)
        _perspective_road(arr, (48, 48, 50), (220, 200, 80))
        im = Image.fromarray(arr)
        _blocks(ImageDraw.Draw(im), w, h, (118, 112, 108))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def ensure_gallery() -> Path:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    for item in GALLERY:
        dest = SEEDS_DIR / f"{item['id']}.jpg"
        if not dest.is_file() or dest.stat().st_size < 800:
            dest.write_bytes(render_seed(item["id"]))
    try:
        extra = config.OUTPUT_DIR.parent / "seeds"
        extra.mkdir(parents=True, exist_ok=True)
        for item in GALLERY:
            dest = SEEDS_DIR / f"{item['id']}.jpg"
            copy = extra / f"{item['id']}.jpg"
            if dest.is_file() and not copy.is_file():
                copy.write_bytes(dest.read_bytes())
    except OSError:
        pass
    return SEEDS_DIR


def catalog() -> list[dict[str, Any]]:
    ensure_gallery()
    out: list[dict[str, Any]] = []
    for item in GALLERY:
        out.append(
            {
                **item,
                "url": f"/api/seeds/{item['id']}.jpg",
                "default": item["id"] == GALLERY[0]["id"],
            }
        )
    return out


def jpeg_for(seed_id: str) -> bytes | None:
    if seed_id not in {g["id"] for g in GALLERY}:
        return None
    ensure_gallery()
    path = SEEDS_DIR / f"{seed_id}.jpg"
    if path.is_file():
        return path.read_bytes()
    return render_seed(seed_id)
