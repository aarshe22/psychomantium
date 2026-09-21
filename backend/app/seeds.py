"""Photoreal start frames for Waypoint.

The DiT continues from pixels. Gallery stills are CC0 eye-level paths and
roads, plus optional Klein-painted seeds. User upload stays first-class.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from . import config

SEEDS_DIR = config.SEEDS_DIR
CC0_DIR = SEEDS_DIR / "cc0"
PAINTED_DIR = SEEDS_DIR / "painted"
HIDDEN_PATH = CC0_DIR / "hidden.json"
USER_AGENT = "Psychomantium/0.1 (https://github.com/aarshe22/psychomantium)"

# CC0 only. Eye-level empty paths/roads.
STARTERS: list[dict[str, str]] = [
    {
        "id": "rural-lane",
        "label": "Dirt lane",
        "caption": "Rural dirt road with trees. CC0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5b/Rural_dirt_road_with_trees_and_stone_fencing.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Rural_dirt_road_with_trees_and_stone_fencing.jpg",
        "license": "CC0",
        "default": "1",
    },
    {
        "id": "forest-path",
        "label": "Forest path",
        "caption": "Woodland trail. CC0, Mary / StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/MCACZTS9BA.jpg",
        "page": "https://stocksnap.io/photo/nature-path-MCACZTS9BA",
        "license": "CC0",
    },
    {
        "id": "woodland-road",
        "label": "Woodland road",
        "caption": "Gravel track under trees. CC0, Bernard Spragg / StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/ESO8XIDNEC.jpg",
        "page": "https://stocksnap.io/photo/nature-path-ESO8XIDNEC",
        "license": "CC0",
    },
    {
        "id": "stone-path",
        "label": "Stone path",
        "caption": "Winding stone path in woods. CC0, World Travel Adventures / StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/ZVYAK6FCCL.jpg",
        "page": "https://stocksnap.io/photo/nature-path-ZVYAK6FCCL",
        "license": "CC0",
    },
    {
        "id": "grass-path",
        "label": "Grass path",
        "caption": "Narrow grass trail. CC0, Bernard Spragg / StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/0KEW2IKHUF.jpg",
        "page": "https://stocksnap.io/photo/nature-path-0KEW2IKHUF",
        "license": "CC0",
    },
    {
        "id": "farm-dirt",
        "label": "Farm dirt road",
        "caption": "Empty rural dirt road. CC0, StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/TQ6F3TW2LU.jpg",
        "page": "https://stocksnap.io/photo/rural-dirt-TQ6F3TW2LU",
        "license": "CC0",
    },
    {
        "id": "moor-highway",
        "label": "Moor highway",
        "caption": "Empty rural highway. CC0, Dave Meier / StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/DC980ABE32.jpg",
        "page": "https://stocksnap.io/photo/road-rural-DC980ABE32",
        "license": "CC0",
    },
    {
        "id": "desert-highway",
        "label": "Desert highway",
        "caption": "Empty desert two-lane. CC0, Salvatore Ventura / StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/XJ2BKV9ASS.jpg",
        "page": "https://stocksnap.io/photo/highway-road-XJ2BKV9ASS",
        "license": "CC0",
    },
    {
        "id": "canyon-road",
        "label": "Canyon road",
        "caption": "Empty canyon highway. CC0, StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/TXCAIIB92G.jpg",
        "page": "https://stocksnap.io/photo/travel-road-TXCAIIB92G",
        "license": "CC0",
    },
    {
        "id": "mountain-curve",
        "label": "Mountain curve",
        "caption": "Empty mountain road. CC0, StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/CVFA77QA7F.jpg",
        "page": "https://stocksnap.io/photo/curve-curvedroad-CVFA77QA7F",
        "license": "CC0",
    },
    {
        "id": "red-rock-track",
        "label": "Red rock track",
        "caption": "Desert dirt track. CC0, StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/OSDDRVNIFI.jpg",
        "page": "https://stocksnap.io/photo/mountain-desert-OSDDRVNIFI",
        "license": "CC0",
    },
]


STARTER_IDS = {item["id"] for item in STARTERS}
PAINTED_ID = re.compile(r"^painted-[0-9]+$")


def _hidden_cc0() -> set[str]:
    if not HIDDEN_PATH.is_file():
        return set()
    try:
        raw = json.loads(HIDDEN_PATH.read_text())
    except Exception:
        return set()
    if isinstance(raw, dict):
        ids = raw.get("ids") or raw.get("hidden") or []
    else:
        ids = raw
    return {str(x) for x in ids if str(x) in STARTER_IDS}


def _save_hidden_cc0(ids: set[str]) -> None:
    CC0_DIR.mkdir(parents=True, exist_ok=True)
    keep = sorted(i for i in ids if i in STARTER_IDS)
    HIDDEN_PATH.write_text(json.dumps({"ids": keep}, indent=2) + "\n")


def _crop_16x9(img: Image.Image) -> Image.Image:
    w, h = img.size
    target = 16 / 9
    if w / h > target:
        nw = int(h * target)
        x = (w - nw) // 2
        img = img.crop((x, 0, x + nw, h))
    else:
        nh = int(w / target)
        y = max(0, (h - nh) // 3)  # bias slightly up (eye-level, less sky-only)
        img = img.crop((0, y, w, y + nh))
    if img.width > 1600:
        nh = int(img.height * 1600 / img.width)
        img = img.resize((1600, nh), Image.Resampling.LANCZOS)
    return img


def _cache_starters() -> None:
    CC0_DIR.mkdir(parents=True, exist_ok=True)
    hidden = _hidden_cc0()
    for item in STARTERS:
        dest = CC0_DIR / f"{item['id']}.jpg"
        if item["id"] in hidden:
            if dest.is_file():
                dest.unlink()
            continue
        if dest.is_file() and dest.stat().st_size > 20_000:
            continue
        req = urllib.request.Request(item["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        out = _crop_16x9(img)
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=90)
        dest.write_bytes(buf.getvalue())
    keep = {item["id"] for item in STARTERS} - hidden
    for stale in CC0_DIR.glob("*.jpg"):
        if stale.stem not in keep:
            stale.unlink()
    meta = [{k: v for k, v in item.items() if k != "url"} for item in STARTERS]
    (CC0_DIR / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


def ensure_gallery() -> Path:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    PAINTED_DIR.mkdir(parents=True, exist_ok=True)
    CC0_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _cache_starters()
    except Exception:
        pass
    return SEEDS_DIR


def save_painted(jpeg: bytes, label: str = "Klein painted", prompt: str = "") -> dict[str, Any]:
    PAINTED_DIR.mkdir(parents=True, exist_ok=True)
    sid = f"painted-{int(time.time() * 1000)}"
    path = PAINTED_DIR / f"{sid}.jpg"
    path.write_bytes(jpeg)
    short = (label or prompt or "Klein painted").strip()[:40] or "Klein painted"
    cap = (prompt or "FLUX.2 Klein seed.").strip()[:240]
    meta = {
        "id": sid,
        "label": short,
        "caption": cap,
        "url": f"/api/seeds/{sid}.jpg",
        "source": "klein",
        "prompt": (prompt or "").strip()[:500],
        "default": False,
        "deletable": True,
    }
    (PAINTED_DIR / f"{sid}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def delete_painted(seed_id: str) -> bool:
    return delete_seed(seed_id)


def delete_seed(seed_id: str) -> bool:
    sid = (seed_id or "").strip()
    if PAINTED_ID.match(sid):
        path = PAINTED_DIR / f"{sid}.jpg"
        info = PAINTED_DIR / f"{sid}.json"
        if not path.is_file() and not info.is_file():
            return False
        if path.is_file():
            path.unlink()
        if info.is_file():
            info.unlink()
        return True
    if sid in STARTER_IDS:
        hidden = _hidden_cc0()
        hidden.add(sid)
        _save_hidden_cc0(hidden)
        path = CC0_DIR / f"{sid}.jpg"
        if path.is_file():
            path.unlink()
        return True
    return False


def _painted_meta(path: Path) -> dict[str, Any]:
    sid = path.stem
    info_path = PAINTED_DIR / f"{sid}.json"
    extra: dict[str, Any] = {}
    if info_path.is_file():
        try:
            extra = json.loads(info_path.read_text())
        except Exception:
            extra = {}
    prompt = str(extra.get("prompt") or extra.get("caption") or "").strip()
    label = str(extra.get("label") or prompt or "Klein painted").strip()[:40]
    return {
        "id": sid,
        "label": label or "Klein painted",
        "caption": prompt or "FLUX.2 Klein seed cached on the host.",
        "url": f"/api/seeds/{sid}.jpg",
        "source": "klein",
        "prompt": prompt,
        "default": False,
        "deletable": True,
    }


def catalog() -> list[dict[str, Any]]:
    ensure_gallery()
    out: list[dict[str, Any]] = []
    painted = sorted(PAINTED_DIR.glob("painted-*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    for i, path in enumerate(painted[:48]):
        rec = _painted_meta(path)
        rec["default"] = i == 0
        out.append(rec)
    for item in STARTERS:
        path = CC0_DIR / f"{item['id']}.jpg"
        if not path.is_file():
            continue
        out.append(
            {
                "id": item["id"],
                "label": item["label"],
                "caption": item["caption"],
                "url": f"/api/seeds/{item['id']}.jpg",
                "source": "cc0",
                "license": item.get("license"),
                "default": bool(item.get("default")) and not painted,
                "deletable": True,
            }
        )
    return out


def jpeg_for(seed_id: str) -> bytes | None:
    ensure_gallery()
    for folder in (CC0_DIR, PAINTED_DIR):
        path = folder / f"{seed_id}.jpg"
        if path.is_file():
            return path.read_bytes()
    return None
