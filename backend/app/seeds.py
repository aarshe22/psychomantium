"""Photoreal start frames for Waypoint.

The 1B DiT continues from pixels. Gallery stills are royalty-free eye-level
urban and rural photographs (Wikimedia Commons / StockSnap), plus optional
Klein-painted seeds. Overworld FPS stills are not listed: they show hands
and weapons. User upload stays first-class.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from . import config

SEEDS_DIR = config.SEEDS_DIR
CC0_DIR = SEEDS_DIR / "cc0"
PAINTED_DIR = SEEDS_DIR / "painted"
USER_AGENT = "Psychomantium/0.1 (https://github.com/aarshe22/psychomantium)"

# Curated empty streets/paths. No first-person hands or weapons.
STARTERS: list[dict[str, str]] = [
    {
        "id": "urban-ghent",
        "label": "Tree boulevard",
        "caption": "Empty Ghent avenue. CC BY-SA 4.0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Empty_street_in_Ghent.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Empty_street_in_Ghent.jpg",
        "license": "CC BY-SA 4.0",
        "default": "1",
    },
    {
        "id": "urban-bergama",
        "label": "Cobblestone street",
        "caption": "Empty Bergama street. CC BY-SA 4.0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/58/Empty_street_during_the_coronavirus_pandemic_in_Bergama%2C_%C4%B0zmir.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Empty_street_during_the_coronavirus_pandemic_in_Bergama,_%C4%B0zmir.jpg",
        "license": "CC BY-SA 4.0",
    },
    {
        "id": "urban-moscow",
        "label": "Plaza morning",
        "caption": "Empty Nikolskaya Street, Moscow. CC BY 4.0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Moscow_-_2025_-_empty_Nikolskaya_Street_in_the_morning.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Moscow_-_2025_-_empty_Nikolskaya_Street_in_the_morning.jpg",
        "license": "CC BY 4.0",
    },
    {
        "id": "urban-street",
        "label": "Snow street",
        "caption": "Snowed-in city street at night. CC BY 2.0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/27/Empty_Street_%2850903495093%29.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Empty_Street_(50903495093).jpg",
        "license": "CC BY 2.0",
    },
    {
        "id": "urban-london",
        "label": "Regent Street",
        "caption": "Empty Regent Street, London. CC BY-SA 4.0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/ef/Regent_Street_Central_London_UK_COVID_19_Empty_Street.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Regent_Street_Central_London_UK_COVID_19_Empty_Street.jpg",
        "license": "CC BY-SA 4.0",
    },
    {
        "id": "rural-fog",
        "label": "Foggy farm road",
        "caption": "Rural dirt road in fog, Texas. CC BY-SA 4.0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Rural_dirt_road_and_trees_in_the_fog_in_Texas.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Rural_dirt_road_and_trees_in_the_fog_in_Texas.jpg",
        "license": "CC BY-SA 4.0",
    },
    {
        "id": "rural-forest",
        "label": "Forest path",
        "caption": "Deciduous forest path, Finland. CC BY-SA 4.0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/72/Forest_path_through_a_deciduous_forest_in_spring%2C_Finland.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Forest_path_through_a_deciduous_forest_in_spring,_Finland.jpg",
        "license": "CC BY-SA 4.0",
    },
    {
        "id": "rural-dales",
        "label": "Dales road",
        "caption": "Yorkshire Dales country road. CC BY-SA 3.0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/06/2014_Yorkshire_Dales_country_road_Swaledale_Askrigg.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:2014_Yorkshire_Dales_country_road_Swaledale_Askrigg.jpg",
        "license": "CC BY-SA 3.0",
    },
    {
        "id": "rural-lane",
        "label": "Dirt lane",
        "caption": "Rural dirt road with trees. CC0, Wikimedia Commons.",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5b/Rural_dirt_road_with_trees_and_stone_fencing.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Rural_dirt_road_with_trees_and_stone_fencing.jpg",
        "license": "CC0",
    },
    {
        "id": "rural-stocksnap",
        "label": "Open highway",
        "caption": "Empty rural highway. CC0, Dave Meier / StockSnap.",
        "url": "https://cdn.stocksnap.io/img-thumbs/960w/DC980ABE32.jpg",
        "page": "https://stocksnap.io/photo/road-rural-DC980ABE32",
        "license": "CC0",
    },
]


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
    for item in STARTERS:
        dest = CC0_DIR / f"{item['id']}.jpg"
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
    meta = [
        {k: v for k, v in item.items() if k != "url"}
        for item in STARTERS
    ]
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


def save_painted(jpeg: bytes, label: str = "Klein painted") -> dict[str, Any]:
    PAINTED_DIR.mkdir(parents=True, exist_ok=True)
    sid = f"painted-{int(time.time())}"
    path = PAINTED_DIR / f"{sid}.jpg"
    path.write_bytes(jpeg)
    return {
        "id": sid,
        "label": label[:40],
        "caption": "FLUX.2 Klein seed from the standing world prompt.",
        "url": f"/api/seeds/{sid}.jpg",
        "source": "klein",
        "default": False,
    }


def catalog() -> list[dict[str, Any]]:
    ensure_gallery()
    out: list[dict[str, Any]] = []
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
                "default": bool(item.get("default")),
            }
        )
    painted = sorted(PAINTED_DIR.glob("painted-*.jpg"), reverse=True)
    for path in painted[:12]:
        sid = path.stem
        out.append(
            {
                "id": sid,
                "label": "Klein painted",
                "caption": "FLUX.2 Klein seed from the standing world prompt.",
                "url": f"/api/seeds/{sid}.jpg",
                "source": "klein",
                "default": False,
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
