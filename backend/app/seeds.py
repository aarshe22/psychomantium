"""Photoreal start frames for Waypoint.

The 1B DiT continues from pixels. Schematic drawings are out of distribution.
Gallery stills are Overworld's public Space starters (cached on the host) plus
optional Klein-painted seeds from the standing prompt. User upload stays first-class.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

from PIL import Image

from . import config

SEEDS_DIR = config.SEEDS_DIR
OVERWORLD_DIR = SEEDS_DIR / "overworld"
PAINTED_DIR = SEEDS_DIR / "painted"

# Official demo stills from the Waypoint HF Space (same pack as the model card).
STARTERS: list[dict[str, str]] = [
    {
        "id": "ow-starter-18",
        "file": "starter_18.png",
        "label": "Official starter",
        "caption": "Public Overworld still used on the Waypoint-1.5 model card.",
        "default": "1",
    },
    {
        "id": "ow-starter-14",
        "file": "starter_14.png",
        "label": "Starter 14",
        "caption": "Public photoreal first-person still from Overworld's Waypoint Space.",
    },
    {
        "id": "ow-starter-22",
        "file": "starter_22.png",
        "label": "Starter 22",
        "caption": "Public photoreal first-person still from Overworld's Waypoint Space.",
    },
    {
        "id": "ow-starter-21",
        "file": "starter_21.png",
        "label": "Starter 21",
        "caption": "Public photoreal first-person still from Overworld's Waypoint Space.",
    },
    {
        "id": "ow-starter-9",
        "file": "starter_9.png",
        "label": "Starter 9",
        "caption": "Public photoreal first-person still from Overworld's Waypoint Space.",
    },
]

SPACE_REPO = "Overworld/waypoint-1-small"


def _to_jpeg(src: Path | bytes, quality: int = 90) -> bytes:
    if isinstance(src, Path):
        img = Image.open(src)
    else:
        img = Image.open(io.BytesIO(src))
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _cache_starters() -> None:
    OVERWORLD_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for item in STARTERS:
        dest = OVERWORLD_DIR / f"{item['id']}.jpg"
        if dest.is_file() and dest.stat().st_size > 20_000:
            continue
        raw = Path(
            hf_hub_download(
                repo_id=SPACE_REPO,
                filename=item["file"],
                repo_type="space",
            )
        )
        dest.write_bytes(_to_jpeg(raw))


def ensure_gallery() -> Path:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    PAINTED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _cache_starters()
    except Exception:
        # Catalog still lists painted seeds / upload; starters appear when HF is reachable.
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
        path = OVERWORLD_DIR / f"{item['id']}.jpg"
        if not path.is_file():
            continue
        out.append(
            {
                "id": item["id"],
                "label": item["label"],
                "caption": item["caption"],
                "url": f"/api/seeds/{item['id']}.jpg",
                "source": "overworld",
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
    for folder in (OVERWORLD_DIR, PAINTED_DIR):
        path = folder / f"{seed_id}.jpg"
        if path.is_file():
            return path.read_bytes()
    return None
