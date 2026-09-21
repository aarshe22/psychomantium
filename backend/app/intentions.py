"""Deterministic intention parser.

Waypoint-1.5-1B has prompt_conditioning: null, so set_prompt() is not a
supported live-text path. Intentions are mapped to:

- controller holds / impulses (supported)
- experimental KV-reset + append_frame after a color-grade of the last
  generated frames (supported engine APIs, not text-to-world)

Never claim visual verification without a measured frame statistic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

Kind = Literal["ongoing", "transition", "transform", "look", "world", "unknown"]
Status = Literal["received", "submitted", "failed", "visually_verified"]

KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE = 87, 65, 83, 68, 32


@dataclass
class Intention:
    id: int
    raw: str
    kind: Kind
    status: Status = "received"
    engine_action: str = ""
    note: str = ""
    hold_buttons: set[int] = field(default_factory=set)
    mouse_bias: tuple[float, float] = (0.0, 0.0)
    burst_batches: int = 0
    transform: Optional[str] = None
    verify: Optional[str] = None
    active: bool = True


def parse_intention(text: str, next_id: int) -> Intention:
    raw = text.strip()
    t = raw.lower()
    t = re.sub(r"[.!?]+$", "", t)

    intent = Intention(id=next_id, raw=raw, kind="unknown", note="No supported mapping.")

    if re.search(r"\b(fly|flying|float|hover|levitat)", t):
        intent.kind = "ongoing"
        intent.hold_buttons = {KEY_SPACE}
        intent.mouse_bias = (0.0, -0.08)
        intent.engine_action = "hold Space (jump) + slight look-up"
        intent.note = "Mapped to jump/look-up. The model has no dedicated fly flag."
        return intent

    if re.search(r"\b(walk|run|go)\s+forward\b|\bkeep walking\b", t):
        intent.kind = "ongoing"
        intent.hold_buttons = {KEY_W}
        intent.engine_action = "hold W"
        intent.note = "Ongoing forward movement."
        return intent

    if re.search(r"\b(enter|go into|go in|inside|through the door)\b", t):
        intent.kind = "transition"
        intent.hold_buttons = {KEY_W}
        intent.burst_batches = 18
        intent.engine_action = "hold W for ~18 generation batches"
        intent.note = "One-shot approach. No object detector; does not identify a building."
        return intent

    if re.search(r"\blook (left|right|up|down)\b", t):
        direction = re.search(r"left|right|up|down", t).group(0)
        deltas = {
            "left": (-0.45, 0.0),
            "right": (0.45, 0.0),
            "up": (0.0, -0.35),
            "down": (0.0, 0.35),
        }
        intent.kind = "look"
        intent.mouse_bias = deltas[direction]
        intent.burst_batches = 6
        intent.engine_action = f"mouse look {direction}"
        intent.note = "Impulse look via mouse velocity."
        return intent

    if re.search(r"\b(night|nighttime|darker|dusk|evening)\b", t):
        intent.kind = "transform"
        intent.transform = "night"
        intent.verify = "luminance_drop"
        intent.engine_action = "experimental recondition: darken+blue last frames, reset KV, append_frame"
        intent.note = (
            "This checkpoint has no live text prompt. Night is a color-grade of the "
            "latest frames, then engine.reset()+append_frame()."
        )
        return intent

    if re.search(r"\b(forest|woods|trees|jungle)\b", t):
        intent.kind = "transform"
        intent.transform = "forest"
        intent.verify = "green_rise"
        intent.engine_action = "experimental recondition: green-grade last frames, reset KV, append_frame"
        intent.note = (
            "Not a true scenery rewrite. Re-seeds from a green-shifted copy of the last frames."
        )
        return intent

    if re.search(r"\b(day|daytime|morning|brighter|noon)\b", t):
        intent.kind = "transform"
        intent.transform = "day"
        intent.verify = "luminance_rise"
        intent.engine_action = "experimental recondition: brighten last frames, reset KV, append_frame"
        intent.note = "Re-seeds from a brightened copy of the last frames."
        return intent

    if re.search(r"\b(stop flying|land|walk normally)\b", t) or t in {"clear", "clear holds"}:
        intent.kind = "ongoing"
        intent.hold_buttons = set()
        intent.engine_action = "clear ongoing holds"
        intent.note = "Clears ongoing controller intentions."
        intent.active = False
        return intent

    # Anything else becomes a standing world rule, not a failure.
    intent.kind = "world"
    intent.status = "received"
    intent.active = True
    intent.engine_action = "Klein inpaint of current still, then session standing prompt"
    intent.note = (
        "Edits the current first-person still via FLUX.2 Klein, then appends this line "
        "to the session standing prompt. Cumulative until Stop. Independent of Auto-InPaint."
    )
    return intent
