#!/usr/bin/env python3
"""Minimal containerized inference probe: seed image, movement, sample, metrics."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from app import config
from app.engine_worker import color_grade, encode_jpeg, image_to_seed

SEED_URLS = [
    "https://huggingface.co/spaces/Overworld/waypoint-1-small/resolve/main/starter_18.png",
    "https://huggingface.co/Overworld/Waypoint-1.5-1B/resolve/main/assets/Overworld_Loop.webp",
]


def download_seed() -> bytes:
    env_path = os.environ.get("PROBE_SEED")
    if env_path and Path(env_path).is_file():
        return Path(env_path).read_bytes()
    last_err = None
    for url in SEED_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as res:
                return res.read()
        except Exception as exc:
            last_err = exc
    # Synthetic 16:9 fallback so the probe still exercises the engine offline.
    print(f"seed download failed ({last_err}); using synthetic gradient")
    w, h = config.FRAME_SIZE
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack(
        [
            (x * 255 / max(w, 1)).astype(np.uint8),
            (y * 255 / max(h, 1)).astype(np.uint8),
            np.full((h, w), 80, dtype=np.uint8),
        ],
        axis=-1,
    )
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    from world_engine import WorldEngine, CtrlInput

    out_dir = config.OUTPUT_DIR / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "model": config.MODEL_ID,
        "engine_sha": config.ENGINE_SHA,
        "quant": config.QUANT,
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "frame_size": list(config.FRAME_SIZE),
    }

    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    engine = WorldEngine(config.MODEL_ID, quant=config.QUANT, device=config.DEVICE)
    load_s = time.perf_counter() - t0
    report["startup_seconds"] = load_s
    report["prompt_conditioning"] = getattr(engine.model_cfg, "prompt_conditioning", None)
    report["vram_after_load_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    seed_bytes = download_seed()
    seed = image_to_seed(seed_bytes, config.FRAME_SIZE)
    engine.reset()
    t1 = time.perf_counter()
    decoded = engine.append_frame(seed)
    report["append_frame_seconds"] = time.perf_counter() - t1
    frames = [decoded.detach().to("cpu").numpy()[i] for i in range(decoded.shape[0])]

    # Warmup + measured movement sequence.
    sequence = (
        [CtrlInput()] * 2
        + [CtrlInput(button={87})] * 12  # W forward
        + [CtrlInput(mouse=(0.25, 0.0))] * 6
        + [CtrlInput(button={32})] * 4  # jump
        + [CtrlInput()] * 4
    )
    gen_times = []
    for i, ctrl in enumerate(sequence):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = engine.gen_frame(ctrl=ctrl)
        cpu = out.detach().to("cpu").numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        n = int(out.shape[0])
        gen_times.append((dt, n))
        for k in range(n):
            frames.append(cpu[k])
        print(f"batch {i:02d} {n} frames in {dt*1000:.1f} ms ({n/dt:.2f} gen-fps)")

    # Skip first two batches as compile/warmup.
    steady = gen_times[2:] if len(gen_times) > 4 else gen_times
    total_s = sum(t for t, _ in steady)
    total_f = sum(n for _, n in steady)
    report["warmup_batches_excluded"] = 2
    report["steady_generation_fps"] = (total_f / total_s) if total_s else 0
    report["steady_batch_ms_mean"] = (1000 * total_s / len(steady)) if steady else 0
    report["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    report["method"] = (
        "Wall-clock around WorldEngine.gen_frame() plus copy of the 4 decoded frames to CPU, "
        "with torch.cuda.synchronize() before/after. First two batches excluded as torch.compile "
        "warmup. FPS is generated frames / elapsed. Not display interpolation."
    )

    # Experimental night recondition.
    last4 = np.stack(frames[-4:], axis=0)
    graded = color_grade(last4, "night")
    engine.reset()
    engine.append_frame(torch.from_numpy(np.ascontiguousarray(graded)))
    night_frames = []
    for _ in range(6):
        out = engine.gen_frame(ctrl=CtrlInput(button={87}))
        cpu = out.detach().to("cpu").numpy()
        for k in range(cpu.shape[0]):
            night_frames.append(cpu[k])
    pre = float(last4.astype(np.float32).mean())
    post = float(np.stack(night_frames, 0).astype(np.float32).mean())
    report["night_recondition"] = {
        "pre_luminance": pre,
        "post_luminance": post,
        "luminance_dropped": post < pre * 0.92,
    }

    sample_path = out_dir / "probe_sample.mp4"
    write_mp4(sample_path, frames + night_frames, fps=24)
    (out_dir / "probe_last.jpg").write_bytes(encode_jpeg(frames[-1], 90))
    report["sample"] = str(sample_path)
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
