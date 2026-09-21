"""Persistent GPU worker: one loaded WorldEngine, one active session."""

from __future__ import annotations

import gc
import io
import os
import subprocess
import types
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from . import config
from .intentions import Intention, parse_intention
from .prefs import clamp_prefs, load_prefs, output_size, save_prefs
from .scene_authoring import authoring
from .seeds import save_painted

FRAME_HEADER_MAGIC = 0x50535943  # 'PSYC'


_GPU_UTIL_LOCK = threading.Lock()
_GPU_UTIL_CACHE: tuple[float, float | None] = (0.0, None)


def gpu_mem_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def gpu_util_pct() -> float | None:
    """SM busy percent from nvidia-smi (cached ~0.8s)."""
    global _GPU_UTIL_CACHE
    now = time.monotonic()
    with _GPU_UTIL_LOCK:
        ts, cached = _GPU_UTIL_CACHE
        if now - ts < 0.8 and cached is not None:
            return cached
    value: float | None = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=1.2,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        line = out.strip().splitlines()[0]
        value = float(line.split(",")[0].strip())
        value = max(0.0, min(100.0, value))
    except Exception:
        try:
            if torch.cuda.is_available() and hasattr(torch.cuda, "utilization"):
                value = float(torch.cuda.utilization())
                value = max(0.0, min(100.0, value))
        except Exception:
            value = None
    with _GPU_UTIL_LOCK:
        _GPU_UTIL_CACHE = (now, value)
    return value


def encode_jpeg(rgb: np.ndarray, quality: int, size_wh: tuple[int, int] | None = None) -> bytes:
    if size_wh and (rgb.shape[1], rgb.shape[0]) != size_wh:
        rgb = cv2.resize(rgb, size_wh, interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def pack_frame(seq: int, jpeg: bytes, width: int, height: int, gen_ms: float) -> bytes:
    header = np.array(
        [FRAME_HEADER_MAGIC, seq, len(jpeg), width, height, int(gen_ms * 1000), 0],
        dtype=np.uint32,
    ).tobytes()
    return header + jpeg


def image_to_seed(image_bytes: bytes, size_wh: tuple[int, int]) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    w, h = size_wh
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    stacked = np.repeat(arr[None, ...], 4, axis=0)
    return torch.from_numpy(stacked.copy())


def color_grade(frames: np.ndarray, mode: str) -> np.ndarray:
    x = frames.astype(np.float32)
    if mode == "night":
        x *= 0.38
        x[..., 2] = np.clip(x[..., 2] * 1.35, 0, 255)
        x[..., 0] *= 0.65
    elif mode == "forest":
        x[..., 1] = np.clip(x[..., 1] * 1.4 + 12, 0, 255)
        x[..., 0] *= 0.82
        x[..., 2] *= 0.88
    elif mode == "day":
        x = np.clip(x * 1.28 + 18, 0, 255)
    return x.astype(np.uint8)


@dataclass
class ControlState:
    buttons: set[int] = field(default_factory=set)
    mouse: tuple[float, float] = (0.0, 0.0)
    analog: tuple[float, float] = (0.0, 0.0)
    arrows: set[str] = field(default_factory=set)
    scroll: int = 0
    seq: int = 0


class EngineWorker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.engine = None
        self.ready = False
        self.loading = False
        self.load_error: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self.session_active = False
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.gpu_jobs: queue.Queue = queue.Queue()
        self.loop_idle = threading.Event()
        self.loop_idle.set()
        self.gpu_thread = threading.Thread(target=self._gpu_main, name="gpu-engine", daemon=True)
        self.gpu_thread.start()
        self.controls = ControlState()
        self.intentions: list[Intention] = []
        self.next_intent_id = 1
        self.latest_payload: Optional[bytes] = None
        self.latest_meta: dict[str, Any] = {}
        self.frame_slot = threading.Condition()
        self.last_frames: Optional[np.ndarray] = None
        self.seed_preview: Optional[bytes] = None
        self.original_seed: Optional[torch.Tensor] = None
        self._seed_reset_pending = False
        self.prompt: str = ""
        self.generation_fps: float = 0.0
        self.last_gen_ms: float = 0.0
        self.batches_done: int = 0
        self.frames_done: int = 0
        self.last_error: Optional[str] = None
        self.clients = 0
        self.disconnect_deadline: Optional[float] = None
        self.prompt_supported = False
        self.prefs = load_prefs()
        self.model_id = config.resolve_model(self.prefs.get("model_id"))
        self.pre_transform_stats: dict[str, float] = {}
        self.verify_until_batch = 0
        self.verify_intent_id: Optional[int] = None
        self.record_frames: list[np.ndarray] = []
        self.recording = False
        self.view_yaw = 0.0
        self.view_pitch = 0.0
        self.reset_look = False
        self.smooth_mouse = [0.0, 0.0]
        self._sigma_base = None
        self._sched_dirty = False
        self._prompt_dirty = True
        self.last_prompt_result = ""
        self.inpaint_status = "off"
        self.inpaint_progress = 0.0
        self._idle_batches = 0
        self._inpainted_this_idle = False
        self._intent_inpaint_pending: Optional[str] = None
        self._intent_inpaint_id: Optional[int] = None

    def _gpu_call(self, op: str, *args, wait: bool = True):
        box: dict[str, Any] = {"err": None, "result": None}
        done = threading.Event()
        self.gpu_jobs.put((op, args, box, done))
        if not wait:
            return None
        done.wait()
        if box["err"] is not None:
            raise box["err"]
        return box["result"]

    def _gpu_main(self) -> None:
        while True:
            item = self.gpu_jobs.get()
            if item is None:
                break
            op, args, box, done = item
            try:
                if op == "load":
                    box["result"] = self._load_engine()
                    done.set()
                elif op == "start":
                    box["result"] = self._start_on_gpu(*args)
                    done.set()
                    self.loop_idle.clear()
                    try:
                        self._loop()
                    finally:
                        self.loop_idle.set()
                elif op == "paint":
                    box["result"] = self._paint_on_gpu(*args)
                    done.set()
                else:
                    raise RuntimeError(f"unknown gpu op {op}")
            except Exception as exc:
                box["err"] = exc
                done.set()

    def load(self) -> None:
        with self.lock:
            if self.ready or self.loading:
                return
            self.loading = True
            self.load_error = None
            self.model_id = config.resolve_model(self.prefs.get("model_id") or self.model_id)
        try:
            self._gpu_call("load")
        except Exception as exc:
            with self.lock:
                self.load_error = f"{type(exc).__name__}: {exc}"
                self.loading = False
                self.ready = False
            raise

    def models_public(self) -> list[dict[str, Any]]:
        return [
            {
                **m,
                "selected": m["id"] == self.model_id,
            }
            for m in config.MODELS
        ]

    def select_model(self, model_id: str) -> dict[str, Any]:
        if model_id not in config.MODEL_IDS:
            return {"ok": False, "error": f"unknown model: {model_id}"}
        if self.loading:
            return {"ok": False, "error": "model load already in progress"}
        if model_id == self.model_id and self.ready:
            return {
                "ok": True,
                "model": self.model_id,
                "frame_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            }
        self.stop_session()
        with self.lock:
            if self.loading:
                return {"ok": False, "error": "model load already in progress"}
            self.loading = True
            self.ready = False
            self.load_error = None
            self.model_id = model_id
            prefs = dict(self.prefs)
            prefs["model_id"] = model_id
            prefs["resolution"] = config.frame_size_for(model_id)[1]
            self.prefs = save_prefs(prefs)
        try:
            self._gpu_call("load")
            w, h = self.frame_size
            return {"ok": True, "model": self.model_id, "frame_size": {"width": w, "height": h}, "prefs": dict(self.prefs)}
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            with self.lock:
                self.load_error = msg
                self.loading = False
                self.ready = False
            if "CUDA" in msg or "AcceleratorError" in type(exc).__name__:
                self._request_cuda_restart(msg)
            return {"ok": False, "error": msg}

    def _unload_engine(self) -> None:
        eng = self.engine
        self.engine = None
        self._sigma_base = None
        if eng is not None:
            del eng
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load_engine(self) -> None:
        t0 = time.perf_counter()
        from world_engine import WorldEngine

        self._unload_engine()
        engine = WorldEngine(self.model_id, quant=config.QUANT, device=config.DEVICE)
        self._install_temperature_hook(engine)
        self._sigma_base = engine.scheduler_sigmas.detach().clone()
        self._apply_scheduler(engine)
        prompt_supported = False
        try:
            cfg = engine.model_cfg
            prompt_supported = getattr(cfg, "prompt_conditioning", None) not in (None, False)
        except Exception:
            prompt_supported = False
        with self.lock:
            self.engine = engine
            self.prompt_supported = prompt_supported
            self.ready = True
            self.load_seconds = time.perf_counter() - t0
            self.loading = False

    @property
    def frame_size(self) -> tuple[int, int]:
        return config.frame_size_for(self.model_id)

    def _install_temperature_hook(self, engine) -> None:
        worker = self

        @torch.inference_mode()
        def gen_frame(eng, ctrl=None, return_img: bool = True):
            scale = float(worker.prefs.get("temperature", 0.4))
            x = torch.randn(eng.frm_shape, device=eng.device, dtype=eng.dtype).mul_(scale)
            inputs = eng.prep_inputs(x=x, ctrl=ctrl)
            x0 = eng._denoise_pass(x, inputs, eng.kv_cache).clone()
            eng._cache_pass(x0, inputs, eng.kv_cache)
            return eng.vae.decode(x0.squeeze(1)) if return_img else x0.squeeze(1)

        engine.gen_frame = types.MethodType(gen_frame, engine)

    def _apply_scheduler(self, engine) -> None:
        """Only safe before torch.compile / CUDA graphs capture."""
        if engine is None or self._sigma_base is None:
            return
        sharp = float(self.prefs.get("dream_sharpness", 0.45))
        base = self._sigma_base
        if abs(sharp - 0.45) < 0.02:
            engine.scheduler_sigmas.copy_(base)
            return
        mild = torch.tensor([1.0, 0.95, 0.88, 0.55, 0.0], device=base.device, dtype=base.dtype)
        wild = torch.tensor([1.0, 0.62, 0.28, 0.10, 0.0], device=base.device, dtype=base.dtype)
        if mild.shape != base.shape:
            return
        engine.scheduler_sigmas.copy_(mild.lerp(wild, sharp))

    def _request_cuda_restart(self, reason: str) -> None:
        self.last_error = reason
        self.ready = False
        self.session_active = False

        def _die() -> None:
            time.sleep(0.8)
            os._exit(1)

        threading.Thread(target=_die, name="cuda-restart", daemon=True).start()

    def apply_prefs(self, raw: dict, persist: bool = False) -> dict:
        prefs = save_prefs(raw) if persist else clamp_prefs(raw)
        with self.lock:
            self.prefs = prefs
            self._sched_dirty = True
            self._prompt_dirty = True
        # Live except scheduler remaps, which are only safe when CUDA graphs are not capturing.
        if not self.session_active and self.engine is not None:
            try:
                self._apply_scheduler(self.engine)
                self._sched_dirty = False
            except Exception:
                pass
        return prefs

    def reset_orientation(self) -> dict:
        with self.lock:
            self.reset_look = True
        return {"ok": True, "yaw": self.view_yaw, "pitch": self.view_pitch}

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "ready": self.ready, "loading": self.loading}

    def readiness(self) -> dict[str, Any]:
        mem = gpu_mem_mb()
        return {
            "status": "ready" if self.ready else ("loading" if self.loading else "not_ready"),
            "ready": self.ready,
            "loading": self.loading,
            "error": self.load_error,
            "model": self.model_id,
            "models": self.models_public(),
            "engine_sha": config.ENGINE_SHA,
            "quant": config.QUANT,
            "frame_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "prompt_conditioning": self.prompt_supported,
            "load_seconds": self.load_seconds,
            "session_active": self.session_active,
            "vram": mem,
            "gpu_util_pct": gpu_util_pct(),
            "generation_fps": self.generation_fps,
            "max_fps": config.MAX_FPS if bool(self.prefs.get("fps_lock", False)) else None,
            "fps_lock": bool(self.prefs.get("fps_lock", False)),
            "inpaint": bool(self.prefs.get("inpaint", False)),
            "inpaint_status": self.inpaint_status,
            "inpaint_progress": self.inpaint_progress,
            "authoring": authoring.status(),
        }

    def set_controls(
        self,
        buttons: list[int],
        mouse: tuple[float, float],
        scroll: int = 0,
        analog: tuple[float, float] = (0.0, 0.0),
        arrows: list[str] | None = None,
    ) -> int:
        with self.lock:
            self.controls.buttons = {int(b) for b in buttons if 0 <= int(b) < 256}
            mx = float(max(-1.5, min(1.5, mouse[0])))
            my = float(max(-1.5, min(1.5, mouse[1])))
            self.controls.mouse = (mx, my)
            ax = float(max(-1.0, min(1.0, analog[0])))
            ay = float(max(-1.0, min(1.0, analog[1])))
            self.controls.analog = (ax, ay)
            allowed = {"left", "right", "up", "down"}
            self.controls.arrows = {a for a in (arrows or []) if a in allowed}
            self.controls.scroll = 0
            self.controls.seq += 1
            return self.controls.seq

    def composed_prompt(self) -> str:
        parts: list[str] = []
        wp = str(self.prefs.get("world_prompt") or "").strip()
        note = (self.prompt or str(self.prefs.get("initial_note") or "")).strip()
        if wp:
            parts.append(wp.rstrip("."))
        if note and note.lower() not in wp.lower():
            parts.append(note.rstrip("."))
        for intent in self.intentions:
            if intent.active and intent.kind == "world":
                line = intent.raw.strip().rstrip(".")
                if line and line.lower() not in " ".join(parts).lower():
                    parts.append(line)
        return ". ".join(parts)[:800]

    def try_set_prompt(self) -> str:
        text = self.composed_prompt()
        engine = self.engine
        if engine is None:
            self.last_prompt_result = "no engine"
            return self.last_prompt_result
        if not text:
            self.last_prompt_result = "empty"
            return self.last_prompt_result
        if not self.prompt_supported:
            self.last_prompt_result = "set_prompt_unavailable: prompt_conditioning=null"
            for intent in self.intentions:
                if intent.kind == "world" and intent.active:
                    intent.status = "submitted"
                    intent.note = (
                        "Active as a standing world rule. The loaded Waypoint-1.5-1B checkpoint "
                        "has prompt_conditioning=null, so set_prompt is not wired into the DiT. "
                        "The seed image and movement still drive the world."
                    )
            return self.last_prompt_result
        try:
            engine.set_prompt(text)
            self.last_prompt_result = "set_prompt_ok"
            for intent in self.intentions:
                if intent.kind == "world" and intent.active:
                    intent.status = "submitted"
                    intent.engine_action = "standing world prompt (set_prompt)"
            return self.last_prompt_result
        except Exception as exc:
            self.last_prompt_result = f"set_prompt_unavailable: {type(exc).__name__}"
            for intent in self.intentions:
                if intent.kind == "world" and intent.active:
                    intent.status = "submitted"
                    intent.note = (
                        "Active as a standing world rule. The loaded Waypoint-1.5-1B checkpoint "
                        "has prompt_conditioning=null, so set_prompt is not wired into the DiT. "
                        "The seed image and movement still drive the world."
                    )
            return self.last_prompt_result

    def add_intention(self, text: str) -> Intention:
        with self.lock:
            intent = parse_intention(text, self.next_intent_id)
            self.next_intent_id += 1
            low = intent.raw.lower().strip().rstrip(".")
            if low in {"stop flying", "land", "walk normally", "clear"}:
                for old in self.intentions:
                    if old.kind == "ongoing":
                        old.active = False
            if "clear world" in low or "forget that" in low or low == "reset world rules":
                for old in self.intentions:
                    if old.kind == "world":
                        old.active = False
            self.intentions.append(intent)
            if len(self.intentions) > 40:
                self.intentions = self.intentions[-40:]
            if intent.kind == "world":
                self._prompt_dirty = True
            if (
                self.session_active
                and intent.active
                and intent.kind in {"world", "transform", "ongoing", "unknown"}
            ):
                self._intent_inpaint_pending = intent.raw
                self._intent_inpaint_id = intent.id
                if "Klein inpaint" not in (intent.engine_action or ""):
                    intent.engine_action = (
                        f"{intent.engine_action}; Klein inpaint of current still"
                        if intent.engine_action
                        else "Klein inpaint of current still"
                    )
                if "inpaint modifier" not in (intent.note or "").lower():
                    extra = (
                        "Spoken line is the inpaint modifier on the current frame, "
                        "independent of Auto-InPaint."
                    )
                    intent.note = f"{intent.note} {extra}".strip() if intent.note else extra
            return intent

    def _merged_ctrl(self):
        from world_engine import CtrlInput

        with self.lock:
            prefs = dict(self.prefs)
            buttons = set(self.controls.buttons)
            mouse = list(self.controls.mouse)
            analog = list(self.controls.analog)
            arrows = set(self.controls.arrows)
            for intent in self.intentions:
                if not intent.active:
                    continue
                buttons |= intent.hold_buttons
                mouse[0] += intent.mouse_bias[0]
                mouse[1] += intent.mouse_bias[1]
                if intent.kind in {"transition", "look"} and intent.burst_batches > 0:
                    intent.burst_batches -= 1
                    if intent.burst_batches <= 0:
                        intent.active = False
                        intent.hold_buttons = set()
                        intent.mouse_bias = (0.0, 0.0)

        sens = float(prefs.get("look_sensitivity", 1.75))
        mouse[0] *= sens
        mouse[1] *= sens
        mouse[0] += analog[0] * 0.55 * sens
        mouse[1] += analog[1] * 0.45 * sens
        if "left" in arrows:
            mouse[0] -= 0.28 * sens
        if "right" in arrows:
            mouse[0] += 0.28 * sens
        if "up" in arrows:
            mouse[1] -= 0.22 * sens
        if "down" in arrows:
            mouse[1] += 0.22 * sens

        if prefs.get("steer_move"):
            mag = (analog[0] ** 2 + analog[1] ** 2) ** 0.5
            if mag > 0.18:
                if analog[1] < -0.18:
                    buttons.add(87)
                if analog[1] > 0.18:
                    buttons.add(83)
                if analog[0] < -0.18:
                    buttons.add(65)
                if analog[0] > 0.18:
                    buttons.add(68)

        if self.reset_look:
            mouse[0] -= self.view_yaw * 0.45
            mouse[1] -= self.view_pitch * 0.85
            if abs(self.view_yaw) < 0.04 and abs(self.view_pitch) < 0.04:
                self.reset_look = False
                self.view_yaw = 0.0
                self.view_pitch = 0.0

        wander = float(prefs.get("wander", 0.0))
        if wander > 0 and abs(mouse[0]) < 0.02 and abs(mouse[1]) < 0.02 and not buttons:
            mouse[0] += float(np.random.normal(0, wander))
            mouse[1] += float(np.random.normal(0, wander * 0.6))

        sm = float(prefs.get("motion_smoothing", 0.0))
        self.smooth_mouse[0] = self.smooth_mouse[0] * sm + mouse[0] * (1.0 - sm)
        self.smooth_mouse[1] = self.smooth_mouse[1] * sm + mouse[1] * (1.0 - sm)
        mx = float(max(-1.5, min(1.5, self.smooth_mouse[0])))
        my = float(max(-1.5, min(1.5, self.smooth_mouse[1])))
        self.view_yaw += mx
        self.view_pitch += my
        self.view_yaw = float(max(-12.0, min(12.0, self.view_yaw)))
        self.view_pitch = float(max(-6.0, min(6.0, self.view_pitch)))
        return CtrlInput(button=buttons, mouse=(mx, my), scroll_wheel=0)

    def paint_seed(self, text: str) -> dict[str, Any]:
        if self.session_active:
            return {"ok": False, "error": "stop the dream before painting a new start frame"}
        if not self.loop_idle.is_set():
            return {"ok": False, "error": "GPU busy"}
        prompt = (text or "").strip()
        if not prompt:
            prompt = self.composed_prompt()
        try:
            return self._gpu_call("paint", prompt)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _paint_on_gpu(self, text: str) -> dict[str, Any]:
        if authoring.pipeline is None:
            authoring.load()
        pil, klein_prompt = authoring.generate_from_text(text, self.frame_size)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        jpeg = buf.getvalue()
        meta = save_painted(jpeg, label="Klein painted")
        return {"ok": True, "seed": meta, "klein_prompt": klein_prompt}

    def start_session(self, image_bytes: bytes, prompt: str = "") -> dict[str, Any]:
        if not self.ready or self.engine is None:
            return {"ok": False, "error": self.load_error or "model not ready"}
        try:
            seed = image_to_seed(image_bytes, self.frame_size)
            preview = encode_jpeg(seed[0].numpy(), 85)
        except Exception as exc:
            return {"ok": False, "error": f"bad seed image: {exc}"}
        self.stop_session()
        with self.lock:
            self.prompt = prompt.strip()
            self.seed_preview = preview
            self.last_error = None
            self.batches_done = 0
            self.frames_done = 0
            self.intentions = []
            self.record_frames = []
            self.session_active = True
            self.stop_event.clear()
            self.view_yaw = 0.0
            self.view_pitch = 0.0
            self.reset_look = False
            self.smooth_mouse = [0.0, 0.0]
            self._prompt_dirty = True
            self.original_seed = seed.detach().clone()
            self._seed_reset_pending = False
            self._idle_batches = 0
            self._inpainted_this_idle = False
            self._intent_inpaint_pending = None
            self._intent_inpaint_id = None
            self.inpaint_status = "armed" if bool(self.prefs.get("inpaint", False)) else "off"
        try:
            return self._gpu_call("start", seed)
        except Exception as exc:
            self.session_active = False
            msg = f"{type(exc).__name__}: {exc}"
            self.last_error = msg
            if "CUDA" in msg or "AcceleratorError" in type(exc).__name__:
                self._request_cuda_restart(msg)
            return {"ok": False, "error": msg}

    def _start_on_gpu(self, seed: torch.Tensor) -> dict[str, Any]:
        engine = self.engine
        assert engine is not None
        engine.reset()
        if self._sched_dirty:
            try:
                self._apply_scheduler(engine)
                self._sched_dirty = False
            except Exception:
                pass
        decoded = engine.append_frame(seed)
        self.try_set_prompt()
        from world_engine import CtrlInput

        try:
            warm = engine.gen_frame(ctrl=CtrlInput())
            frames = warm.detach().to("cpu").numpy()
        except Exception:
            frames = decoded.detach().to("cpu").numpy()
        self.last_frames = frames
        self._publish_batch(frames, gen_ms=0.0, kind="seed")
        self.frames_done += int(frames.shape[0])
        return {"ok": True, "prompt_applied": False, "reason": "prompt_conditioning is null on this checkpoint"}

    def request_seed_reset(self) -> dict[str, Any]:
        if not self.session_active or self.original_seed is None:
            return {"ok": False, "error": "no seed in this session"}
        self._seed_reset_pending = True
        return {"ok": True}

    def _reseed_on_gpu(self) -> None:
        engine = self.engine
        seed = self.original_seed
        if engine is None or seed is None:
            return
        engine.reset()
        decoded = engine.append_frame(seed)
        self.try_set_prompt()
        frames = decoded.detach().to("cpu").numpy()
        self.last_frames = frames
        self.view_yaw = 0.0
        self.view_pitch = 0.0
        self.reset_look = False
        self.smooth_mouse = [0.0, 0.0]
        self._publish_batch(frames, gen_ms=0.0, kind="seed")

    def stop_session(self) -> None:
        self.stop_event.set()
        if threading.current_thread() is not self.gpu_thread:
            self.loop_idle.wait(timeout=180)
        with self.lock:
            self.session_active = False
            self.thread = None
            self.disconnect_deadline = None
            if not bool(self.prefs.get("inpaint", False)):
                self.inpaint_status = "off"

    def note_client(self, delta: int) -> None:
        with self.lock:
            self.clients = max(0, self.clients + delta)
            if self.clients == 0 and self.session_active:
                self.disconnect_deadline = time.time() + config.DISCONNECT_GRACE_SEC
            elif self.clients > 0:
                self.disconnect_deadline = None

    def snapshot(self) -> Optional[str]:
        frames = self.last_frames
        if frames is None:
            return None
        path = config.OUTPUT_DIR / "snapshots" / f"snap-{int(time.time())}.jpg"
        rgb = frames[-1]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return str(path)

    def apply_pending_transforms(self) -> None:
        pending: list[Intention] = []
        with self.lock:
            for intent in self.intentions:
                if intent.active and intent.kind == "transform" and intent.transform and intent.status == "received":
                    pending.append(intent)
        if not pending or self.last_frames is None or self.engine is None:
            return
        frames = self.last_frames
        for intent in pending:
            if intent.id == self._intent_inpaint_id and self._intent_inpaint_pending:
                continue
            try:
                graded = color_grade(frames, intent.transform)
                self.pre_transform_stats = {
                    "luminance": float(frames.astype(np.float32).mean()),
                    "green": float(frames[..., 1].astype(np.float32).mean()),
                }
                seed = torch.from_numpy(np.ascontiguousarray(graded))
                self.engine.reset()
                decoded = self.engine.append_frame(seed)
                frames = decoded.detach().to("cpu").numpy()
                self.last_frames = frames
                intent.status = "submitted"
                intent.active = False
                self.verify_intent_id = intent.id
                self.verify_until_batch = self.batches_done + 4
                self._publish_batch(frames, gen_ms=0.0, kind="recondition")
            except Exception as exc:
                intent.status = "failed"
                intent.note = f"{intent.note} | engine error: {exc}"
                intent.active = False

    def _maybe_verify(self, frames: np.ndarray) -> None:
        if self.verify_intent_id is None or self.batches_done < self.verify_until_batch:
            return
        intent = next((i for i in self.intentions if i.id == self.verify_intent_id), None)
        self.verify_intent_id = None
        if intent is None:
            return
        lum = float(frames.astype(np.float32).mean())
        green = float(frames[..., 1].astype(np.float32).mean())
        pre = self.pre_transform_stats
        ok = False
        if intent.verify == "luminance_drop" and lum < pre.get("luminance", lum) * 0.92:
            ok = True
        elif intent.verify == "luminance_rise" and lum > pre.get("luminance", lum) * 1.05:
            ok = True
        elif intent.verify == "green_rise" and green > pre.get("green", green) * 1.05:
            ok = True
        if ok:
            intent.status = "visually_verified"
            intent.note += f" Measured post-frame lum={lum:.1f} green={green:.1f} (pre={pre})."
        else:
            intent.status = "submitted"
            intent.note += f" Submitted to engine; luminance/green did not clearly change (lum={lum:.1f}, green={green:.1f}, pre={pre})."

    def _publish_batch(self, frames: np.ndarray, gen_ms: float, kind: str) -> None:
        prefs = self.prefs
        quality = int(prefs.get("jpeg_quality", config.JPEG_QUALITY))
        out_wh = output_size(int(prefs.get("resolution", 360)))
        payloads = []
        base_seq = self.frames_done
        per = gen_ms / max(len(frames), 1)
        w, h = out_wh
        for i, frame in enumerate(frames):
            jpeg = encode_jpeg(frame, quality, out_wh)
            payloads.append(pack_frame(base_seq + i, jpeg, w, h, per))
        meta = {
            "type": "stats",
            "kind": kind,
            "generation_fps": self.generation_fps,
            "last_gen_ms": gen_ms,
            "batches": self.batches_done,
            "frames": self.frames_done + len(frames),
            "vram": gpu_mem_mb(),
            "gpu_util_pct": gpu_util_pct(),
            "fps_lock": bool(prefs.get("fps_lock", False)),
            "max_fps": config.MAX_FPS if bool(prefs.get("fps_lock", False)) else None,
            "inpaint": bool(prefs.get("inpaint", False)),
            "inpaint_status": self.inpaint_status,
            "inpaint_progress": self.inpaint_progress,
            "session_active": self.session_active,
            "intentions": [self._intent_public(i) for i in self.intentions[-12:]],
            "prompt_conditioning": self.prompt_supported,
            "error": self.last_error,
            "model": self.model_id,
            "models": self.models_public(),
            "prefs": dict(prefs),
            "composed_prompt": self.composed_prompt(),
            "prompt_apply": self.last_prompt_result,
            "view": {"yaw": self.view_yaw, "pitch": self.view_pitch, "resetting": self.reset_look},
            "native_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "stream_size": {"width": w, "height": h},
        }
        blob = b"".join(payloads)
        with self.frame_slot:
            self.latest_payload = blob
            self.latest_meta = meta
            self.frame_slot.notify_all()

    def peek_latest(self) -> tuple[Optional[bytes], dict[str, Any]]:
        with self.frame_slot:
            return self.latest_payload, dict(self.latest_meta)

    def wait_latest(self, timeout: float = 1.0) -> tuple[Optional[bytes], dict[str, Any]]:
        with self.frame_slot:
            self.frame_slot.wait(timeout=timeout)
            return self.latest_payload, dict(self.latest_meta)

    def _intent_public(self, intent: Intention) -> dict[str, Any]:
        return {
            "id": intent.id,
            "text": intent.raw,
            "kind": intent.kind,
            "status": intent.status,
            "engine_action": intent.engine_action,
            "note": intent.note,
            "active": intent.active,
        }

    def _user_steering(self) -> bool:
        """True if the player is actually moving/looking, ignoring idle wander."""
        with self.lock:
            if self.controls.buttons:
                return True
            if self.controls.arrows:
                return True
            mx, my = self.controls.mouse
            if abs(mx) > 0.03 or abs(my) > 0.03:
                return True
            ax, ay = self.controls.analog
            if abs(ax) > 0.12 or abs(ay) > 0.12:
                return True
            if self.reset_look:
                return True
            for intent in self.intentions:
                if not intent.active:
                    continue
                if intent.hold_buttons:
                    return True
                if abs(intent.mouse_bias[0]) > 0.02 or abs(intent.mouse_bias[1]) > 0.02:
                    return True
        return False

    def _publish_inpaint_progress(self, frac: float) -> None:
        self.inpaint_progress = float(max(0.0, min(1.0, frac)))
        with self.frame_slot:
            meta = dict(self.latest_meta) if self.latest_meta else {"type": "stats"}
            meta["inpaint_status"] = self.inpaint_status
            meta["inpaint_progress"] = self.inpaint_progress
            meta["kind"] = "inpaint"
            self.latest_meta = meta
            self.frame_slot.notify_all()

    def _idle_inpaint_on_gpu(self, modifier: str | None = None) -> None:
        engine = self.engine
        frames = self.last_frames
        if engine is None or frames is None or frames.size == 0:
            return
        self.inpaint_status = "running"
        self.inpaint_progress = 0.08
        self._publish_inpaint_progress(0.08)
        pil, _prompt = authoring.refine_frame(
            frames[-1],
            self.frame_size,
            user_request=modifier,
            on_progress=self._publish_inpaint_progress,
        )
        arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        stacked = np.repeat(arr[None, ...], 4, axis=0)
        seed = torch.from_numpy(np.ascontiguousarray(stacked))
        engine.reset()
        decoded = engine.append_frame(seed)
        out = decoded.detach().to("cpu").numpy()
        self.last_frames = out
        self.original_seed = seed.detach().clone()
        self.inpaint_status = "done"
        self.inpaint_progress = 1.0
        self._publish_batch(out, gen_ms=0.0, kind="inpaint")
        self.inpaint_progress = 0.0

    def _maybe_directed_inpaint(self) -> bool:
        """Speak → Klein edit of the current still. Ignores the Auto-InPaint toggle."""
        with self.lock:
            text = (self._intent_inpaint_pending or "").strip()
            intent_id = self._intent_inpaint_id
            standing = str(self.prefs.get("world_prompt") or "").strip()
            note = str(self.prefs.get("initial_note") or "").strip()
            if text:
                self._intent_inpaint_pending = None
        if not text:
            return False
        if self.last_frames is None or self.engine is None:
            with self.lock:
                if not (self._intent_inpaint_pending or "").strip():
                    self._intent_inpaint_pending = text
                    self._intent_inpaint_id = intent_id
            return False
        modifier = ". ".join(part for part in (standing, note, text) if part)
        intent = next((i for i in self.intentions if i.id == intent_id), None)
        try:
            if authoring.pipeline is None:
                self.inpaint_status = "loading"
                self.inpaint_progress = 0.0
                self._publish_inpaint_progress(0.0)
                authoring.load()
            self._idle_inpaint_on_gpu(modifier=modifier)
            self._inpainted_this_idle = True
            if intent is not None:
                if intent.status != "failed":
                    intent.status = "submitted"
                intent.engine_action = "Klein inpaint of current still (spoken modifier)"
            return True
        except Exception as exc:
            self.inpaint_status = "error"
            self._inpainted_this_idle = True
            self.last_error = f"inpaint: {type(exc).__name__}: {exc}"
            if intent is not None:
                intent.status = "failed"
                intent.note = f"{intent.note} | Klein error: {exc}".strip(" |")
            return False

    def _maybe_idle_inpaint(self) -> bool:
        if not bool(self.prefs.get("inpaint", False)):
            if self.inpaint_status not in {"off", "error"}:
                self.inpaint_status = "off"
            self._idle_batches = 0
            self._inpainted_this_idle = False
            return False
        if self._user_steering():
            self._idle_batches = 0
            self._inpainted_this_idle = False
            if self.inpaint_status in {"done", "running"}:
                self.inpaint_status = "armed"
            elif self.inpaint_status == "off":
                self.inpaint_status = "armed"
            return False
        self._idle_batches += 1
        if self._inpainted_this_idle:
            return False
        if self._idle_batches < 10:
            if self.inpaint_status in {"off", "done"}:
                self.inpaint_status = "armed"
            return False
        try:
            if authoring.pipeline is None:
                self.inpaint_status = "loading"
                self.inpaint_progress = 0.0
                self._publish_inpaint_progress(0.0)
                authoring.load()
            self._idle_inpaint_on_gpu()
            self._inpainted_this_idle = True
            return True
        except Exception as exc:
            self.inpaint_status = "error"
            self._inpainted_this_idle = True
            self.last_error = f"inpaint: {type(exc).__name__}: {exc}"
            return False

    def _loop(self) -> None:
        from world_engine import CtrlInput  # noqa: F401  (imported for side-effect-free type use)

        engine = self.engine
        assert engine is not None
        while not self.stop_event.is_set():
            deadline = self.disconnect_deadline
            if deadline is not None and time.time() > deadline:
                break
            try:
                self.apply_pending_transforms()
                if self._seed_reset_pending:
                    self._reseed_on_gpu()
                    self._seed_reset_pending = False
                if self._prompt_dirty:
                    self.try_set_prompt()
                    self._prompt_dirty = False
                if self._maybe_directed_inpaint():
                    continue
                if self._maybe_idle_inpaint():
                    continue
                for intent in self.intentions:
                    if intent.status == "received" and intent.kind != "transform" and intent.engine_action != "none":
                        intent.status = "submitted"
                ctrl = self._merged_ctrl()
                t_cycle = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = engine.gen_frame(ctrl=ctrl)
                frames = out.detach().to("cpu").numpy()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_ms = (time.perf_counter() - t0) * 1000.0
                n = int(frames.shape[0])
                self.last_gen_ms = gen_ms
                self.batches_done += 1
                self.frames_done += n
                self.last_frames = frames
                if self.recording:
                    self.record_frames.extend([frames[i] for i in range(n)])
                self._maybe_verify(frames)
                elapsed = time.perf_counter() - t_cycle
                fps_lock = bool(self.prefs.get("fps_lock", False))
                if fps_lock:
                    min_interval = n / max(config.MAX_FPS, 1.0)
                    self.generation_fps = n / max(elapsed, min_interval)
                    self._publish_batch(frames, gen_ms=gen_ms, kind="generated")
                    remain = min_interval - (time.perf_counter() - t_cycle)
                    if remain > 0 and self.stop_event.wait(remain):
                        break
                else:
                    self.generation_fps = n / max(elapsed, 1e-6)
                    self._publish_batch(frames, gen_ms=gen_ms, kind="generated")
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                if "CUDA" in self.last_error or "AcceleratorError" in type(exc).__name__:
                    self._request_cuda_restart(self.last_error)
                    break
                time.sleep(0.25)
        with self.lock:
            self.session_active = False

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.readiness(),
            "generation_fps": self.generation_fps,
            "last_gen_ms": self.last_gen_ms,
            "batches": self.batches_done,
            "frames": self.frames_done,
            "clients": self.clients,
            "error": self.last_error,
            "intentions": [self._intent_public(i) for i in self.intentions[-12:]],
            "prompt": self.prompt,
            "world_prompt": self.prefs.get("world_prompt"),
            "composed_prompt": self.composed_prompt(),
            "prompt_apply": self.last_prompt_result,
            "prefs": dict(self.prefs),
            "view": {"yaw": self.view_yaw, "pitch": self.view_pitch, "resetting": self.reset_look},
        }


worker = EngineWorker()
