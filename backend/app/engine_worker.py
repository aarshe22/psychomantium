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
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from . import config
from .dream_scene import (
    DREAM_BREATH_PROMPT,
    DREAM_LOOKOUT_PROMPT,
    LOCK_STREAK,
    analyze_frame,
    drift_prompt,
    rgb_to_seed,
)
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


def image_to_seed(image_bytes: bytes, size_wh: tuple[int, int], n_frames: int = 4) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    w, h = size_wh
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    stacked = np.repeat(arr[None, ...], max(int(n_frames), 1), axis=0)
    return torch.from_numpy(stacked.copy())


def as_frame_batch(decoded) -> np.ndarray:
    """Normalize engine RGB to [T, H, W, C] uint8. OWL VAE returns one frame; TAEHV returns 4."""
    if torch.is_tensor(decoded):
        arr = decoded.detach().to("cpu").numpy()
    else:
        arr = np.asarray(decoded)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"expected HW3 or THW3 frames, got {arr.shape}")
    return np.ascontiguousarray(arr)


KLEIN_MAX_H = 360


def klein_work_size(native_wh: tuple[int, int]) -> tuple[int, int]:
    """Klein + 720p Waypoint on one GPU OOMs / CUDA-asserts. Edit at 360p, then scale."""
    w, h = int(native_wh[0]), int(native_wh[1])
    if h <= KLEIN_MAX_H:
        return w, h
    nh = KLEIN_MAX_H
    nw = int(round(w * nh / max(h, 1)))
    nw -= nw % 2
    nh -= nh % 2
    return max(16, nw), max(16, nh)


def seed_for_append(seed: torch.Tensor, n_frames: int) -> torch.Tensor:
    """OWL encode wants [H,W,C]; TAEHV wants [T,H,W,C] with T=temporal_compression."""
    n = max(int(n_frames), 1)
    if seed.dim() == 3:
        if n == 1:
            return seed
        return seed.unsqueeze(0).repeat(n, 1, 1, 1)
    if n == 1:
        return seed[0]
    if seed.shape[0] == n:
        return seed
    return seed[:1].repeat(n, 1, 1, 1)


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
        self._intent_inpaint_queue: deque[tuple[str, int]] = deque()
        self._open_seeds: deque[torch.Tensor] = deque(maxlen=6)
        self._lock_streak = 0
        self._freeze_walk = False
        self._lookout_pending = False
        self._last_scene_event = ""
        self._last_scene = None
        self._last_rescue_at = 0.0
        self._idle_since: Optional[float] = None
        self._last_drift_at: Optional[float] = None
        self._grace_batches = 0
        self.bootstrap_phase = "waiting"
        self.bootstrap = "waiting for GPU worker"
        self.bootstrap_detail = "The API is up. The GPU thread has not started loading weights yet."
        self.bootstrap_at = time.monotonic()

    def _set_bootstrap(self, phase: str, label: str, detail: str = "") -> None:
        with self.lock:
            self.bootstrap_phase = phase
            self.bootstrap = label
            self.bootstrap_detail = detail or label
            self.bootstrap_at = time.monotonic()
        try:
            self._publish_status("bootstrap")
        except Exception:
            pass

    def _bootstrap_public(self) -> dict[str, Any]:
        with self.lock:
            elapsed = time.monotonic() - float(self.bootstrap_at)
            return {
                "bootstrap_phase": self.bootstrap_phase,
                "bootstrap": self.bootstrap,
                "bootstrap_detail": self.bootstrap_detail,
                "bootstrap_elapsed_s": round(elapsed, 1),
            }

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
        self._set_bootstrap(
            "gpu_queue",
            "queued on GPU thread",
            "Waiting for exclusive GPU access so this checkpoint can load.",
        )
        try:
            self._gpu_call("load")
        except Exception as exc:
            with self.lock:
                self.load_error = f"{type(exc).__name__}: {exc}"
                self.loading = False
                self.ready = False
            self._set_bootstrap(
                "load_failed",
                "weight load failed",
                f"{type(exc).__name__}: {exc}",
            )
            raise

    def models_public(self) -> list[dict[str, Any]]:
        return [
            {
                **m,
                "selected": m["id"] == self.model_id,
                "prompt_conditioning": bool(m.get("prompt_conditioning")),
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
        self._set_bootstrap(
            "gpu_queue",
            "queued on GPU thread",
            f"Switching to {model_id}. Waiting for exclusive GPU access.",
        )
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
            self._set_bootstrap("load_failed", "weight load failed", msg)
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
        # Blackwell + world_engine's torch.compile (triton cudagraphs) indexes
        # the 5-sigma Euler ladder out of range (device-side assert `tmp21 < 5`),
        # kills the worker, and 502/503s the UI so WASD never reaches a live loop.
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        torch._dynamo.config.disable = True
        import world_engine.world_engine as we
        from world_engine import WorldEngine

        self._unload_engine()
        self._set_bootstrap(
            "load_config",
            "reading checkpoint config",
            f"Opening {self.model_id} (config.yaml / model card).",
        )
        orig_ae = we.get_ae
        orig_from = we.WorldModel.from_pretrained
        orig_pe = we.PromptEncoder
        orig_kv = we.StaticKVCache
        worker = self

        def _get_ae(*args, **kwargs):
            worker._set_bootstrap(
                "load_vae",
                "loading VAE",
                "Pixel decoder for this checkpoint. Host cache after the first download.",
            )
            return orig_ae(*args, **kwargs)

        def _from_pretrained(*args, **kwargs):
            worker._set_bootstrap(
                "load_dit",
                "loading DiT weights",
                "World transformer. This is the long step if the files are not already on disk.",
            )
            return orig_from(*args, **kwargs)

        class _PromptEncoder(orig_pe):
            def __init__(self, *args, **kwargs):
                worker._set_bootstrap(
                    "load_text",
                    "loading text encoder",
                    "UMT5 (or the checkpoint’s prompt encoder) for set_prompt.",
                )
                super().__init__(*args, **kwargs)

        class _KV(orig_kv):
            def __init__(self, *args, **kwargs):
                worker._set_bootstrap(
                    "load_kv",
                    "allocating KV cache",
                    "Static attention cache used by the live generation loop.",
                )
                super().__init__(*args, **kwargs)

        we.get_ae = _get_ae
        we.WorldModel.from_pretrained = _from_pretrained
        we.PromptEncoder = _PromptEncoder
        we.StaticKVCache = _KV
        try:
            engine = WorldEngine(self.model_id, quant=config.QUANT, device=config.DEVICE)
        finally:
            we.get_ae = orig_ae
            we.WorldModel.from_pretrained = orig_from
            we.PromptEncoder = orig_pe
            we.StaticKVCache = orig_kv
        self._set_bootstrap(
            "load_runtime",
            "wiring sampler",
            "Temperature hook, scheduler, and moving the prompt encoder onto the GPU.",
        )
        self._install_temperature_hook(engine)
        if getattr(engine, "prompt_encoder", None) is not None:
            engine.prompt_encoder.to(engine.device)
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
        self._set_bootstrap(
            "weights_ready",
            "weights ready",
            "Checkpoint is on the GPU. Pick a seed and press Start Dreaming.",
        )

    @property
    def frame_size(self) -> tuple[int, int]:
        return config.frame_size_for(self.model_id)

    @property
    def n_seed_frames(self) -> int:
        if self.engine is not None:
            try:
                t = int(getattr(self.engine.model_cfg, "temporal_compression", 0) or 0)
                if t > 0:
                    return t
            except Exception:
                pass
        return config.temporal_for(self.model_id)

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
        """Keep the checkpoint sigma ladder.

        Rewriting interior sigmas (dream sharpness) caused CUDA device-side
        asserts on both 1B 720p and 1.1 Small after torch.compile. The slider
        stays in prefs/UI; it is not remapped onto Euler until there is a
        per-checkpoint table that is known-safe.
        """
        if engine is None or self._sigma_base is None:
            return
        engine.scheduler_sigmas.copy_(self._sigma_base)

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
        return {"status": "ok", "ready": self.ready, "loading": self.loading, **self._bootstrap_public()}

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
            **self._bootstrap_public(),
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
            self.controls.scroll = int(scroll)
            if abs(int(scroll)) >= 80:
                self._lookout_pending = True
            self.controls.seq += 1
            return self.controls.seq

    def session_world_prompt(self) -> str:
        """Standing prompt plus every active session intention. Not persisted."""
        parts: list[str] = []
        wp = str(self.prefs.get("world_prompt") or "").strip()
        if wp:
            parts.append(wp.rstrip("."))
        for intent in self.intentions:
            if not intent.active:
                continue
            line = intent.raw.strip().rstrip(".")
            if line and line.lower() not in " ".join(parts).lower():
                parts.append(line)
        return ". ".join(parts)[:800]

    def composed_prompt(self) -> str:
        parts: list[str] = []
        standing = self.session_world_prompt()
        note = (self.prompt or str(self.prefs.get("initial_note") or "")).strip()
        if standing:
            parts.append(standing.rstrip("."))
        if note and note.lower() not in standing.lower():
            parts.append(note.rstrip("."))
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
            self.last_prompt_result = "not wired on 1B (prompt_conditioning=null)"
            for intent in self.intentions:
                if intent.kind == "world" and intent.active:
                    intent.status = "submitted"
                    intent.note = (
                        "Active as a standing world rule. This 1B checkpoint has "
                        "prompt_conditioning=null, so set_prompt cannot run. "
                        "Pick Waypoint 1.1 Small for live text, or keep using Klein inpaint. "
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
                    extra = "DiT cross-attention received this session standing prompt."
                    if extra.lower() not in (intent.note or "").lower():
                        intent.note = f"{intent.note} {extra}".strip() if intent.note else extra
            return self.last_prompt_result
        except Exception as exc:
            self.last_prompt_result = f"set_prompt_unavailable: {type(exc).__name__}: {exc}"
            for intent in self.intentions:
                if intent.kind == "world" and intent.active:
                    intent.status = "submitted"
                    intent.note = (
                        f"set_prompt failed ({type(exc).__name__}). Standing rule is kept for Klein; "
                        "the seed image and movement still drive the world."
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
            self.intentions.append(intent)
            if len(self.intentions) > 40:
                self.intentions = self.intentions[-40:]
            if intent.active:
                self._prompt_dirty = True
            if self.session_active and intent.active:
                self._intent_inpaint_queue.append((intent.raw, intent.id))
                if "Klein inpaint" not in (intent.engine_action or ""):
                    intent.engine_action = (
                        f"{intent.engine_action}; Klein inpaint of current still"
                        if intent.engine_action
                        else "Klein inpaint of current still"
                    )
                extra = (
                    "Inpaints the current view, then appends this line to the session "
                    "standing prompt. Earlier intentions stay. Not saved after Stop."
                )
                if extra.lower() not in (intent.note or "").lower():
                    intent.note = f"{intent.note} {extra}".strip() if intent.note else extra
        self._publish_status("intention")
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

        # Texture-lock can still trigger a Klein/open-memory rescue, but must not
        # eat WASD: dirt paths and packed foliage look "locked" to the metric and
        # that used to freeze walking for the rest of the session.

        if prefs.get("steer_move") and not self._freeze_walk:
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
        work = klein_work_size(self.frame_size)
        pil, klein_prompt = authoring.generate_from_text(text, work)
        native_w, native_h = self.frame_size
        if pil.size != (native_w, native_h):
            pil = pil.resize((native_w, native_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        jpeg = buf.getvalue()
        meta = save_painted(jpeg, label=text[:40] or "Klein painted", prompt=text)
        return {"ok": True, "seed": meta, "klein_prompt": klein_prompt}

    def start_session(self, image_bytes: bytes, prompt: str = "") -> dict[str, Any]:
        if not self.ready or self.engine is None:
            return {"ok": False, "error": self.load_error or "model not ready"}
        self._set_bootstrap(
            "encode_seed",
            "encoding start still",
            "Resizing the seed JPEG to the checkpoint’s native frame size.",
        )
        try:
            seed = image_to_seed(image_bytes, self.frame_size, n_frames=self.n_seed_frames)
            preview = encode_jpeg(seed[0].numpy(), 85)
        except Exception as exc:
            self._set_bootstrap("weights_ready", "weights ready", "Seed image failed; checkpoint is still loaded.")
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
            self._intent_inpaint_queue.clear()
            self._open_seeds.clear()
            self._lock_streak = 0
            self._freeze_walk = False
            self._lookout_pending = False
            self._last_scene_event = ""
            self._last_scene = None
            self._last_rescue_at = 0.0
            self._idle_since = None
            self._last_drift_at = None
            self._grace_batches = 0
            self.inpaint_status = "armed" if bool(self.prefs.get("inpaint", False)) else "off"
        self._set_bootstrap(
            "gpu_queue",
            "queued on GPU thread",
            "Session is waiting for the GPU worker (it may still be finishing a previous job).",
        )
        try:
            return self._gpu_call("start", seed)
        except Exception as exc:
            self.session_active = False
            msg = f"{type(exc).__name__}: {exc}"
            self.last_error = msg
            self._set_bootstrap("weights_ready", "weights ready", f"Start failed: {msg}")
            if "CUDA" in msg or "AcceleratorError" in type(exc).__name__:
                self._request_cuda_restart(msg)
            return {"ok": False, "error": msg}

    def _start_on_gpu(self, seed: torch.Tensor) -> dict[str, Any]:
        engine = self.engine
        assert engine is not None
        self._set_bootstrap(
            "reset_engine",
            "resetting world cache",
            "Clearing KV cache and applying the dream-sharpness schedule.",
        )
        engine.reset()
        if self._sched_dirty:
            try:
                self._apply_scheduler(engine)
                self._sched_dirty = False
            except Exception:
                pass
        self._set_bootstrap(
            "seed_compile",
            "seeding world · first compile",
            "append_frame into the DiT. The first call compiles CUDA graphs and can take minutes.",
        )
        decoded = engine.append_frame(seed_for_append(seed, self.n_seed_frames))
        self._set_bootstrap(
            "apply_prompt",
            "applying standing prompt",
            "set_prompt on text-capable checkpoints; a no-op on 1B.",
        )
        self.try_set_prompt()
        frames = as_frame_batch(decoded)
        self.last_frames = frames
        self._note_open_frame(frames)
        self._publish_batch(frames, gen_ms=0.0, kind="seed")
        self.frames_done += int(frames.shape[0])
        from world_engine import CtrlInput

        try:
            self._set_bootstrap(
                "warm_frame",
                "warming first generated frame",
                "First gen_frame. Compiles the live denoising path if this checkpoint has not run yet.",
            )
            warm = engine.gen_frame(ctrl=CtrlInput())
            warm_np = as_frame_batch(warm)
            seed_m = analyze_frame(frames)
            warm_m = analyze_frame(warm_np)
            if warm_m.openness + 0.06 < seed_m.openness or (warm_m.locked and not seed_m.locked):
                engine.reset()
                decoded = engine.append_frame(seed_for_append(seed, self.n_seed_frames))
                frames = as_frame_batch(decoded)
                self.last_frames = frames
            else:
                self.last_frames = warm_np
                self._publish_batch(warm_np, gen_ms=0.0, kind="generated")
                self.frames_done += int(warm_np.shape[0])
                self._note_open_frame(warm_np)
        except Exception:
            pass
        applied = self.last_prompt_result == "set_prompt_ok"
        reason = self.last_prompt_result or "prompt_conditioning is null on this checkpoint"
        self._set_bootstrap(
            "live",
            "streaming",
            "World is seeded. The live generation loop is starting.",
        )
        return {"ok": True, "prompt_applied": applied, "reason": reason}

    def request_seed_reset(self) -> dict[str, Any]:
        if not self.session_active or self.original_seed is None:
            return {"ok": False, "error": "no seed in this session"}
        self._seed_reset_pending = True
        return {"ok": True}

    def _reseed_on_gpu(self) -> None:
        engine = self.engine
        seed = self._open_seeds[-1] if self._open_seeds else self.original_seed
        if engine is None or seed is None:
            return
        self._reseed_from_tensor(seed, kind="seed")
        self._last_scene_event = "open-reseed" if self._open_seeds else "start-reseed"

    def _reseed_from_tensor(self, seed: torch.Tensor, kind: str = "seed") -> None:
        engine = self.engine
        if engine is None:
            return
        engine.reset()
        decoded = engine.append_frame(seed_for_append(seed, self.n_seed_frames))
        self.try_set_prompt()
        frames = as_frame_batch(decoded)
        self.last_frames = frames
        self.view_yaw = 0.0
        self.view_pitch = 0.0
        self.reset_look = False
        self.smooth_mouse = [0.0, 0.0]
        self._freeze_walk = False
        self._lookout_pending = False
        self._lock_streak = 0
        self._publish_batch(frames, gen_ms=0.0, kind=kind)

    def stop_session(self) -> None:
        with self.lock:
            was_active = self.session_active
            self.session_active = False
        self.stop_event.set()
        if was_active:
            self._set_bootstrap(
                "stopping",
                "stopping session",
                "Waiting for the GPU generation loop to unwind.",
            )
        try:
            if self.last_frames is not None:
                self._publish_batch(self.last_frames, gen_ms=0.0, kind="stop")
        except Exception:
            pass
        if threading.current_thread() is not self.gpu_thread:
            self.loop_idle.wait(timeout=180)
        with self.lock:
            self.thread = None
            self.disconnect_deadline = None
            self.intentions = []
            self._intent_inpaint_queue.clear()
            if not bool(self.prefs.get("inpaint", False)):
                self.inpaint_status = "off"
        if self.ready and not self.loading:
            self._set_bootstrap(
                "weights_ready",
                "weights ready",
                "Session stopped. Checkpoint is still on the GPU.",
            )
        else:
            self._publish_status("stop")

    def note_client(self, delta: int) -> None:
        with self.lock:
            self.clients = max(0, self.clients + delta)
            if self.clients == 0 and self.session_active and self.bootstrap_phase == "live":
                self.disconnect_deadline = time.time() + config.DISCONNECT_GRACE_SEC
            elif self.clients > 0:
                self.disconnect_deadline = None
            else:
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
        queued_ids = {iid for _, iid in self._intent_inpaint_queue}
        for intent in pending:
            if intent.id in queued_ids:
                continue
            try:
                graded = color_grade(frames, intent.transform)
                self.pre_transform_stats = {
                    "luminance": float(frames.astype(np.float32).mean()),
                    "green": float(frames[..., 1].astype(np.float32).mean()),
                }
                seed = torch.from_numpy(np.ascontiguousarray(graded))
                self.engine.reset()
                decoded = self.engine.append_frame(seed_for_append(seed, self.n_seed_frames))
                frames = as_frame_batch(decoded)
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
            "session_world_prompt": self.session_world_prompt(),
            "error": self.last_error,
            "model": self.model_id,
            "models": self.models_public(),
            "prefs": dict(prefs),
            "composed_prompt": self.composed_prompt(),
            "prompt_apply": self.last_prompt_result,
            "view": {"yaw": self.view_yaw, "pitch": self.view_pitch, "resetting": self.reset_look},
            "scene": self._scene_public(),
            "native_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "stream_size": {"width": w, "height": h},
            **self._bootstrap_public(),
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

    def _publish_status(self, kind: str = "stats") -> None:
        with self.frame_slot:
            meta = dict(self.latest_meta) if self.latest_meta else {"type": "stats"}
            meta["type"] = "stats"
            meta["kind"] = kind
            meta["session_active"] = self.session_active
            meta["inpaint_status"] = self.inpaint_status
            meta["inpaint_progress"] = self.inpaint_progress
            meta["intentions"] = [self._intent_public(i) for i in self.intentions[-12:]]
            meta["composed_prompt"] = self.composed_prompt()
            meta["session_world_prompt"] = self.session_world_prompt()
            meta["prompt_apply"] = self.last_prompt_result
            meta["scene"] = self._scene_public()
            meta.update(self._bootstrap_public())
            self.latest_meta = meta
            self.frame_slot.notify_all()

    def _publish_inpaint_progress(self, frac: float) -> None:
        self.inpaint_progress = float(max(0.0, min(1.0, frac)))
        with self.frame_slot:
            meta = dict(self.latest_meta) if self.latest_meta else {"type": "stats"}
            meta["inpaint_status"] = self.inpaint_status
            meta["inpaint_progress"] = self.inpaint_progress
            meta["kind"] = "inpaint"
            self.latest_meta = meta
            self.frame_slot.notify_all()

    def _scene_public(self) -> dict[str, Any]:
        m = self._last_scene
        return {
            "lock": None if m is None else round(m.lock, 3),
            "openness": None if m is None else round(m.openness, 3),
            "locked": bool(m.locked) if m is not None else False,
            "open": bool(m.open) if m is not None else False,
            "event": self._last_scene_event,
            "open_memories": len(self._open_seeds),
            "walk_held": 87 in self.controls.buttons or 83 in self.controls.buttons,
            "walk_frozen": self._freeze_walk,
        }

    def _note_open_frame(self, frames: np.ndarray | None) -> None:
        if frames is None or frames.size == 0:
            return
        metrics = analyze_frame(frames)
        self._last_scene = metrics
        if not metrics.open:
            return
        stacked = rgb_to_seed(frames, n_frames=self.n_seed_frames)
        self._open_seeds.append(torch.from_numpy(np.ascontiguousarray(stacked)))

    def _observe_scene(self, frames: np.ndarray) -> None:
        metrics = analyze_frame(frames)
        self._last_scene = metrics
        if metrics.open:
            self._lock_streak = 0
            stacked = rgb_to_seed(frames, n_frames=self.n_seed_frames)
            self._open_seeds.append(torch.from_numpy(np.ascontiguousarray(stacked)))
            self._freeze_walk = False
            return
        if metrics.locked:
            self._lock_streak += 1
            if self._lock_streak >= LOCK_STREAK:
                self._freeze_walk = True
            if metrics.lock >= 0.62 and self.view_pitch < -1.35:
                self._lookout_pending = True
        else:
            self._lock_streak = max(0, self._lock_streak - 1)

    def _rescue_ready(self) -> bool:
        if self._grace_batches < 6:
            return False
        return (time.monotonic() - self._last_rescue_at) >= 8.0

    def _maybe_scene_rescue(self) -> bool:
        if self.stop_event.is_set() or self.last_frames is None:
            return False
        lookout = self._lookout_pending
        locked = self._lock_streak >= LOCK_STREAK and self._freeze_walk
        if lookout:
            self._lookout_pending = False
            last = self._last_scene
            if last is not None and last.open and not last.locked:
                return False
            if not self._rescue_ready():
                return False
            if self._rescue_from_memory("lookout"):
                return True
            return self._klein_scene_cut(DREAM_LOOKOUT_PROMPT, "lookout")
        if locked and self._rescue_ready():
            if self._rescue_from_memory("breath"):
                return True
            return self._klein_scene_cut(DREAM_BREATH_PROMPT, "breath")
        return False

    def _rescue_from_memory(self, event: str) -> bool:
        if not self._open_seeds:
            return False
        seed = self._open_seeds[-1]
        self._reseed_from_tensor(seed, kind="open")
        self._last_scene_event = event
        self._last_rescue_at = time.monotonic()
        self._grace_batches = 0
        return True

    def _klein_scene_cut(self, prompt: str, event: str) -> bool:
        if self.engine is None or self.last_frames is None:
            return False
        try:
            if authoring.pipeline is None:
                self.inpaint_status = "loading"
                self.inpaint_progress = 0.0
                self._publish_inpaint_progress(0.0)
                authoring.load()
            self._scene_inpaint_on_gpu(fallback_prompt=prompt, kind=event)
            self._last_scene_event = event
            self._last_rescue_at = time.monotonic()
            self._grace_batches = 0
            self._inpainted_this_idle = True
            return True
        except Exception as exc:
            self.inpaint_status = "error"
            self.last_error = f"scene-{event}: {type(exc).__name__}: {exc}"
            if self.original_seed is not None:
                self._reseed_from_tensor(self.original_seed, kind="seed")
                self._last_scene_event = "start-fallback"
                return True
            return False

    def _scene_inpaint_on_gpu(
        self,
        modifier: str | None = None,
        fallback_prompt: str | None = None,
        kind: str = "inpaint",
        replace_original: bool = False,
        blend: float | None = None,
        steps: int | None = None,
    ) -> None:
        engine = self.engine
        frames = self.last_frames
        if engine is None or frames is None or frames.size == 0:
            return
        self.inpaint_status = "running"
        self.inpaint_progress = 0.08
        self._publish_inpaint_progress(0.08)
        prev = np.asarray(frames[-1])
        native_h, native_w = int(prev.shape[0]), int(prev.shape[1])
        work = klein_work_size((native_w, native_h))
        src_in = prev
        if (native_w, native_h) != work:
            src_in = cv2.resize(prev[..., :3], work, interpolation=cv2.INTER_AREA)
        pil, _prompt = authoring.refine_frame(
            src_in,
            work,
            user_request=modifier,
            on_progress=self._publish_inpaint_progress,
            fallback_prompt=fallback_prompt,
            steps=steps,
        )
        arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        if arr.shape[1] != native_w or arr.shape[0] != native_h:
            arr = cv2.resize(arr, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
        if blend is not None:
            s = float(max(0.0, min(1.0, blend)))
            src = prev[..., :3].astype(np.float32)
            if src.shape[:2] != arr.shape[:2]:
                src = cv2.resize(src, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_AREA)
            arr = np.clip((1.0 - s) * src + s * arr.astype(np.float32), 0, 255).astype(np.uint8)
        stacked = np.repeat(arr[None, ...], self.n_seed_frames, axis=0)
        seed = torch.from_numpy(np.ascontiguousarray(stacked))
        engine.reset()
        decoded = engine.append_frame(seed_for_append(seed, self.n_seed_frames))
        out = as_frame_batch(decoded)
        self.last_frames = out
        if replace_original:
            self.original_seed = seed.detach().clone()
        self._note_open_frame(out)
        self.try_set_prompt()
        self._freeze_walk = False
        self._lock_streak = 0
        self.inpaint_status = "done"
        self.inpaint_progress = 1.0
        self._publish_batch(out, gen_ms=0.0, kind=kind)
        self.inpaint_progress = 0.0

    def _idle_inpaint_on_gpu(self, modifier: str | None = None) -> None:
        if modifier:
            self._scene_inpaint_on_gpu(
                modifier=modifier,
                kind="inpaint",
                replace_original=False,
            )
            return
        strength = float(self.prefs.get("drift_strength", 0.55))
        steps = int(self.prefs.get("drift_steps", 4))
        self._scene_inpaint_on_gpu(
            fallback_prompt=drift_prompt(strength),
            kind="drift",
            replace_original=False,
            blend=strength,
            steps=steps,
        )

    def _maybe_directed_inpaint(self) -> bool:
        """Send Intention → Klein edit of the current still. Ignores Auto-InPaint."""
        with self.lock:
            if not self._intent_inpaint_queue:
                return False
            text, intent_id = self._intent_inpaint_queue.popleft()
            text = (text or "").strip()
        if not text:
            return False
        if self.stop_event.is_set():
            return False
        if self.last_frames is None or self.engine is None:
            with self.lock:
                self._intent_inpaint_queue.appendleft((text, intent_id))
            return False
        modifier = self.composed_prompt()
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
                intent.engine_action = "Klein inpaint of current still, then session standing prompt"
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
        if self.stop_event.is_set():
            return False
        if not bool(self.prefs.get("inpaint", False)):
            if self.inpaint_status not in {"off", "error"}:
                self.inpaint_status = "off"
            self._idle_batches = 0
            self._inpainted_this_idle = False
            self._last_drift_at = None
            return False
        if self._user_steering():
            self._idle_batches = 0
            self._idle_since = None
            self._inpainted_this_idle = False
            self._last_drift_at = None
            if self.inpaint_status in {"done", "running"}:
                self.inpaint_status = "armed"
            elif self.inpaint_status == "off":
                self.inpaint_status = "armed"
            return False
        now = time.monotonic()
        if self._idle_since is None:
            self._idle_since = now
        self._idle_batches += 1
        delay = float(self.prefs.get("drift_delay", 5.0))
        interval = float(self.prefs.get("drift_interval", 0.0))
        if self._last_drift_at is None:
            if (now - self._idle_since) < delay:
                if self.inpaint_status in {"off", "done"}:
                    self.inpaint_status = "armed"
                return False
        elif interval < 1.0:
            return False
        elif (now - self._last_drift_at) < interval:
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
            self._last_drift_at = time.monotonic()
            return True
        except Exception as exc:
            self.inpaint_status = "error"
            self._inpainted_this_idle = True
            self._last_drift_at = time.monotonic()
            self.last_error = f"inpaint: {type(exc).__name__}: {exc}"
            return False

    def _loop(self) -> None:
        from world_engine import CtrlInput  # noqa: F401  (imported for side-effect-free type use)

        engine = self.engine
        assert engine is not None
        if self.bootstrap_phase != "live":
            self._set_bootstrap(
                "live",
                "streaming",
                "Live DiT loop. Frames are going to the browser.",
            )
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
                if self._maybe_scene_rescue():
                    continue
                if self._maybe_idle_inpaint():
                    continue
                if self.stop_event.is_set():
                    break
                for intent in self.intentions:
                    if intent.status == "received" and intent.kind != "transform" and intent.engine_action != "none":
                        intent.status = "submitted"
                ctrl = self._merged_ctrl()
                t_cycle = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = engine.gen_frame(ctrl=ctrl)
                frames = as_frame_batch(out)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_ms = (time.perf_counter() - t0) * 1000.0
                n = int(frames.shape[0])
                self.last_gen_ms = gen_ms
                self.batches_done += 1
                self._grace_batches += 1
                self.frames_done += n
                self.last_frames = frames
                self._observe_scene(frames)
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
            "session_world_prompt": self.session_world_prompt(),
            "composed_prompt": self.composed_prompt(),
            "prompt_apply": self.last_prompt_result,
            "prefs": dict(self.prefs),
            "view": {"yaw": self.view_yaw, "pitch": self.view_pitch, "resetting": self.reset_look},
            "scene": self._scene_public(),
            "native_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
        }


worker = EngineWorker()
