"""Gemma VLM + FLUX.2 Klein 4B: paint a first-person JPEG, then Waypoint continues.

Waypoint-1.5-1B has prompt_conditioning=null. Language changes the world by
rewriting the seed image, not by DiT cross-attention.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from . import config

log = logging.getLogger("psychomantium.authoring")

KLEIN_ID = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_GGUF_REPO = "unsloth/FLUX.2-klein-4B-GGUF"
KLEIN_GGUF_FILE = "flux-2-klein-4b-Q8_0.gguf"
VLM_REPO = "unsloth/gemma-4-E4B-it-GGUF"
VLM_FILE = "gemma-4-E4B-it-UD-Q4_K_XL.gguf"
VLM_MMPROJ = "mmproj-F16.gguf"
KLEIN_STEPS = 4
VLM_CTX = 4096
VLM_MAX_TOKENS = 1024
VLM_RETRIES = 3
VLM_IMAGE_MAX = 384

VLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_edit_instruction",
            "description": "Submit the final instruction for the image model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Instruction or full scene description for FLUX.2 Klein.",
                    }
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_request",
            "description": "Reject a request that is only unsafe content with no salvageable intent.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

POLICY = (
    "Do not depict sexual content, nudity, or anyone who is or appears under 18. "
    "Replace named copyrighted characters and brands with generic lookalikes. "
    "If the request is only disallowed content, call reject_request."
)

EDIT_SYSTEM = (
    "You write short image-edit instructions for FLUX.2 Klein. "
    "The editor gets a first-person reference frame plus your instruction. "
    "Describe what to change, not the whole scene. Add elements unless told to replace. "
    "Scene objects sit in the world; handheld items go in a right hand at the bottom-right, FPS-style. "
    f"{POLICY} "
    "End with 'Keep everything else unchanged.' "
    "Think briefly, then call submit_edit_instruction."
)

GENERATE_SYSTEM = (
    "You write a detailed text-to-image prompt for FLUX.2 Klein. "
    "The image is a first-person starting frame for an explorable world. "
    "Describe setting, lighting, atmosphere, and a handheld object in the bottom-right. "
    f"{POLICY} "
    "Call submit_edit_instruction with the full prompt."
)


class AuthoringRejected(RuntimeError):
    pass


class AuthoringNotReady(RuntimeError):
    pass


@dataclass
class AuthoringStatus:
    enabled: bool
    ready: bool
    loading: bool
    error: Optional[str]
    klein_id: str = KLEIN_ID
    vlm_id: str = f"{VLM_REPO}/{VLM_FILE}"
    transformer: str = f"{KLEIN_GGUF_REPO}/{KLEIN_GGUF_FILE}"


def _pil_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _extract_instruction(message: dict[str, Any]) -> str:
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name")
        if name == "reject_request":
            raise AuthoringRejected("request rejected by VLM policy")
        if name == "submit_edit_instruction":
            raw = fn.get("arguments") or call.get("arguments") or {}
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {"instruction": raw}
            inst = str((raw or {}).get("instruction") or "").strip()
            if inst:
                return inst
    content = str(message.get("content") or "")
    if "reject_request" in content and "submit_edit_instruction" not in content:
        raise AuthoringRejected("request rejected by VLM policy")
    match = re.search(
        r"submit_edit_instruction[^\n]*instruction[\"']?\s*[:=]\s*[\"'](.+?)[\"']",
        content,
        re.I | re.S,
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.S)
    if match:
        block = match.group(1)
        if "reject_request" in block:
            raise AuthoringRejected("request rejected by VLM policy")
        inst_m = re.search(r"instruction[\"']?\s*[:=]\s*[\"'](.+?)[\"']", block, re.S)
        if inst_m:
            return inst_m.group(1).strip()
        try:
            data = json.loads(block)
            inst = str((data.get("arguments") or data).get("instruction") or "").strip()
            if inst:
                return inst
        except Exception:
            pass
    raise ValueError("VLM did not return submit_edit_instruction")


class SceneAuthoring:
    def __init__(self) -> None:
        self.pipeline = None
        self.vlm = None
        self.ready = False
        self.loading = False
        self.error: Optional[str] = None
        self.load_seconds: Optional[float] = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": config.SCENE_AUTHORING,
            "ready": self.ready,
            "loading": self.loading,
            "error": self.error,
            "load_seconds": self.load_seconds,
            "klein": KLEIN_ID,
            "transformer_gguf": f"{KLEIN_GGUF_REPO}/{KLEIN_GGUF_FILE}",
            "vlm": f"{VLM_REPO}/{VLM_FILE}",
            "steps": KLEIN_STEPS,
        }

    def load(self) -> None:
        if not config.SCENE_AUTHORING:
            self.error = "SCENE_AUTHORING disabled"
            return
        if self.pipeline is not None:
            self.ready = True
            return
        if self.loading:
            return
        self.loading = True
        self.error = None
        import time

        t0 = time.perf_counter()
        try:
            self._load_klein()
            try:
                self._load_vlm()
            except Exception as exc:
                log.warning("Gemma VLM skipped (Klein idle inpaint still works): %s", exc)
            self.ready = self.pipeline is not None
            self.load_seconds = time.perf_counter() - t0
            if not self.ready:
                raise RuntimeError("Klein pipeline did not load")
            log.info("scene authoring loaded in %.1fs (vlm=%s)", self.load_seconds, self.vlm is not None)
        except Exception as exc:
            self.ready = False
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("scene authoring load failed")
            self.unload()
            raise
        finally:
            self.loading = False

    def unload(self) -> None:
        if self.vlm is not None:
            try:
                self.vlm.close()
            except Exception:
                pass
        self.vlm = None
        self.pipeline = None
        self.ready = False
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_vlm(self) -> None:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Gemma4ChatHandler

        model_path = hf_hub_download(repo_id=VLM_REPO, filename=VLM_FILE)
        mmproj_path = hf_hub_download(repo_id=VLM_REPO, filename=VLM_MMPROJ)
        handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False)
        self.vlm = Llama(
            model_path=model_path,
            chat_handler=handler,
            n_ctx=VLM_CTX,
            n_gpu_layers=-1,
            verbose=False,
        )

    def _load_klein(self) -> None:
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        gguf_path = hf_hub_download(repo_id=KLEIN_GGUF_REPO, filename=KLEIN_GGUF_FILE)
        transformer = Flux2Transformer2DModel.from_single_file(
            gguf_path,
            config=KLEIN_ID,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        text_encoder = AutoModelForCausalLM.from_pretrained(
            KLEIN_ID,
            subfolder="text_encoder",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            torch_dtype=torch.bfloat16,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            KLEIN_ID,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
        )
        pipe.to(config.DEVICE)
        pipe.set_progress_bar_config(disable=True)
        self.pipeline = pipe

    def _vlm_instruction(self, messages: list[dict[str, Any]]) -> str:
        if self.vlm is None:
            raise AuthoringNotReady("VLM not loaded")
        last: Exception | None = None
        for _ in range(VLM_RETRIES):
            result = self.vlm.create_chat_completion(
                messages=messages,
                tools=VLM_TOOLS,
                max_tokens=VLM_MAX_TOKENS,
                temperature=1.0,
                top_p=0.95,
            )
            message = result["choices"][0]["message"]
            try:
                return _extract_instruction(message)
            except AuthoringRejected:
                raise
            except Exception as exc:
                last = exc
        raise RuntimeError(f"VLM did not produce an instruction: {last}")

    def prompt_for_edit(self, frame: Image.Image, user_request: str) -> str:
        thumb = frame.copy()
        thumb.thumbnail((VLM_IMAGE_MAX, VLM_IMAGE_MAX), Image.Resampling.LANCZOS)
        messages = [
            {"role": "system", "content": EDIT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _pil_data_uri(thumb)}},
                    {
                        "type": "text",
                        "text": (
                            f'The player asked: "{user_request}"\n'
                            "Look at the frame and submit a specific edit instruction."
                        ),
                    },
                ],
            },
        ]
        return self._vlm_instruction(messages)

    def prompt_for_generate(self, user_request: str) -> str:
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f'The player wants this world: "{user_request}"\n'
                    "Submit a detailed first-person scene prompt."
                ),
            },
        ]
        return self._vlm_instruction(messages)

    @staticmethod
    def _align(h: int, w: int) -> tuple[int, int]:
        return max(16, h // 16 * 16), max(16, w // 16 * 16)

    def run_klein(self, image: Image.Image, prompt: str, height: int, width: int) -> Image.Image:
        if self.pipeline is None:
            raise AuthoringNotReady("Klein pipeline not loaded")
        out = self.pipeline(
            image=image,
            prompt=prompt,
            num_inference_steps=KLEIN_STEPS,
            height=height,
            width=width,
        )
        return out.images[0]

    def generate_from_text(self, user_request: str, size_wh: tuple[int, int]) -> tuple[Image.Image, str]:
        """Paint a first-person seed with Klein. Uses the user's words directly (no VLM)."""
        if self.pipeline is None:
            raise AuthoringNotReady(self.error or "Klein pipeline not loaded")
        w, h = size_wh
        th, tw = self._align(h, w)
        text = (user_request or "").strip() or "an explorable first-person world"
        prompt = (
            "Photoreal first-person screenshot, eye-level, 16:9, natural lighting, "
            "detailed materials, coherent environment. "
            f"{text}. "
            "A handheld object in the bottom-right of the frame, FPS view. "
            "No text overlay, no UI chrome."
        )
        blank = Image.new("RGB", (tw, th), (255, 255, 255))
        result = self.run_klein(blank, prompt, th, tw)
        return result.resize((w, h), Image.Resampling.LANCZOS), prompt

    def generate(self, user_request: str, size_wh: tuple[int, int]) -> tuple[Image.Image, str]:
        if not self.ready:
            raise AuthoringNotReady(self.error or "authoring not ready")
        w, h = size_wh
        th, tw = self._align(h, w)
        klein_prompt = self.prompt_for_generate(user_request)
        blank = Image.new("RGB", (tw, th), (255, 255, 255))
        result = self.run_klein(blank, klein_prompt, th, tw)
        return result.resize((w, h), Image.Resampling.LANCZOS), klein_prompt

    def inpaint(self, frame: np.ndarray, user_request: str, size_wh: tuple[int, int]) -> tuple[Image.Image, str]:
        if not self.ready:
            raise AuthoringNotReady(self.error or "authoring not ready")
        w, h = size_wh
        pil = Image.fromarray(frame).convert("RGB")
        klein_prompt = self.prompt_for_edit(pil, user_request)
        th, tw = self._align(pil.height, pil.width)
        resized = pil.resize((tw, th), Image.Resampling.LANCZOS)
        result = self.run_klein(resized, klein_prompt, th, tw)
        return result.resize((w, h), Image.Resampling.LANCZOS), klein_prompt

    DETAIL_PROMPT = (
        "Increase photorealistic detail, texture, materials, and lighting of this first-person view. "
        "Keep the same camera angle, composition, objects, and layout. "
        "Do not add or remove subjects. Keep everything else unchanged."
    )

    def refine_frame(self, frame: np.ndarray, size_wh: tuple[int, int]) -> tuple[Image.Image, str]:
        """Klein edit of the current view. No VLM — fixed detail prompt for idle inpaint."""
        if self.pipeline is None:
            raise AuthoringNotReady(self.error or "Klein pipeline not loaded")
        w, h = size_wh
        pil = Image.fromarray(np.asarray(frame)).convert("RGB")
        th, tw = self._align(pil.height, pil.width)
        resized = pil.resize((tw, th), Image.Resampling.LANCZOS)
        prompt = self.DETAIL_PROMPT
        result = self.run_klein(resized, prompt, th, tw)
        return result.resize((w, h), Image.Resampling.LANCZOS), prompt


authoring = SceneAuthoring()
