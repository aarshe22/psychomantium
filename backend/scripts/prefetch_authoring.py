#!/usr/bin/env python3
"""Prefetch Gemma + FLUX.2 Klein into HF_HOME.

Prefer the host entrypoint: ``./scripts/pre-cache-models`` (also pulls Waypoint).
This helper remains for ``docker compose exec backend python scripts/prefetch_authoring.py``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# In the GPU image, /app is backend/; the host script lives at repo scripts/.
candidates = [
    Path("/opt/psychomantium-scripts/pre_cache_models.py"),
    Path(__file__).resolve().parents[2] / "scripts" / "pre_cache_models.py",
]
for path in candidates:
    if path.is_file():
        sys.argv = [str(path), *[a for a in sys.argv[1:] if a]]
        runpy.run_path(str(path), run_name="__main__")
        raise SystemExit(0)

from huggingface_hub import hf_hub_download, snapshot_download

print("Gemma 4 E4B GGUF + mmproj...", flush=True)
hf_hub_download("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-UD-Q4_K_XL.gguf")
hf_hub_download("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf")
print("Klein Q8 transformer GGUF...", flush=True)
hf_hub_download("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf")
print("Klein pipeline (text encoder, VAE, tokenizer; skip duplicate DiT files)...", flush=True)
snapshot_download(
    "black-forest-labs/FLUX.2-klein-4B",
    ignore_patterns=[
        "flux-2-klein-4b.safetensors",
        "transformer/diffusion_pytorch_model.safetensors",
        "*.jpg",
    ],
)
print("authoring weights ready", flush=True)
