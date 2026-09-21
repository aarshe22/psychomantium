#!/usr/bin/env python3
"""Prefetch Gemma + FLUX.2 Klein weights into HF_HOME (skip unused full DiT shards)."""

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
