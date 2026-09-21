#!/usr/bin/env python3
"""Download Psychomantium weights into the host HF cache that Docker bind-mounts.

Idempotent: huggingface_hub resumes and skips files that are already complete.
Writes under HF_HOME (default: <repo>/data/hf-cache), which compose maps to
/data/hf-cache in the GPU container.

Usage (from repo root):
  ./scripts/pre-cache-models
  python3 scripts/pre_cache_models.py --skip-authoring
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WAYPOINT_REPOS = (
    "Overworld/Waypoint-1.5-1B-360P",
    "Overworld/Waypoint-1.5-1B",
    "Overworld-Models/taehv1_5",
    "Overworld/Waypoint-1.1-Small",
    "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan",
    "google/umt5-xl",
)

AUTHORING_FILES = (
    ("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-UD-Q4_K_XL.gguf"),
    ("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
    ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
)

KLEIN_PIPELINE = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_IGNORE = (
    "flux-2-klein-4b.safetensors",
    "transformer/diffusion_pytorch_model.safetensors",
    "*.jpg",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def apply_cache_env(hf_home: Path) -> None:
    hf_home.mkdir(parents=True, exist_ok=True)
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ["DIFFUSERS_CACHE"] = str(hf_home / "diffusers")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
    for name in ("transformers", "diffusers", "datasets", "xet"):
        (hf_home / name).mkdir(parents=True, exist_ok=True)


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "  python3 -m pip install huggingface_hub\n"
            "or run ./scripts/pre-cache-models (it can use Docker)."
        ) from exc
    return hf_hub_download, snapshot_download


def prefetch(*, authoring: bool) -> None:
    hf_hub_download, snapshot_download = _hub()
    token = _token()
    kwargs: dict = {"token": token}

    for repo in WAYPOINT_REPOS:
        print(f"snapshot {repo} ...", flush=True)
        path = snapshot_download(repo_id=repo, **kwargs)
        print(f"  -> {path}", flush=True)

    if not authoring:
        print("skipping Klein / Gemma authoring weights", flush=True)
        return

    for repo, filename in AUTHORING_FILES:
        print(f"file {repo}/{filename} ...", flush=True)
        path = hf_hub_download(repo_id=repo, filename=filename, **kwargs)
        print(f"  -> {path}", flush=True)

    print(f"snapshot {KLEIN_PIPELINE} (skip unused full DiT shards) ...", flush=True)
    path = snapshot_download(repo_id=KLEIN_PIPELINE, ignore_patterns=list(KLEIN_IGNORE), **kwargs)
    print(f"  -> {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Host cache directory (default: <repo>/data/hf-cache)",
    )
    parser.add_argument(
        "--skip-authoring",
        action="store_true",
        help="Skip Gemma + FLUX.2 Klein (Waypoint + TAEHV only)",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    hf_home = (args.hf_home or (root / "data" / "hf-cache")).resolve()
    apply_cache_env(hf_home)
    (root / "data" / "torch-cache").mkdir(parents=True, exist_ok=True)
    (root / "data" / "xdg-cache").mkdir(parents=True, exist_ok=True)
    (root / "data" / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "seeds").mkdir(parents=True, exist_ok=True)

    print(f"HF_HOME={hf_home}", flush=True)
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
             or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "")
    print(f"proxy={'set' if proxy.strip() else 'none'}", flush=True)
    prefetch(authoring=not args.skip_authoring)
    print("pre-cache complete (compose bind-mounts data/hf-cache -> /data/hf-cache)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
