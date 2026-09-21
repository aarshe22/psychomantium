# Psychomantium

Local, interactive lucid-dream sketch. Pick a first-person start frame (or upload a photograph). A Waypoint world model generates successive frames; you steer with WASD / mouse. On **Waypoint 1.5 1B**, typed text is not live DiT conditioning. On **Waypoint 1.1 Small**, the standing prompt is sent with `set_prompt` (UMT5 cross-attention). Same core loop as Overworld’s [Biome](https://github.com/Overworldai/Biome) client: **seed pixels + `CtrlInput`**, plus optional text on 1.1 Small.

This is exploration, not a psychological instrument. Generated images are not evidence about the player.

## Hardware used for this PoC

- NVIDIA RTX PRO 6000 Blackwell Max-Q (~98 GiB)
- Driver 595.84, CUDA 13.2 (host)
- Docker 29.8 + NVIDIA Container Toolkit
- Bound to **127.0.0.1 and 10.1.1.100**

## What the model actually supports

Pinned engine: [`Overworldai/world_engine`](https://github.com/Overworldai/world_engine) commit `b3f1e725b222679a517632918cc78bba0c9fa433` (GPL-3).

Default weights: [`Overworld/Waypoint-1.5-1B-360P`](https://huggingface.co/Overworld/Waypoint-1.5-1B-360P) (Apache-2.0), plus autoencoder `Overworld-Models/taehv1_5`. Text steering uses [`Overworld/Waypoint-1.1-Small`](https://huggingface.co/Overworld/Waypoint-1.1-Small) (~2.3B) with `OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan` and `google/umt5-xl`.

From the published configs:

| Capability | 1B / 1B-360P | 1.1 Small |
|---|---|---|
| Keyboard / mouse `CtrlInput` | yes | yes |
| `gen_frame()` RGB | 4 frames (TAEHV) | 1 frame (OWL VAE) |
| `append_frame()` re-seed | yes | yes |
| `reset()` new session | yes | yes |
| Live `set_prompt()` | **no** (`prompt_conditioning: null`) | **yes** (`cross_attention`) |

Intentions therefore use controller mapping and an **experimental** color-grade → `reset()` → `append_frame()` path. See `docs/LIMITATIONS.md`.

## Launch

From `/opt/psychomantium`. This host’s outbound HTTP proxy is not reachable at `10.1.1.100` from the Docker bridge, so Compose uses `http://host.docker.internal:3128`. `./backend` is bind-mounted into the GPU container.

```bash
docker compose up --build
```

Wait until `GET http://127.0.0.1:8791/ready` JSON has `"ready": true` (model download + load). `GET /health` only means the HTTP server is up. `/ready` stays HTTP 200 while loading so the UI does not flood the browser console.

- UI: http://127.0.0.1:8790 or http://10.1.1.100:8790
- API: http://127.0.0.1:8791 or http://10.1.1.100:8791

### Remote access (SSH tunnel)

Do not publish the GPU service on a public interface.

```bash
ssh -N -L 8790:127.0.0.1:8790 user@this-host
```

Then open http://127.0.0.1:8790 on your laptop.

### Shutdown

```bash
docker compose down
```

### Model weight cache (survives rebuilds)

Weights are **not** stored in the image. Host directories are bind-mounted; `docker compose build`, `down`, and container recreate leave them in place. Prefetch with `docker compose exec backend python scripts/prefetch_authoring.py`.

| Host path | Container path | What lives there |
|---|---|---|
| `data/hf-cache/` | `/data/hf-cache` | Hugging Face hub (`HF_HOME` / `HF_HUB_CACHE`): Waypoint 1B 360p+720p, TAEHV, Waypoint 1.1 Small, OWL VAE, UMT5-XL, Gemma 4 E4B GGUF, FLUX.2-klein-4B |
| `data/torch-cache/` | `/data/torch-cache` | `torch.compile` inductor/triton + Torch Hub |
| `data/xdg-cache/` | `/data/xdg-cache` and `/root/.cache` | Catch-all if a library ignores `HF_HOME` |
| `data/outputs/` | `/data/outputs` | Snapshots / probe movies |
| `data/seeds/` | `/data/seeds` | CC0 path/road starter JPEGs and Klein-painted seeds (`painted/`) that survive container rebuilds |

Do not delete `data/hf-cache/` unless you intend to re-download tens of gigabytes.

### Logs

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

### Isolated inference probe (no UI)

```bash
docker compose --profile probe run --rm probe
```

Writes `data/outputs/probe/probe_report.json` and `probe_sample.mp4`.

### Checkpoints

`Overworld/Waypoint-1.5-1B-360P` (640×360), `Overworld/Waypoint-1.5-1B` (1280×720), and `Overworld/Waypoint-1.1-Small` (640×360, live `set_prompt`) are in the model dropdown. Switching unloads the current engine and loads the other on the GPU (the dream session stops). Stream resolution is set to that checkpoint’s native height. 1.1 Small also pulls `google/umt5-xl` the first time.

## Controls

- **W** walk forward · **Z** walk back
- **← →** turn left / right · **↑ ↓** look up / down
- **R** reset orientation to the horizon
- **U** reset to the last open view (`engine.reset()` + `append_frame`), or the original seed if none
- **Space** jump · click the viewport for pointer-lock mouse look
- Session: upload a photograph (best prior), CC0 empty path/road stills, or type a one-liner and FLUX.2 Klein caches a new seed under `data/seeds/painted/` (trash can deletes those)
- Left rail: accordion of session, intention, navigate, knobs, diagnostics. **Pin** keeps it open; **Hide** unpins and collapses it.
- Movement keys are ignored while the intention/prompt fields are focused
- **Enter** in the intention field submits; that line inpaints the current still (Klein), even if Dream drift is off
- Mouse wheel look-out from a close surface reseeds the last open view (not engine zoom)

## Experience knobs

Sliders apply live (not only after save). **Save preferences** writes `data/preferences/preferences.json` on the host.

| Knob | What it actually does |
|---|---|
| Output resolution | Resizes generated frames for the JPEG stream, capped at the loaded checkpoint’s height. |
| Inference temperature | Multiplies the `torch.randn` start-noise inside `gen_frame`. Default 0.40 (engine 1.0 is hotter). |
| Look sensitivity | Scales mouse / ball / arrow look velocity |
| Stream JPEG quality | Encode quality only (default 86). Low values add mush the KV cache copies. |
| Idle wander | Random look when you are not steering |
| Motion smoothing | Exponential blend on look |
| Dream sharpness | Remaps the 4-step noise schedule (same step count; compiled graph stays valid) |

**Standing world prompt** is persisted with preferences. On 1B `set_prompt` is not wired into the DiT. On **1.1 Small** the composed standing prompt is sent with `set_prompt` at session start and whenever it changes. **Send Intention** still Klein-inpaints the current still. **Dream drift** is idle-only: stand still and Klein continues into an adjacent place (not a detail-sharpen of one patch). Walking into a repeating surface pauses forward motion and reseeds the last open view, or Klein-opens the sky. **Paint seed from prompt** paints a start frame before you enter.

The viewport paces the 4 JPEG subframes from each `gen_frame` across the batch interval (EMA), instead of flashing all four at once. Generation is uncapped by default; click the FPS badge to lock 30.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `/health` ok, `/ready` has `"ready": false` | model still downloading/loading; `docker compose logs backend` |
| CUDA / GPU errors | another process using the GPU is fine if VRAM remains; do not kill unrelated jobs |
| Hugging Face download stalls | this host uses an HTTP proxy; Compose forwards `HTTP_PROXY`/`HTTPS_PROXY` |
| First batches very slow | `torch.compile` warmup; generation FPS in diagnostics ignores UI interpolation |
| Intention “failed” | transform path error; unmatched Speak lines are standing world rules, not failures |
| After Stop, UI is idle but container lives | expected. Enter dream again with a new/same image |

## Layout

- `backend/` FastAPI + GPU worker
- `frontend/` React + Vite
- `docker/` Dockerfiles
- `data/hf-cache/` Hugging Face weight cache (bind-mounted, not in git)
- `data/torch-cache/` compile caches
- `data/xdg-cache/` extra `~/.cache` bind-mount
- `data/outputs/` snapshots / probe movies
- `docs/` limitations and measured results
