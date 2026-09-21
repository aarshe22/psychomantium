# Psychomantium

Local, interactive lucid-dream sketch. You upload a photograph, a Waypoint world model generates successive frames, and you steer with WASD / mouse. Typed intentions are **not** live text-to-world on this checkpoint.

This is exploration, not a psychological instrument. Generated images are not evidence about the player.

## Hardware used for this PoC

- NVIDIA RTX PRO 6000 Blackwell Max-Q (~98 GiB)
- Driver 595.84, CUDA 13.2 (host)
- Docker 29.8 + NVIDIA Container Toolkit
- Bound to **127.0.0.1 and 10.1.1.100**

## What the model actually supports

Pinned engine: [`Overworldai/world_engine`](https://github.com/Overworldai/world_engine) commit `b3f1e725b222679a517632918cc78bba0c9fa433` (GPL-3).

Default weights: [`Overworld/Waypoint-1.5-1B-360P`](https://huggingface.co/Overworld/Waypoint-1.5-1B-360P) (Apache-2.0), plus autoencoder `Overworld-Models/taehv1_5`.

From the published `config.yaml`:

| Capability | 1B / 1B-360P |
|---|---|
| Keyboard / mouse `CtrlInput` | yes (`n_buttons: 256`) |
| `gen_frame()` → 4 RGB frames | yes (temporal compression 4) |
| `append_frame()` re-seed | yes |
| `reset()` new session | yes |
| Live `set_prompt()` | **no** (`prompt_conditioning: null`) |

Intentions therefore use controller mapping and an **experimental** color-grade → `reset()` → `append_frame()` path. See `docs/LIMITATIONS.md`.

## Launch

From `/opt/psychomantium`. This host’s outbound HTTP proxy is not reachable at `10.1.1.100` from the Docker bridge, so Compose uses `http://host.docker.internal:3128`. `./backend` is bind-mounted into the GPU container.

```bash
docker compose up --build
```

Wait until `GET http://127.0.0.1:8791/ready` returns 200 (model download + load). `GET /health` only means the HTTP server is up.

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

Weights stay in `data/hf-cache/`. Snapshots stay in `data/outputs/`. Saved experience knobs stay in `data/preferences/preferences.json` (bind-mounted, survives rebuilds).

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

### 720p checkpoint

Both `Overworld/Waypoint-1.5-1B-360P` (640×360) and `Overworld/Waypoint-1.5-1B` (1280×720) are in the local HF cache. Pick one from the **model** dropdown in the top bar. Switching unloads the current engine and loads the other on the GPU (the dream session stops). Stream resolution is set to that checkpoint’s native height.

## Controls

- **W** walk forward · **Z** walk back
- **← →** turn left / right · **↑ ↓** look up / down
- **R** reset orientation to the horizon
- **Space** jump · click the viewport for pointer-lock mouse look
- Left rail: accordion of session, intention, navigate, knobs, diagnostics. **Pin** keeps it open; **Hide** collapses it to the left.
- Movement keys are ignored while the intention/prompt fields are focused
- **Enter** in the intention field submits
- Mouse scroll wheel is ignored (not sent as `CtrlInput.scroll_wheel`)

## Experience knobs

Sliders apply live (not only after save). **Save preferences** writes `data/preferences/preferences.json` on the host.

| Knob | What it actually does |
|---|---|
| Output resolution | Resizes generated frames for the JPEG stream (360–720). Native generation is 640×360 or 1280×720 from the selected checkpoint. |
| Inference temperature | Multiplies the `torch.randn` start-noise inside `gen_frame`. 1.0 is the engine default. |
| Look sensitivity | Scales mouse / ball / arrow look velocity |
| Stream JPEG quality | Encode quality only |
| Idle wander | Random look when you are not steering |
| Motion smoothing | Exponential blend on look |
| Dream sharpness | Remaps the 4-step noise schedule (same step count; compiled graph stays valid) |

**Standing world prompt** (Session accordion, default *There is a standard road grid, and buildings.*) is persisted with preferences. Spoken unmatched lines become extra world rules. On this 1B checkpoint `set_prompt` is not wired into the DiT; the seed image and movement still drive the world. Say `forget that` to drop extra spoken rules.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `/health` ok, `/ready` 503 | model still downloading/loading; `docker compose logs backend` |
| CUDA / GPU errors | another process using the GPU is fine if VRAM remains; do not kill unrelated jobs |
| Hugging Face download stalls | this host uses an HTTP proxy; Compose forwards `HTTP_PROXY`/`HTTPS_PROXY` |
| First batches very slow | `torch.compile` warmup; generation FPS in diagnostics ignores UI interpolation |
| Intention “failed” | transform path error; unmatched Speak lines are standing world rules, not failures |
| After Stop, UI is idle but container lives | expected. Enter dream again with a new/same image |

## Layout

- `backend/` FastAPI + GPU worker
- `frontend/` React + Vite
- `docker/` Dockerfiles
- `data/hf-cache/` model cache (not source)
- `data/outputs/` snapshots / probe movies
- `docs/` limitations and measured results
