# Benchmark / results

Measured on 2026-09-21 from `data/outputs/probe/probe_report.json` (written by `docker compose --profile probe run --rm probe`) and `nvidia-smi`. Do not treat other logs as this table.

## Model and software

| Item | Value |
|---|---|
| Weights | `Overworld/Waypoint-1.5-1B-360P` |
| Engine SHA | `b3f1e725b222679a517632918cc78bba0c9fa433` |
| Quant | `null` (none) |
| Frame size | 640×360 |
| Prompt conditioning | `null` |
| Torch (container) | `2.11.0+cu128` |
| CUDA (container / torch) | `12.8` |

## Hardware (`nvidia-smi`)

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Driver | 595.84 |
| Host CUDA (driver reports) | 13.2 |
| Framebuffer | 97887 MiB |
| Used after probe (idle, other jobs present) | 3296 MiB, GPU util 0% |
| Used with backend loaded, session idle | 7393 MiB, GPU util 0% |

Unrelated host process PID 4309 (`python3`) was left running and accounted for ~3282 MiB of the idle used memory.

## Probe method

From `probe_report.json`:

> Wall-clock time around WorldEngine.gen_frame() only. Each call returns 4 decoded RGB frames. First two batches excluded as torch.compile warmup. FPS is generated frames / elapsed, not display interpolation.

Sequence: seed `append_frame`, then W / mouse / jump / idle batches. Night path color-grades the last frames, `reset()`, `append_frame()`, then more W batches.

## Probe timings

| Metric | Value |
|---|---|
| Startup (construct `WorldEngine`) | 78.49181345500983 s |
| First `append_frame` (includes compile) | 73.40552759403363 s |
| Warmup batches excluded | 2 |
| Steady generation FPS | 274.93622167797974 |
| Steady batch time (mean) | 14.548828726849305 ms |
| Torch peak allocated after load | 7086.888671875 MiB |
| Torch peak allocated (probe) | 7086.888671875 MiB |
| Night pre luminance | 71.14205169677734 |
| Night post luminance | 42.76904296875 |
| Night luminance dropped | true |

## Sample

Host path: [`data/outputs/probe/probe_sample.mp4`](../data/outputs/probe/probe_sample.mp4)

Also written: `data/outputs/probe/probe_last.jpg` and `data/outputs/probe/probe_report.json`.

## Later live-session note (synchronized)

After adding `torch.cuda.synchronize()` and including the GPU→CPU copy in the worker timer, an HTTP e2e session (synthetic 640×360 seed, 140 batches) reported **186.8 generation FPS** via `/api/status` (`frames` 564). That is still model generation throughput, not browser delivered FPS. The unsynchronized probe figure above can overstate GPU completion time.
