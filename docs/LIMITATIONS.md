# Supported commands and limitations

Engine: world_engine `b3f1e725b222679a517632918cc78bba0c9fa433`.
Checkpoint default: `Overworld/Waypoint-1.5-1B-360P`.
Both 1B configs ship `prompt_conditioning: null`, so `WorldEngine.set_prompt` cannot run (it raises if the prompt encoder was never created).

## Movement (supported)

| Input | Engine |
|---|---|
| W A S D | `CtrlInput.button` 87, 65, 83, 68 |
| Space | 32 (jump in official samples) |
| Mouse / arrows / nav ball | `CtrlInput.mouse` (dx, dy) velocity |
| Cardinal look buttons | labeled Up/Down/Left/Right; same mouse tensor |
| Reset view | inverse look until tracked yaw/pitch ~ 0; not a geometric camera |
| Nav ball walk | optional WASD from stick angle |
| Click viewport | pointer-lock mouse look |

These are the same control tensors the DiT was built for.

## Experience knobs

See README. Resolution above 360p is **stream upscale**, not a 720p model swap. Temperature is start-noise scale, not an LLM sampler.

## Intentions

| Phrase examples | Kind | What actually happens | When “visually verified” |
|---|---|---|---|
| I can fly / hover | ongoing | Hold Space + slight look-up | never auto-verified |
| Go into that building | one-shot | Hold W for ~18 batches | never auto-verified; no detector |
| Look left/right/up/down | one-shot | Mouse impulse | never auto-verified |
| It is nighttime now | transform | Color-grade last 4 frames darker/blue, `reset()`, `append_frame()`, continue | if mean luminance drops >8% after a few batches |
| Turn this into a forest | transform | Green-grade + re-seed | if green channel rises >5% |
| Daytime / brighter | transform | Brighten + re-seed | if luminance rises >5% |
| Stop flying / clear | meta | Clears ongoing holds | n/a |
| forget that / clear world | world | Drops extra spoken world lines; standing pre-prompt stays | n/a |
| Anything else | world | Standing world rule, composed with the Session pre-prompt | never auto-verified |

Statuses: `received` → `submitted` (engine API called) → `visually_verified` only if the statistic above fires. Unmatched Speak lines stay `submitted` as world rules. `failed` is reserved for engine errors on the transform path.

The Session **standing world prompt** (default: *There is a standard road grid, and buildings.*) is always composed into `set_prompt`. On this 1B checkpoint that call is skipped because `prompt_conditioning` is null; the text is still stored and shown. The seed photograph and movement remain the real world drivers.

## Known limitations

- No permanent geography, physics, or object persistence (by design of the model).
- Backtracking invents new scenery.
- Live language conditioning is **not** implemented by these weights. Typing a sentence does not steer the DiT’s cross-attention.
- Experimental transforms change the seed image, then the world model continues. That is not “the model understood night.”
- Click-to-enter-building cannot bind to a specific facade.
- One session at a time. Disconnect waits `DISCONNECT_GRACE_SEC` (default 8s) then stops generation; the process and loaded weights remain.
- `torch.compile` makes the first batches slow. Diagnostics report generation FPS from `gen_frame` wall time, separately from browser delivered FPS.
- world_engine is GPL-3; the Waypoint weights are Apache-2.0.
- Do not interpret imagery as a clinical or personal truth.
