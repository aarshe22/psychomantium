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
| Reset seed (U) | `reset()` + `append_frame(original seed)` |
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
| Anything else | world | Klein inpaint of the current still (spoken modifier + standing prompt) | n/a |

Statuses: `received` → `submitted` (engine API called) → `visually_verified` only if the statistic above fires. Unmatched Speak lines stay `submitted` as world rules and also Klein-inpaint the current still. `failed` is reserved for engine errors on the transform or Klein path.

The Session **standing world prompt** is stored and shown. On this 1B checkpoint it is **not** DiT conditioning (`prompt_conditioning=null`). [Biome](https://github.com/Overworldai/Biome) uses the same honesty: its prompt notification resets the engine and ignores the text. Visual quality is the **start frame** (gallery or upload) plus movement. A warmup `gen_frame` runs after the seed so the first walk is not a compile hitch.

## Scene authoring

Biome’s “custom prompting” is a **second model**: Gemma VLM + FLUX.2-klein-4B write a new first-person JPEG, then Waypoint continues from that seed. Psychomantium can paint a start frame with Klein from the standing prompt (`POST /api/seeds/paint`). **Speak** always Klein-inpaints the current still with that sentence as the modifier, whether or not **Auto-InPaint** is on. Auto-InPaint only refines detail while you stand still. Color-grade Speak lines (night/forest/day) still reseed if Klein is skipped; when Klein runs they are applied as the inpaint modifier instead.

## Known limitations

- No permanent geography, physics, or object persistence (by design of the model).
- Backtracking invents new scenery.
- Live language conditioning is **not** implemented by these weights. Typing a sentence does not steer the DiT’s cross-attention. Biome does not `set_prompt` on 1B either.
- The seed image is the world prior. Gallery stills are royalty-free eye-level urban and rural photographs (Wikimedia Commons / StockSnap, cached on the host) or Klein-painted frames, plus user upload. Overworld FPS stills are not used: they show hands and weapons. Schematic drawings are out of distribution.
- Experimental transforms change the seed image, then the world model continues. That is not “the model understood night.”
- The browser spreads each 4-frame batch across the last batch interval (EMA). That is display pacing, not extra inference.
- `torch.compile` makes the first batches slow; a warmup `gen_frame` after seed absorbs some of that. Diagnostics report generation FPS from `gen_frame` wall time, separately from paced delivered FPS.
- Click-to-enter-building cannot bind to a specific facade.
- world_engine is GPL-3; the Waypoint weights are Apache-2.0.
- Do not interpret imagery as a clinical or personal truth.
