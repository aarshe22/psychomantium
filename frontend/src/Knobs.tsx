export type Prefs = {
  resolution: number;
  temperature: number;
  look_sensitivity: number;
  jpeg_quality: number;
  wander: number;
  motion_smoothing: number;
  dream_sharpness: number;
  steer_move: boolean;
  initial_note: string;
  world_prompt: string;
};

export const PREF_FIELDS: {
  key: keyof Prefs;
  label: string;
  min: number;
  max: number;
  step: number;
  hint: string;
}[] = [
  {
    key: "resolution",
    label: "Output resolution",
    min: 360,
    max: 720,
    step: 60,
    hint: "Native generation is 640×360. Higher values upscale frames for the stream, not a second world model.",
  },
  {
    key: "temperature",
    label: "Inference temperature",
    min: 0.4,
    max: 1.8,
    step: 0.05,
    hint: "Scales the diffusion start-noise. 1.0 is the engine default. Higher is wilder; lower is stickier.",
  },
  {
    key: "look_sensitivity",
    label: "Look sensitivity",
    min: 0.25,
    max: 2.5,
    step: 0.05,
    hint: "Mouse, arrows, and nav-ball look velocity.",
  },
  {
    key: "jpeg_quality",
    label: "Stream JPEG quality",
    min: 40,
    max: 95,
    step: 1,
    hint: "Browser encode quality. Does not change the model.",
  },
  {
    key: "wander",
    label: "Idle wander",
    min: 0,
    max: 0.3,
    step: 0.01,
    hint: "Small random look when you are not steering.",
  },
  {
    key: "motion_smoothing",
    label: "Motion smoothing",
    min: 0,
    max: 0.85,
    step: 0.05,
    hint: "Blend toward new look/move inputs so turns are less twitchy.",
  },
  {
    key: "dream_sharpness",
    label: "Dream sharpness",
    min: 0,
    max: 1,
    step: 0.05,
    hint: "Remaps the 4-step noise schedule. Higher jumps toward the end of the ODE faster.",
  },
];

type Props = {
  prefs: Prefs;
  saved: boolean;
  onChange: (next: Prefs) => void;
  onSave: () => void;
};

export default function Knobs({ prefs, saved, onChange, onSave }: Props) {
  return (
    <section className="knobs">
      <header>
        <button type="button" onClick={onSave}>
          {saved ? "Preferences saved" : "Save preferences"}
        </button>
      </header>
      {PREF_FIELDS.map((f) => (
        <label key={f.key} className="knob" title={f.hint}>
          <span>
            {f.label}
            <strong>
              {f.key === "resolution"
                ? `${prefs.resolution}p`
                : Number(prefs[f.key]).toFixed(f.step < 1 ? 2 : 0)}
            </strong>
          </span>
          <input
            type="range"
            min={f.min}
            max={f.max}
            step={f.step}
            value={Number(prefs[f.key])}
            onChange={(e) => onChange({ ...prefs, [f.key]: Number(e.target.value) })}
          />
        </label>
      ))}
      <label className="check">
        <input
          type="checkbox"
          checked={prefs.steer_move}
          onChange={(e) => onChange({ ...prefs, steer_move: e.target.checked })}
        />
        Nav ball also walks (WASD from stick direction)
      </label>
    </section>
  );
}
