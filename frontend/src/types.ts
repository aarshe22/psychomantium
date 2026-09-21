export type Intention = {
  id: number;
  text: string;
  kind: string;
  status: string;
  engine_action: string;
  note: string;
  active: boolean;
};

export type Stats = {
  type?: string;
  kind?: string;
  generation_fps?: number;
  last_gen_ms?: number;
  batches?: number;
  frames?: number;
  vram?: { allocated_mb?: number; reserved_mb?: number; max_allocated_mb?: number };
  gpu_util_pct?: number | null;
  fps_lock?: boolean;
  inpaint?: boolean;
  inpaint_status?: string;
  inpaint_progress?: number;
  max_fps?: number | null;
  session_active?: boolean;
  intentions?: Intention[];
  prompt_conditioning?: boolean;
  session_world_prompt?: string;
  prompt_apply?: string;
  error?: string | null;
  model?: string;
  native_size?: { width?: number; height?: number };
  stream_size?: { width?: number; height?: number };
  scene?: {
    lock?: number | null;
    openness?: number | null;
    locked?: boolean;
    open?: boolean;
    event?: string;
    open_memories?: number;
    walk_held?: boolean;
  };
  models?: { id: string; label: string; width?: number; height?: number; selected?: boolean; prompt_conditioning?: boolean; temporal?: number }[];
  ready?: boolean;
  loading?: boolean;
  clients?: number;
  bootstrap_phase?: string;
  bootstrap?: string;
  bootstrap_detail?: string;
  bootstrap_elapsed_s?: number;
};

export const KEY = { W: 87, A: 65, S: 83, D: 68, SPACE: 32 };

export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}
