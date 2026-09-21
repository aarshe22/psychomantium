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
  session_active?: boolean;
  intentions?: Intention[];
  prompt_conditioning?: boolean;
  composed_prompt?: string;
  prompt_apply?: string;
  error?: string | null;
  model?: string;
  ready?: boolean;
  loading?: boolean;
  clients?: number;
};

export const KEY = { W: 87, A: 65, S: 83, D: 68, SPACE: 32 };

export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}
