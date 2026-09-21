export type PacedFrame = { blob: Blob; seq: number };

/** Spread a gen_frame JPEG batch across the predicted inter-batch interval (Biome-style). */
export class FramePacer {
  private ema = 133;
  private lastBatch = 0;
  private timers: number[] = [];
  private show: (frame: PacedFrame) => void;

  constructor(show: (frame: PacedFrame) => void) {
    this.show = show;
  }

  ingest(frames: PacedFrame[]) {
    for (const id of this.timers) window.clearTimeout(id);
    this.timers = [];
    if (!frames.length) return;
    const now = performance.now();
    if (this.lastBatch > 0) {
      const dt = now - this.lastBatch;
      if (dt > 40 && dt < 2500) this.ema = this.ema * 0.75 + dt * 0.25;
    }
    this.lastBatch = now;
    const slot = this.ema / Math.max(frames.length, 1);
    this.show(frames[0]);
    for (let i = 1; i < frames.length; i++) {
      const frame = frames[i];
      const id = window.setTimeout(() => this.show(frame), slot * i);
      this.timers.push(id);
    }
  }

  reset() {
    for (const id of this.timers) window.clearTimeout(id);
    this.timers = [];
    this.lastBatch = 0;
  }
}

export function splitPackedFrames(buf: ArrayBuffer): PacedFrame[] {
  const MAGIC = 0x50535943;
  const view = new DataView(buf);
  const out: PacedFrame[] = [];
  let offset = 0;
  while (offset + 28 <= buf.byteLength) {
    const magic = view.getUint32(offset, true);
    if (magic !== MAGIC) break;
    const seq = view.getUint32(offset + 4, true);
    const jpegLen = view.getUint32(offset + 8, true);
    offset += 28;
    if (offset + jpegLen > buf.byteLength) break;
    const slice = buf.slice(offset, offset + jpegLen);
    out.push({ blob: new Blob([slice], { type: "image/jpeg" }), seq });
    offset += jpegLen;
  }
  return out;
}
