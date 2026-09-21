import { FormEvent, MouseEvent, useCallback, useEffect, useRef, useState } from "react";
import Accordion from "./Accordion";
import Knobs, { Prefs } from "./Knobs";
import NavBall from "./NavBall";
import { KEY, Stats, Intention, wsUrl } from "./types";

const MAGIC = 0x50535943;

const DEFAULT_PREFS: Prefs = {
  resolution: 360,
  temperature: 1.0,
  look_sensitivity: 1.0,
  jpeg_quality: 78,
  wander: 0.0,
  motion_smoothing: 0.15,
  dream_sharpness: 0.45,
  steer_move: true,
  initial_note: "An explorable dream",
  world_prompt: "There is a standard road grid, and buildings.",
  model_id: "Overworld/Waypoint-1.5-1B-360P",
};

function parseJson(text: string, fallback: string) {
  try {
    return JSON.parse(text) as { error?: string; ok?: boolean };
  } catch {
    return { error: text.slice(0, 240) || fallback };
  }
}

function parseBinary(buf: ArrayBuffer, onFrame: (blob: Blob, seq: number) => void) {
  const view = new DataView(buf);
  let offset = 0;
  while (offset + 28 <= buf.byteLength) {
    const magic = view.getUint32(offset, true);
    if (magic !== MAGIC) break;
    const seq = view.getUint32(offset + 4, true);
    const jpegLen = view.getUint32(offset + 8, true);
    offset += 28;
    if (offset + jpegLen > buf.byteLength) break;
    const slice = buf.slice(offset, offset + jpegLen);
    onFrame(new Blob([slice], { type: "image/jpeg" }), seq);
    offset += jpegLen;
  }
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PREFS.initial_note);
  const [intentionText, setIntentionText] = useState("");
  const [intentions, setIntentions] = useState<Intention[]>([]);
  const [dreaming, setDreaming] = useState(false);
  const [conn, setConn] = useState<"idle" | "connecting" | "live" | "closed" | "error">("idle");
  const [stats, setStats] = useState<Stats>({});
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [loadingModel, setLoadingModel] = useState(true);
  const [deliveredFps, setDeliveredFps] = useState(0);
  const [prefs, setPrefs] = useState<Prefs>(DEFAULT_PREFS);
  const [saved, setSaved] = useState(true);
  const [railCollapsed, setRailCollapsed] = useState(false);
  const [railPinned, setRailPinned] = useState(true);
  const [openAcc, setOpenAcc] = useState<Record<string, boolean>>({
    session: true,
    intention: true,
    navigate: true,
    knobs: false,
    diag: false,
  });

  const imgRef = useRef<HTMLImageElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const keysRef = useRef<Set<number>>(new Set());
  const mouseRef = useRef<[number, number]>([0, 0]);
  const analogRef = useRef<[number, number]>([0, 0]);
  const arrowsRef = useRef<Set<string>>(new Set());
  const typingRef = useRef(false);
  const lastUrl = useRef<string | null>(null);
  const frameTimes = useRef<number[]>([]);
  const prefsTimer = useRef<number | null>(null);
  const resetViewRef = useRef<() => void>(() => undefined);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("psychomantium.rail.v2");
      if (!raw) return;
      const v = JSON.parse(raw) as { collapsed?: boolean; pinned?: boolean; open?: Record<string, boolean> };
      if (typeof v.collapsed === "boolean") setRailCollapsed(v.collapsed);
      if (typeof v.pinned === "boolean") setRailPinned(v.pinned);
      if (v.open) setOpenAcc((s) => ({ ...s, ...v.open }));
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(
      "psychomantium.rail.v2",
      JSON.stringify({ collapsed: railCollapsed, pinned: railPinned, open: openAcc }),
    );
  }, [railCollapsed, railPinned, openAcc]);

  useEffect(() => {
    fetch("/api/preferences")
      .then((r) => r.json())
      .then((body) => {
        if (body.prefs) {
          setPrefs({ ...DEFAULT_PREFS, ...body.prefs });
          if (body.prefs.initial_note) setPrompt(body.prefs.initial_note);
          setSaved(true);
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    let stop = false;
    const poll = async () => {
      try {
        const r = await fetch("/ready");
        const body = await r.json();
        if (stop) return;
        setReady(!!body.ready);
        setLoadingModel(!!body.loading);
        setStats((s) => ({ ...s, ...body }));
      } catch {
        if (!stop) setReady(false);
      }
    };
    poll();
    const id = setInterval(poll, 1000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, []);

  const sendControls = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(
      JSON.stringify({
        type: "controls",
        buttons: [...keysRef.current],
        mouse: mouseRef.current,
        analog: analogRef.current,
        arrows: [...arrowsRef.current],
      }),
    );
    mouseRef.current = [0, 0];
  }, []);

  const clearKeys = useCallback(() => {
    keysRef.current.clear();
    mouseRef.current = [0, 0];
    analogRef.current = [0, 0];
    arrowsRef.current.clear();
    sendControls();
  }, [sendControls]);

  useEffect(() => {
    if (!dreaming) return;
    const id = setInterval(sendControls, 50);
    return () => clearInterval(id);
  }, [dreaming, sendControls]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (typingRef.current) return;
      if (e.code === "KeyW") {
        e.preventDefault();
        keysRef.current.add(KEY.W);
      }
      if (e.code === "KeyZ") {
        e.preventDefault();
        keysRef.current.add(KEY.S);
      }
      if (e.code === "Space") {
        e.preventDefault();
        keysRef.current.add(KEY.SPACE);
      }
      if (e.code === "ArrowLeft") {
        e.preventDefault();
        arrowsRef.current.add("left");
      }
      if (e.code === "ArrowRight") {
        e.preventDefault();
        arrowsRef.current.add("right");
      }
      if (e.code === "ArrowUp") {
        e.preventDefault();
        arrowsRef.current.add("up");
      }
      if (e.code === "ArrowDown") {
        e.preventDefault();
        arrowsRef.current.add("down");
      }
      if (e.code === "KeyR") {
        e.preventDefault();
        if (!e.repeat) resetViewRef.current();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "KeyW") keysRef.current.delete(KEY.W);
      if (e.code === "KeyZ") keysRef.current.delete(KEY.S);
      if (e.code === "Space") keysRef.current.delete(KEY.SPACE);
      if (e.code === "ArrowLeft") arrowsRef.current.delete("left");
      if (e.code === "ArrowRight") arrowsRef.current.delete("right");
      if (e.code === "ArrowUp") arrowsRef.current.delete("up");
      if (e.code === "ArrowDown") arrowsRef.current.delete("down");
      if (e.code === "Escape") {
        document.exitPointerLock();
        clearKeys();
      }
    };
    const onBlur = () => clearKeys();
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [clearKeys]);

  const connectWs = useCallback(() => {
    setConn("connecting");
    const ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    ws.onopen = () => setConn("live");
    ws.onerror = () => setConn("error");
    ws.onclose = () => {
      setConn("closed");
      wsRef.current = null;
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data) as Stats & { type?: string; seq?: number };
        if (msg.type === "ack") return;
        if (msg.intentions) setIntentions(msg.intentions);
        setStats((s) => ({ ...s, ...msg }));
        if (msg.error) setError(msg.error);
        return;
      }
      parseBinary(ev.data as ArrayBuffer, (blob) => {
        const url = URL.createObjectURL(blob);
        if (imgRef.current) imgRef.current.src = url;
        if (lastUrl.current) URL.revokeObjectURL(lastUrl.current);
        lastUrl.current = url;
        const now = performance.now();
        frameTimes.current.push(now);
        frameTimes.current = frameTimes.current.filter((t) => now - t < 1000);
        setDeliveredFps(frameTimes.current.length);
      });
    };
  }, []);

  async function enterDream(e: FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a starting photograph.");
      return;
    }
    setError(null);
    await fetch("/api/preferences", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persist: false, prefs: { ...prefs, initial_note: prompt } }),
    });
    const body = new FormData();
    body.append("image", file);
    body.append("prompt", prompt);
    const res = await fetch("/api/session/start", { method: "POST", body });
    const data = parseJson(await res.text(), res.statusText);
    if (!res.ok) {
      setError(data.error || `Failed to start (${res.status})`);
      return;
    }
    setDreaming(true);
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) connectWs();
  }

  async function stopDream() {
    setDreaming(false);
    clearKeys();
    document.exitPointerLock();
    await fetch("/api/session/stop", { method: "POST" });
  }

  async function changeModel(nextId: string) {
    const current = stats.model || prefs.model_id;
    if (!nextId || nextId === current) return;
    setError(null);
    if (dreaming) await stopDream();
    setLoadingModel(true);
    setReady(false);
    const res = await fetch("/api/model", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: nextId }),
    });
    const data = parseJson(await res.text(), res.statusText) as {
      error?: string;
      prefs?: Prefs;
      model?: string;
    };
    if (!res.ok) {
      setError(data.error || `Could not switch model (${res.status})`);
      setLoadingModel(false);
      return;
    }
    if (data.prefs) setPrefs({ ...DEFAULT_PREFS, ...data.prefs });
    setStats((s) => ({ ...s, model: data.model || nextId }));
    setReady(true);
    setLoadingModel(false);
  }

  async function submitIntention(e: FormEvent) {
    e.preventDefault();
    const text = intentionText.trim();
    if (!text) return;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "intention", text }));
    } else {
      const res = await fetch("/api/intention", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const intent = await res.json();
      setIntentions((xs) => [...xs.slice(-11), intent]);
    }
    setIntentionText("");
  }

  async function saveSnapshot() {
    const res = await fetch("/api/session/snapshot", { method: "POST" });
    const data = await res.json();
    if (!res.ok) setError(data.error || "snapshot failed");
  }

  function onPick(f: File | null) {
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  function onViewportClick(ev: MouseEvent<HTMLDivElement>) {
    const el = ev.currentTarget;
    if (!dreaming) return;
    if (!document.pointerLockElement) {
      el.requestPointerLock();
    }
  }

  function livePrefs(next: Prefs) {
    setPrefs(next);
    setSaved(false);
    if (prefsTimer.current) window.clearTimeout(prefsTimer.current);
    prefsTimer.current = window.setTimeout(() => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "prefs", prefs: next }));
      } else {
        fetch("/api/preferences", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ persist: false, prefs: next }),
        });
      }
    }, 180);
  }

  async function savePrefs() {
    const body = { ...prefs, initial_note: prompt };
    const res = await fetch("/api/preferences", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persist: true, prefs: body }),
    });
    if (res.ok) {
      setPrefs(body);
      setSaved(true);
    } else {
      setError("Could not save preferences");
    }
  }

  function resetView() {
    analogRef.current = [0, 0];
    arrowsRef.current.clear();
    mouseRef.current = [0, 0];
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reset_orientation" }));
    } else {
      fetch("/api/session/reset-view", { method: "POST" });
    }
  }
  resetViewRef.current = resetView;

  function toggleAcc(id: string) {
    setOpenAcc((s) => ({ ...s, [id]: !s[id] }));
  }

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (document.pointerLockElement) {
        mouseRef.current[0] += e.movementX / 420;
        mouseRef.current[1] += e.movementY / 420;
      }
    };
    window.addEventListener("mousemove", onMove);
    return () => {
      window.removeEventListener("mousemove", onMove);
    };
  }, [dreaming]);

  return (
    <div className={`shell ${railCollapsed && !railPinned ? "rail-collapsed" : ""} ${railPinned ? "rail-pinned" : ""}`}>
      <header className="top">
        <div>
          <h1>Psychomantium</h1>
          <p className="lede">A locally generated lucid-dream sketch. Not a diagnosis. Not a map of you.</p>
        </div>
        <div className="top-meta">
          <label className="pill model-pick" title="World model checkpoint">
            <span className="metric-k">model</span>
            <select
              value={stats.model || prefs.model_id}
              disabled={loadingModel}
              onChange={(e) => changeModel(e.target.value)}
            >
              {(stats.models && stats.models.length > 0
                ? stats.models
                : [
                    { id: "Overworld/Waypoint-1.5-1B-360P", label: "Waypoint 1B · 360p" },
                    { id: "Overworld/Waypoint-1.5-1B", label: "Waypoint 1B · 720p" },
                  ]
              ).map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <div className="pill metric" title="GPU SM utilization from nvidia-smi">
            <span className="metric-k">gpu</span>
            <strong>
              {typeof stats.gpu_util_pct === "number" ? `${Math.round(stats.gpu_util_pct)}%` : "—"}
            </strong>
          </div>
          <div className="pill metric" title="Generation frames per second">
            <span className="metric-k">fps</span>
            <strong>
              {dreaming && typeof stats.generation_fps === "number" && stats.generation_fps > 0
                ? stats.generation_fps.toFixed(1)
                : dreaming && deliveredFps > 0
                  ? deliveredFps.toFixed(0)
                  : "—"}
            </strong>
          </div>
          <div className={`pill ${ready ? "ok" : loadingModel ? "wait" : "bad"}`}>
            {ready ? "model ready" : loadingModel ? "loading weights" : "server only"}
          </div>
        </div>
      </header>

      <div className="stage">
        <aside className="rail" aria-label="Controls">
          <div className="rail-tools">
            <button
              type="button"
              className="ghost rail-toggle"
              onClick={() => {
                if (railPinned) return;
                setRailCollapsed((v) => !v);
              }}
              disabled={railPinned}
              title={railPinned ? "Unpin to collapse" : railCollapsed ? "Show controls" : "Collapse left"}
            >
              {railCollapsed && !railPinned ? "Controls" : "Hide"}
            </button>
            <button
              type="button"
              className={`ghost pin ${railPinned ? "on" : ""}`}
              onClick={() => {
                setRailPinned((v) => !v);
                if (!railPinned) setRailCollapsed(false);
              }}
            >
              {railPinned ? "Pinned" : "Pin"}
            </button>
          </div>

          <div className="rail-body">
            <Accordion id="session" title="Session" open={!!openAcc.session} onToggle={toggleAcc}>
              <form onSubmit={enterDream}>
                <label className="file">
                  Starting photograph
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={(e) => onPick(e.target.files?.[0] ?? null)}
                  />
                </label>
                {preview && <img className="thumb" src={preview} alt="Seed preview" />}
                <label>
                  Standing world prompt
                  <textarea
                    rows={2}
                    value={prefs.world_prompt}
                    onChange={(e) => livePrefs({ ...prefs, world_prompt: e.target.value })}
                    onFocus={() => (typingRef.current = true)}
                    onBlur={() => (typingRef.current = false)}
                    placeholder="There is a standard road grid, and buildings."
                  />
                </label>
                <label>
                  Initial dream note
                  <textarea
                    rows={1}
                    value={prompt}
                    onChange={(e) => {
                      setPrompt(e.target.value);
                      setSaved(false);
                    }}
                    onFocus={() => (typingRef.current = true)}
                    onBlur={() => (typingRef.current = false)}
                    placeholder="Optional session note. Combined with the standing prompt."
                  />
                </label>
                <div className="row">
                  <button type="submit" disabled={!ready || !file}>
                    Enter dream
                  </button>
                  <button type="button" className="ghost" onClick={stopDream} disabled={!dreaming}>
                    Stop
                  </button>
                </div>
              </form>
            </Accordion>

            <Accordion id="intention" title="Intention" open={!!openAcc.intention} onToggle={toggleAcc}>
              <form className="intent" onSubmit={submitIntention}>
                <label>
                  Intention
                  <input
                    value={intentionText}
                    onChange={(e) => setIntentionText(e.target.value)}
                    onFocus={() => (typingRef.current = true)}
                    onBlur={() => {
                      typingRef.current = false;
                      clearKeys();
                    }}
                    placeholder="I can fly.  More buildings.  forget that"
                  />
                </label>
                <button type="submit" disabled={!dreaming}>
                  Speak
                </button>
              </form>
              <ul className="intents">
                {intentions.length === 0 && <li className="muted">No intentions yet.</li>}
                {intentions
                  .slice()
                  .reverse()
                  .map((it) => (
                    <li key={it.id}>
                      <span className={`st ${it.status}`}>{it.status}</span>
                      {it.kind === "world" && <span className="st world">world</span>}
                      <strong>{it.text}</strong>
                      <em>{it.engine_action}</em>
                      {it.note && <small>{it.note}</small>}
                    </li>
                  ))}
              </ul>
              <button type="button" className="ghost" onClick={saveSnapshot} disabled={!dreaming}>
                Save snapshot
              </button>
            </Accordion>

            <Accordion id="navigate" title="Navigate" open={!!openAcc.navigate} onToggle={toggleAcc}>
              <NavBall
                disabled={!dreaming}
                onVector={(x, y) => {
                  analogRef.current = [x, y];
                }}
                onArrow={(dir, down) => {
                  if (down) arrowsRef.current.add(dir);
                  else arrowsRef.current.delete(dir);
                }}
                onReset={resetView}
              />
            </Accordion>

            <Accordion id="knobs" title="Experience knobs" open={!!openAcc.knobs} onToggle={toggleAcc}>
              <Knobs prefs={prefs} saved={saved} onChange={livePrefs} onSave={savePrefs} />
            </Accordion>

            <Accordion id="diag" title="Diagnostics" open={!!openAcc.diag} onToggle={toggleAcc}>
              <dl className="diag-grid">
                <div>
                  <dt>connection</dt>
                  <dd>{conn}</dd>
                </div>
                <div>
                  <dt>generation fps</dt>
                  <dd>{stats.generation_fps?.toFixed(1) ?? "—"}</dd>
                </div>
                <div>
                  <dt>delivered fps</dt>
                  <dd>{deliveredFps.toFixed(0)}</dd>
                </div>
                <div>
                  <dt>batch ms</dt>
                  <dd>{stats.last_gen_ms?.toFixed(0) ?? "—"}</dd>
                </div>
                <div>
                  <dt>VRAM</dt>
                  <dd>{stats.vram?.allocated_mb?.toFixed(0) ?? "—"} MiB</dd>
                </div>
                <div>
                  <dt>stream</dt>
                  <dd>
                    {prefs.resolution}p · t={prefs.temperature.toFixed(2)}
                  </dd>
                </div>
                <div>
                  <dt>model</dt>
                  <dd>{stats.model ?? "—"}</dd>
                </div>
                <div>
                  <dt>world prompt</dt>
                  <dd>{stats.composed_prompt || prefs.world_prompt}</dd>
                </div>
                <div>
                  <dt>prompt apply</dt>
                  <dd>{stats.prompt_apply ?? (stats.prompt_conditioning ? "—" : "not wired on 1B")}</dd>
                </div>
                <div>
                  <dt>error</dt>
                  <dd className="err">{error || stats.error || "none"}</dd>
                </div>
              </dl>
            </Accordion>
          </div>
        </aside>

        <main className="stage-main">
          <div
            className="viewport"
            onClick={onViewportClick}
            onContextMenu={(e) => e.preventDefault()}
          >
            <img ref={imgRef} alt="Dream viewport" />
            {!dreaming && <div className="veil">Upload a photograph, then enter.</div>}
          </div>
          <p className="help">
            W walk forward · Z walk back · ← → turn · ↑ ↓ look · R reset to horizon · Space jump · click dream to
            mouse-look · Esc releases lock
          </p>
        </main>
      </div>
    </div>
  );
}
