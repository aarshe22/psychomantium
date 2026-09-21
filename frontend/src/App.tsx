import { CSSProperties, FormEvent, MouseEvent, useCallback, useEffect, useRef, useState } from "react";
import Accordion from "./Accordion";
import { FramePacer, splitPackedFrames } from "./FramePacer";
import Knobs, { Prefs } from "./Knobs";
import NavBall from "./NavBall";
import { KEY, Stats, Intention, wsUrl } from "./types";

type GallerySeed = {
  id: string;
  label: string;
  caption: string;
  url: string;
  default?: boolean;
  source?: string;
  deletable?: boolean;
};

const DEFAULT_PREFS: Prefs = {
  resolution: 360,
  temperature: 0.4,
  look_sensitivity: 1.75,
  jpeg_quality: 78,
  wander: 0.0,
  motion_smoothing: 0.15,
  dream_sharpness: 0.45,
  steer_move: true,
  initial_note: "An explorable dream",
  world_prompt:
    "There is a standard road grid, and buildings. Purely exploratory first-person walk: empty unarmed hands. No weapons, firearms, hammers, swords, tools, or any held object.",
  model_id: "Overworld/Waypoint-1.5-1B-360P",
  fps_lock: false,
  inpaint: false,
};

function parseJson(text: string, fallback: string) {
  try {
    return JSON.parse(text) as { error?: string; ok?: boolean };
  } catch {
    return { error: text.slice(0, 240) || fallback };
  }
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [seedId, setSeedId] = useState<string>("");
  const [gallery, setGallery] = useState<GallerySeed[]>([]);
  const [painting, setPainting] = useState(false);
  const [seedLine, setSeedLine] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PREFS.initial_note);
  const [intentionText, setIntentionText] = useState("");
  const [intentions, setIntentions] = useState<Intention[]>([]);
  const [dreaming, setDreaming] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [conn, setConn] = useState<"idle" | "connecting" | "live" | "closed" | "error">("idle");
  const [stats, setStats] = useState<Stats>({});
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [loadingModel, setLoadingModel] = useState(true);
  const [deliveredFps, setDeliveredFps] = useState(0);
  const [hasFrame, setHasFrame] = useState(false);
  const [prefs, setPrefs] = useState<Prefs>(DEFAULT_PREFS);
  const [saved, setSaved] = useState(true);
  const [railCollapsed, setRailCollapsed] = useState(false);
  const [railPinned, setRailPinned] = useState(true);
  const [openAcc, setOpenAcc] = useState<Record<string, boolean>>({
    session: true,
    intention: true,
    navigate: true,
    knobs: true,
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
  const resetSeedRef = useRef<() => void>(() => undefined);
  const pacerRef = useRef<FramePacer | null>(null);
  const stoppingRef = useRef(false);

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
    fetch("/api/seeds")
      .then((r) => r.json())
      .then((body) => {
        const seeds = (body.seeds || []) as GallerySeed[];
        setGallery(seeds);
        setSeedId((current) => {
          if (current) return current;
          const def = seeds.find((s) => s.default) || seeds[0];
          if (def) setPreview(def.url);
          return def ? def.id : "";
        });
      })
      .catch(() => undefined);
  }, [ready]);

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
      if (e.code === "KeyU") {
        e.preventDefault();
        if (!e.repeat) resetSeedRef.current();
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
        if (typeof msg.session_active === "boolean") {
          if (!(stoppingRef.current && msg.session_active)) setDreaming(msg.session_active);
        }
        setStats((s) => ({ ...s, ...msg }));
        if (msg.error) setError(msg.error);
        return;
      }
      const packed = splitPackedFrames(ev.data as ArrayBuffer);
      let pacer = pacerRef.current;
      if (!pacer) {
        pacer = new FramePacer((frame) => {
          const url = URL.createObjectURL(frame.blob);
          if (imgRef.current) imgRef.current.src = url;
          setHasFrame(true);
          if (lastUrl.current) URL.revokeObjectURL(lastUrl.current);
          lastUrl.current = url;
          const now = performance.now();
          frameTimes.current.push(now);
          frameTimes.current = frameTimes.current.filter((t) => now - t < 1000);
          setDeliveredFps(frameTimes.current.length);
        });
        pacerRef.current = pacer;
      }
      pacer.ingest(packed);
    };
  }, []);

  async function enterDream(e: FormEvent) {
    e.preventDefault();
    if (dreaming || stoppingRef.current) return;
    if (!file && !seedId) {
      setError("Choose a starting photograph or a gallery seed.");
      return;
    }
    setError(null);
    pacerRef.current?.reset();
    setDreaming(true);
    await fetch("/api/preferences", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persist: false, prefs: { ...prefs, initial_note: prompt } }),
    });
    const body = new FormData();
    if (file) body.append("image", file);
    else body.append("seed_id", seedId);
    body.append("prompt", prompt);
    const res = await fetch("/api/session/start", { method: "POST", body });
    const data = parseJson(await res.text(), res.statusText);
    if (!res.ok) {
      setDreaming(false);
      setError(data.error || `Failed to start (${res.status})`);
      return;
    }
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) connectWs();
  }

  async function stopDream() {
    if (stoppingRef.current || !dreaming) return;
    stoppingRef.current = true;
    setStopping(true);
    setDreaming(false);
    clearKeys();
    pacerRef.current?.reset();
    document.exitPointerLock();
    try {
      await fetch("/api/session/stop", { method: "POST" });
    } finally {
      stoppingRef.current = false;
      setStopping(false);
      setDreaming(false);
      setIntentions([]);
    }
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
    if (preview && preview.startsWith("blob:")) URL.revokeObjectURL(preview);
    if (f) {
      setSeedId("");
      setPreview(URL.createObjectURL(f));
    }
  }

  function pickGallery(s: GallerySeed) {
    setFile(null);
    if (preview && preview.startsWith("blob:")) URL.revokeObjectURL(preview);
    setSeedId(s.id);
    setPreview(s.url);
  }

  async function createSeed(text: string) {
    const line = text.trim();
    if (!line) {
      setError("Enter a one-line prompt for the seed.");
      return;
    }
    setError(null);
    setPainting(true);
    try {
      const res = await fetch("/api/seeds/paint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: line }),
      });
      const data = parseJson(await res.text(), res.statusText) as {
        error?: string;
        seed?: GallerySeed;
      };
      if (!res.ok || !data.seed) {
        setError(data.error || `Paint failed (${res.status})`);
        return;
      }
      const seed = {
        ...data.seed,
        url: data.seed.url || `/api/seeds/${data.seed.id}.jpg`,
        deletable: true,
        source: data.seed.source || "klein",
      };
      setGallery((g) => [seed, ...g.filter((s) => s.id !== seed.id)]);
      setFile(null);
      setSeedId(seed.id);
      setPreview(`${seed.url}?t=${Date.now()}`);
    } finally {
      setPainting(false);
    }
  }

  async function paintFromPrompt() {
    await createSeed([prefs.world_prompt, prompt].filter(Boolean).join(". "));
  }

  async function deleteSeed(id: string, ev: MouseEvent<HTMLButtonElement>) {
    ev.preventDefault();
    ev.stopPropagation();
    setError(null);
    const res = await fetch(`/api/seeds/${encodeURIComponent(id)}`, { method: "DELETE" });
    const data = parseJson(await res.text(), res.statusText);
    if (!res.ok) {
      setError(data.error || `Could not delete (${res.status})`);
      return;
    }
    setGallery((g) => g.filter((s) => s.id !== id));
    if (seedId === id) {
      setSeedId("");
      if (preview && !preview.startsWith("blob:")) setPreview(null);
    }
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

  async function resetPrefs() {
    const next: Prefs = {
      ...prefs,
      resolution: DEFAULT_PREFS.resolution,
      temperature: DEFAULT_PREFS.temperature,
      look_sensitivity: DEFAULT_PREFS.look_sensitivity,
      jpeg_quality: DEFAULT_PREFS.jpeg_quality,
      wander: DEFAULT_PREFS.wander,
      motion_smoothing: DEFAULT_PREFS.motion_smoothing,
      dream_sharpness: DEFAULT_PREFS.dream_sharpness,
      steer_move: DEFAULT_PREFS.steer_move,
    };
    livePrefs(next);
    const res = await fetch("/api/preferences", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persist: true, prefs: { ...next, initial_note: prompt } }),
    });
    if (res.ok) setSaved(true);
    else setError("Could not reset preferences");
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

  function resetSeed() {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reset_seed" }));
    } else {
      fetch("/api/session/reset-seed", { method: "POST" });
    }
  }
  resetSeedRef.current = resetSeed;

  async function toggleFpsLock() {
    const next = { ...prefs, fps_lock: !prefs.fps_lock };
    setPrefs(next);
    setSaved(false);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "prefs", prefs: next }));
    }
    const res = await fetch("/api/preferences", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persist: true, prefs: { ...next, initial_note: prompt } }),
    });
    if (res.ok) setSaved(true);
  }

  async function toggleInpaint() {
    const next = { ...prefs, inpaint: !prefs.inpaint };
    setPrefs(next);
    setSaved(false);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "prefs", prefs: next }));
    }
    const res = await fetch("/api/preferences", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persist: true, prefs: { ...next, initial_note: prompt } }),
    });
    if (res.ok) setSaved(true);
  }

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
          <div
            className={`pill metric run ${dreaming ? "running" : "stopped"}`}
            title={dreaming ? "Dream session is running. Stop to end it." : "No dream session. Start Dreaming to begin."}
            aria-live="polite"
            role="status"
            aria-label={dreaming ? "session RUNNING" : "session NOT RUNNING"}
          >
            <span className="metric-k">session</span>
            <strong>{dreaming ? "RUNNING" : "NOT RUNNING"}</strong>
          </div>
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
          <button
            type="button"
            className={`pill metric toggle ${prefs.fps_lock ? "on" : ""}`}
            title={
              prefs.fps_lock
                ? "30 fps cap on. Click to stream uncapped."
                : "Uncapped. Click to lock 30 fps."
            }
            aria-pressed={prefs.fps_lock}
            onClick={toggleFpsLock}
          >
            <span className="metric-k">{prefs.fps_lock ? "fps 30" : "fps"}</span>
            <strong>
              {dreaming && typeof stats.generation_fps === "number" && stats.generation_fps > 0
                ? stats.generation_fps.toFixed(1)
                : dreaming && deliveredFps > 0
                  ? deliveredFps.toFixed(0)
                  : "—"}
            </strong>
          </button>
          <button
            type="button"
            className={`pill metric toggle ${prefs.inpaint ? "on" : ""} ${
              stats.inpaint_status === "running" ? "inpaint-busy" : ""
            } ${stats.inpaint_status === "loading" ? "inpaint-load" : ""}`}
            style={
              stats.inpaint_status === "running"
                ? ({ ["--inpaint-pct"]: `${Math.round(Math.max(0, Math.min(1, stats.inpaint_progress ?? 0)) * 100)}%` } as CSSProperties)
                : undefined
            }
            title={
              stats.inpaint_status === "running"
                ? `Inpainting ${Math.round(Math.max(0, Math.min(1, stats.inpaint_progress ?? 0)) * 100)}%`
                : stats.inpaint_status === "loading"
                  ? "Loading FLUX.2 Klein for inpaint"
                  : prefs.inpaint
                    ? "Auto-InPaint on. Stand still and FLUX.2 Klein refines this frame; walking continues from the detailed seed. Send Intention always inpaints the current view, even if this is off."
                    : "Auto-InPaint off. Idle standing still will not refine. Send Intention still inpaints the current view with your intention."
            }
            aria-pressed={prefs.inpaint}
            onClick={toggleInpaint}
          >
            <span className="metric-k">Auto-InPaint</span>
            <strong>
              {stats.inpaint_status === "loading"
                ? "load"
                : stats.inpaint_status === "running"
                  ? `${Math.round(Math.max(0, Math.min(1, stats.inpaint_progress ?? 0)) * 100)}%`
                  : stats.inpaint_status === "error"
                    ? "err"
                    : !prefs.inpaint
                      ? "off"
                      : stats.inpaint_status === "done"
                        ? "ok"
                        : stats.inpaint_status === "armed"
                          ? "on"
                          : "on"}
            </strong>
          </button>
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
                <p className="fine">
                  Waypoint continues from a <strong>photoreal first-person still</strong>. Upload a photograph,
                  pick a CC0 path or road still, or paint one with Klein from the standing prompt.
                </p>
                <label className="file">
                  Your photograph (best prior)
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
                    rows={3}
                    value={
                      dreaming
                        ? stats.session_world_prompt || prefs.world_prompt
                        : prefs.world_prompt
                    }
                    readOnly={dreaming}
                    onChange={(e) => livePrefs({ ...prefs, world_prompt: e.target.value })}
                    onFocus={() => (typingRef.current = true)}
                    onBlur={() => (typingRef.current = false)}
                    placeholder="Peaceful exploration. Empty hands. No weapons."
                  />
                </label>
                {dreaming && (
                  <p className="fine">
                    Session standing prompt. Each intention is appended here until Stop. Saved preferences stay the original prompt.
                  </p>
                )}
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
                <button
                  type="button"
                  className="ghost"
                  onClick={paintFromPrompt}
                  disabled={painting || dreaming}
                  title="FLUX.2 Klein paints a first-person JPEG from the standing prompt, then you enter from that seed."
                >
                  {painting ? "Creating seed…" : "Paint seed from standing prompt"}
                </button>
                <label>
                  One-line seed prompt
                  <input
                    value={seedLine}
                    onChange={(e) => setSeedLine(e.target.value)}
                    onFocus={() => (typingRef.current = true)}
                    onBlur={() => (typingRef.current = false)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        void createSeed(seedLine);
                      }
                    }}
                    placeholder="Sunlit empty alley, cobblestones, no people"
                    maxLength={500}
                    disabled={painting || dreaming}
                  />
                </label>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => void createSeed(seedLine)}
                  disabled={painting || dreaming || !seedLine.trim()}
                  title="Best local generator: FLUX.2 Klein 4B. JPEG is cached on the host under data/seeds/painted/."
                >
                  {painting ? "Creating seed…" : "Create seed"}
                </button>
                {gallery.length > 0 && (
                  <>
                    <p className="fine">Cached Klein seeds and CC0 stills</p>
                    <div className="seed-grid">
                    {gallery.map((s) => (
                      <div key={s.id} className="seed-wrap">
                        <button
                          type="button"
                          className={`seed-card ${seedId === s.id && !file ? "on" : ""}`}
                          onClick={() => pickGallery(s)}
                          title={s.caption}
                        >
                          <img src={s.url} alt={s.label} />
                          <span>{s.label}</span>
                        </button>
                        {(s.deletable || s.source === "klein") && (
                          <button
                            type="button"
                            className="seed-trash"
                            title="Delete this cached seed"
                            aria-label={`Delete ${s.label}`}
                            onClick={(e) => void deleteSeed(s.id, e)}
                          >
                            <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
                              <path
                                fill="currentColor"
                                d="M6 1h4l.5 1H14v1H2V2h3.5L6 1zm1 4v7H6V5h1zm3 0v7H9V5h1zM3.5 4h9l-.6 10.2A1 1 0 0 1 11 15H5a1 1 0 0 1-1-.8L3.5 4z"
                              />
                            </svg>
                          </button>
                        )}
                      </div>
                    ))}
                    </div>
                  </>
                )}
                <div className="row">
                  <button
                    type="submit"
                    className={dreaming || stopping ? "is-dreaming" : undefined}
                    disabled={stopping || (!dreaming && (!ready || (!file && !seedId) || painting))}
                    aria-pressed={dreaming}
                  >
                    {dreaming || stopping ? "Dreaming" : "Start Dreaming"}
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={stopDream}
                    disabled={stopping}
                    aria-busy={stopping}
                  >
                    {stopping ? "Stopping" : dreaming ? "Stop" : "Stopped"}
                  </button>
                </div>
                <button type="button" className="ghost" onClick={resetSeed} disabled={!dreaming} title="U">
                  Reset seed
                </button>
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
                    placeholder="A river on the left. More buildings ahead."
                  />
                </label>
                <button type="submit" disabled={!dreaming}>
                  Send Intention
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
              <Knobs prefs={prefs} saved={saved} onChange={livePrefs} onSave={savePrefs} onReset={resetPrefs} />
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
            className={`viewport ${hasFrame ? "" : "idle"}`.trim()}
            onClick={onViewportClick}
            onContextMenu={(e) => e.preventDefault()}
          >
            <img ref={imgRef} alt="Dream viewport" className={hasFrame ? undefined : "is-empty"} />
            {!hasFrame && (
              <div className="get-started">
                <div className="get-started-inner">
                  <p className="eyebrow">Get started</p>
                  <h2>Psychomantium</h2>
                  <p>
                    A locally generated lucid-dream sketch. You choose a first-person still; a world model on this
                    machine continues that view as you look and walk. It is not a diagnosis and not a map of you.
                    On the loaded Waypoint 1B checkpoint, typed text does not drive the DiT — movement and the
                    current pixels do. Klein inpaint is how an intention edits the view you are in.
                  </p>

                  <h3>Start a dream</h3>
                  <ol>
                    <li>
                      Wait until the top bar says <strong>model ready</strong>.
                    </li>
                    <li>
                      In <strong>Session</strong>, pick a CC0 still, upload a photograph, paint from the
                      standing prompt, or type a one-liner and <strong>Create seed</strong>.
                    </li>
                    <li>
                      Optional: edit the standing world prompt. Defaults are unarmed exploration.
                    </li>
                    <li>
                      Click <strong>Start Dreaming</strong>. Live frames replace this card. <strong>Stop</strong>{" "}
                      ends the session; while it winds down the button reads <strong>Stopping</strong>, then{" "}
                      <strong>Stopped</strong>.
                    </li>
                  </ol>

                  <h3>Top bar</h3>
                  <dl className="ctl">
                    <dt>session</dt>
                    <dd>RUNNING while generating. NOT RUNNING when idle.</dd>
                    <dt>model</dt>
                    <dd>Waypoint 1B at 360p or 720p. Switching reloads weights.</dd>
                    <dt>gpu</dt>
                    <dd>SM utilization from this machine.</dd>
                    <dt>fps</dt>
                    <dd>Click to lock ~30 fps, or leave uncapped.</dd>
                    <dt>Auto-InPaint</dt>
                    <dd>
                      When on, standing still lets FLUX.2 Klein refine the current frame. Walking continues from
                      those pixels. Fill is progress while Klein runs.
                    </dd>
                  </dl>

                  <h3>Left rail</h3>
                  <dl className="ctl">
                    <dt>Session</dt>
                    <dd>
                      Seed the dream. Pin keeps the rail open; Hide collapses it. Reset seed (U) returns to the
                      original still.
                    </dd>
                    <dt>Intention</dt>
                    <dd>
                      <strong>Send Intention</strong> Klein-inpaints the view where you are now, then appends that
                      line to the session standing prompt. Lines stack; they are discarded on Stop and are not
                      saved in preferences.
                    </dd>
                    <dt>Navigate</dt>
                    <dd>
                      W walk · Z back · arrows look/turn · Space jump · nav ball look (and walk if the knob is on)
                      · Reset view (R) to the horizon. Click the dream for mouse-look; Esc releases the pointer.
                    </dd>
                    <dt>Knobs</dt>
                    <dd>
                      Stream resolution, temperature, look sensitivity, JPEG quality, idle wander, motion
                      smoothing, dream sharpness. Save writes preferences; Reset to Defaults restores the knob
                      fields.
                    </dd>
                    <dt>Diagnostics</dt>
                    <dd>Generation fps, VRAM, composed prompt, and whether set_prompt is wired (it is not on 1B).</dd>
                  </dl>
                </div>
              </div>
            )}
            {hasFrame && !dreaming && <div className="veil">Stopped. Start Dreaming to continue from a seed.</div>}
          </div>
          <p className="help">
            W walk forward · Z walk back · ← → turn · ↑ ↓ look · R reset to horizon · U reset seed · Space jump ·
            click dream to mouse-look · Esc releases lock
          </p>
        </main>
      </div>
    </div>
  );
}
