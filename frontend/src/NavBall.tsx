import { useCallback, useRef } from "react";

type Props = {
  disabled?: boolean;
  onVector: (x: number, y: number) => void;
  onArrow: (dir: "left" | "right" | "up" | "down", down: boolean) => void;
  onReset: () => void;
};

function localXY(el: HTMLElement, clientX: number, clientY: number) {
  const r = el.getBoundingClientRect();
  const x = (clientX - r.left) / r.width * 2 - 1;
  const y = (clientY - r.top) / r.height * 2 - 1;
  const m = Math.hypot(x, y);
  if (m < 0.08) return { x: 0, y: 0, px: 0, py: 0 };
  const s = m > 1 ? 1 / m : 1;
  return { x: x * s, y: y * s, px: x * s, py: y * s };
}

export default function NavBall({ disabled, onVector, onArrow, onReset }: Props) {
  const ballRef = useRef<HTMLDivElement>(null);
  const puckRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const setPuck = (x: number, y: number) => {
    const puck = puckRef.current;
    if (!puck) return;
    puck.style.transform = `translate(${x * 24}px, ${y * 24}px)`;
  };

  const apply = useCallback(
    (clientX: number, clientY: number) => {
      const el = ballRef.current;
      if (!el || disabled) return;
      const v = localXY(el, clientX, clientY);
      setPuck(v.px, v.py);
      onVector(v.x, v.y);
    },
    [disabled, onVector],
  );

  const release = useCallback(() => {
    dragging.current = false;
    setPuck(0, 0);
    onVector(0, 0);
  }, [onVector]);

  return (
    <div className={`navwrap ${disabled ? "off" : ""}`}>
      <button
        type="button"
        className="arrow n"
        disabled={disabled}
        onPointerDown={(e) => {
          e.preventDefault();
          onArrow("up", true);
        }}
        onPointerUp={() => onArrow("up", false)}
        onPointerLeave={() => onArrow("up", false)}
      >
        Up
      </button>
      <button
        type="button"
        className="arrow w"
        disabled={disabled}
        onPointerDown={(e) => {
          e.preventDefault();
          onArrow("left", true);
        }}
        onPointerUp={() => onArrow("left", false)}
        onPointerLeave={() => onArrow("left", false)}
      >
        Left
      </button>

      <div
        ref={ballRef}
        className="navball"
        onPointerDown={(e) => {
          if (disabled) return;
          dragging.current = true;
          (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
          apply(e.clientX, e.clientY);
        }}
        onPointerMove={(e) => {
          if (dragging.current) apply(e.clientX, e.clientY);
        }}
        onPointerUp={release}
        onPointerCancel={release}
      >
        <div className="ring" />
        <div className="cross-h" />
        <div className="cross-v" />
        <div ref={puckRef} className="puck" />
      </div>

      <button
        type="button"
        className="arrow e"
        disabled={disabled}
        onPointerDown={(e) => {
          e.preventDefault();
          onArrow("right", true);
        }}
        onPointerUp={() => onArrow("right", false)}
        onPointerLeave={() => onArrow("right", false)}
      >
        Right
      </button>
      <button
        type="button"
        className="arrow s"
        disabled={disabled}
        onPointerDown={(e) => {
          e.preventDefault();
          onArrow("down", true);
        }}
        onPointerUp={() => onArrow("down", false)}
        onPointerLeave={() => onArrow("down", false)}
      >
        Down
      </button>

      <button type="button" className="ghost reset-view" disabled={disabled} onClick={onReset}>
        Reset view
      </button>
    </div>
  );
}
