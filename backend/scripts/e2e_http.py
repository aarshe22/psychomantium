#!/usr/bin/env python3
"""HTTP/WS checks against a running backend. Does not load a second GPU copy."""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8791"


def get(path: str, timeout: float = 10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as res:
        return res.status, json.loads(res.read().decode())


def post_json(path: str, payload: dict):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return res.status, json.loads(res.read().decode())


def png_bytes() -> bytes:
    arr = np.zeros((360, 640, 3), dtype=np.uint8)
    arr[:, :, 0] = 40
    arr[:, :, 1] = 80
    arr[:, :, 2] = 120
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    st, health = get("/health")
    print("health", st, health)
    assert health.get("status") == "ok"

    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            st, ready = get("/ready")
            print("ready", st, {k: ready.get(k) for k in ("ready", "loading", "error", "model")})
            if ready.get("ready"):
                break
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read().decode())
            print("ready", exc.code, {k: body.get(k) for k in ("ready", "loading", "error")})
            if not body.get("loading") and body.get("error"):
                raise SystemExit(f"model failed: {body['error']}")
        time.sleep(5)
    else:
        raise SystemExit("timeout waiting for /ready")

    png = png_bytes()
    boundary = "----psych"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        "test dream\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="image"; filename="seed.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + png + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + "/api/session/start",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as res:
        started = json.loads(res.read().decode())
    print("start", started)

    frames_deadline = time.time() + 600
    status = {}
    while time.time() < frames_deadline:
        st, status = get("/api/status")
        print(
            "status frames",
            status.get("frames"),
            "batches",
            status.get("batches"),
            "fps",
            status.get("generation_fps"),
            "err",
            status.get("error"),
        )
        if (status.get("batches") or 0) >= 1:
            break
        time.sleep(5)
    else:
        raise SystemExit("frames did not increase (torch.compile still running or gen failed)")

    _, intent = post_json("/api/intention", {"text": "It is nighttime now."})
    print("intention", intent)

    night_deadline = time.time() + 120
    while time.time() < night_deadline:
        st, status = get("/api/status")
        intents = status.get("intentions") or []
        print("after night", intents[-3:] if intents else None)
        last = intents[-1] if intents else {}
        if last.get("status") in {"submitted", "visually_verified", "failed"}:
            break
        time.sleep(3)

    _, stopped = post_json("/api/session/stop", {})
    print("stop", stopped)
    time.sleep(1)
    st, status = get("/api/status")
    print("after stop active", status.get("session_active"))

    req = urllib.request.Request(
        BASE + "/api/session/start",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as res:
        started2 = json.loads(res.read().decode())
    print("restart", started2)
    time.sleep(2)
    post_json("/api/session/stop", {})
    print("ok")


if __name__ == "__main__":
    main()
