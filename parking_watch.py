#!/usr/bin/env python3
"""
Change-detection parking notifier — resilient, stabilized, with Telegram commands.

- Threaded reader keeps only the LATEST frame -> never freezes / backs up.
- Auto-stabilized: every frame aligned to data/reference.jpg (ORB) before boxes
  are read, so camera drift/bumps don't cause false alerts.
- Alerts (Telegram photo) when a spot changes AND goes flatter (car -> asphalt).
- Telegram commands (text the bot):
      live    -> replies with the current annotated view (red=occupied, green=free)
      status  -> replies with a one-line summary
"""
import cv2, json, time, os, datetime, threading
import numpy as np
import requests
import parking_notifier as cfg

# ----------------------------- CONFIG ---------------------------------
SPOTS_FILE      = "spots.json"
REFERENCE_IMG   = "data/reference.jpg"
CHECK_EVERY_SEC = 3
EMPTY_HOLD_SEC  = 7
CROP_SIZE       = 64
CHANGE_THRESH   = 0.12
EDGE_DROP       = 0.70
STABILIZE       = True
MIN_INLIERS     = 25
TIMELAPSE_SEC   = 120
LOG_CSV         = "data/metrics_log.csv"
EVENT_DIR       = "data/events"
TL_DIR          = "data/timelapse"
NOTIFY_TAKEN    = True
DISCONNECT_SEC  = 30     # alert if no NEW (changed) frame for this long
FREEZE_RECONNECT_SEC = 12  # if the frame is frozen this long, force-reopen the stream
# ----------------------------------------------------------------------

os.makedirs(EVENT_DIR, exist_ok=True)
os.makedirs(TL_DIR, exist_ok=True)

_orb = cv2.ORB_create(2000)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
_ref_kp = _ref_des = _ref_size = None


def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# --------------------------- Telegram ---------------------------------
def send_photo(frame, caption):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        cfg.send_telegram(caption); return
    url = f"https://api.telegram.org/bot{cfg.BOT_TOKEN}/sendPhoto"
    try:
        requests.post(url, data={"chat_id": cfg.CHAT_ID, "caption": caption},
                      files={"photo": ("spot.jpg", buf.tobytes(), "image/jpeg")}, timeout=15)
    except Exception as e:
        log(f"sendPhoto failed: {e}"); cfg.send_telegram(caption)


class TG:
    """Minimal long-poll listener for text commands from the authorized chat."""
    def __init__(self):
        self.base = f"https://api.telegram.org/bot{cfg.BOT_TOKEN}"
        self.chat = str(cfg.CHAT_ID)
        self.offset = None
        try:  # skip any backlog so old messages don't fire on startup
            r = requests.get(f"{self.base}/getUpdates", params={"timeout": 0}, timeout=10).json()
            if r.get("ok") and r["result"]:
                self.offset = r["result"][-1]["update_id"] + 1
        except Exception:
            pass

    def poll(self):
        cmds = []
        try:
            p = {"timeout": 0}
            if self.offset is not None:
                p["offset"] = self.offset
            r = requests.get(f"{self.base}/getUpdates", params=p, timeout=10).json()
            if r.get("ok"):
                for u in r["result"]:
                    self.offset = u["update_id"] + 1
                    m = u.get("message") or {}
                    if str(m.get("chat", {}).get("id")) == self.chat:
                        t = (m.get("text") or "").strip().lower().lstrip("/")
                        if t:
                            cmds.append(t)
        except Exception:
            pass
        return cmds


# ------------------------- stabilization ------------------------------
def load_reference():
    global _ref_kp, _ref_des, _ref_size
    ref = cv2.imread(REFERENCE_IMG)
    if ref is None:
        raise SystemExit(f"Missing {REFERENCE_IMG}")
    _ref_kp, _ref_des = _orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    _ref_size = (ref.shape[1], ref.shape[0])
    log(f"reference loaded, {len(_ref_kp)} landmarks")


def stabilize(frame):
    if not STABILIZE:
        return frame
    if (frame.shape[1], frame.shape[0]) != _ref_size:   # camera resolution changed
        frame = cv2.resize(frame, _ref_size)
    kp, des = _orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    if des is None or len(kp) < 2 * MIN_INLIERS:
        return None
    matches = _bf.match(_ref_des, des)
    if len(matches) < 2 * MIN_INLIERS:
        return None
    src = np.float32([kp[m.trainIdx].pt for m in matches])
    dst = np.float32([_ref_kp[m.queryIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5)
    if M is None or inl is None or int(inl.sum()) < MIN_INLIERS:
        return None
    if not (0.85 < (M[0, 0] ** 2 + M[0, 1] ** 2) ** 0.5 < 1.15):
        return None
    return cv2.warpAffine(frame, M, _ref_size)


def prep(crop):
    g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (CROP_SIZE, CROP_SIZE))
    edge = float(cv2.Canny(g, 50, 150).mean()) / 255.0
    norm = g.astype(np.float32); norm -= norm.mean()
    return norm, edge


def diff(a, b):
    return float(np.abs(a - b).mean()) / 255.0


# --------------------- threaded latest-frame reader -------------------
class LatestFrame:
    def __init__(self, url):
        self.url = url; self.frame = None; self.running = True
        self.last_ok = time.time()
        self.last_change = time.time()   # last time frame CONTENT actually changed
        self.cap = self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        prev = None
        while self.running:
            ok, f = self.cap.read()
            if not ok or f is None:
                time.sleep(0.3); self.cap.release(); self.cap = self._open(); prev = None; continue
            self.frame = f; self.last_ok = time.time()
            # frozen-stream detection: a live camera never sends byte-identical frames
            if prev is None or not np.array_equal(f, prev):
                self.last_change = time.time()
            prev = f
            if time.time() - self.last_change > FREEZE_RECONNECT_SEC:
                # stream frozen -> force reopen (keep prev so a still-frozen stream stays detected)
                self.cap.release(); self.cap = self._open(); time.sleep(0.5)

    def read(self):
        return None if self.frame is None else self.frame.copy()


def render_live(frame, spots, S):
    """Annotate current view with per-spot state for the 'live' command."""
    img = frame.copy()
    free = 0
    for i, (x1, y1, x2, y2) in enumerate(spots):
        empty = S[i]["state"] == "empty"
        free += empty
        color = (0, 255, 0) if empty else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{i}:{'FREE' if empty else 'occ'}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img, free


def main():
    spots = json.load(open(SPOTS_FILE))
    n = len(spots)
    load_reference()
    cam = LatestFrame(cfg.RTSP_URL)
    tg = TG()
    log(f"Watching {n} spots (stabilized + Telegram commands).")

    # first stabilizable frame -> references
    frame = None
    t0 = time.time()
    while time.time() - t0 < 60:
        f = cam.read()
        if f is not None:
            a = stabilize(f)
            if a is not None:
                frame = a; break
        time.sleep(0.2)
    if frame is None:
        raise SystemExit("No stabilized frame in 60s.")

    S = []
    for (x1, y1, x2, y2) in spots:
        norm, edge = prep(frame[y1:y2, x1:x2])
        S.append({"state": "occupied", "occ_ref": norm, "occ_edge": edge,
                  "empty_ref": None, "empty_edge": None, "empty_since": None, "occ_since": None})

    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w") as f:
            f.write("time,spot,state,change,edge,ref_edge\n")

    last_aligned = frame
    cfg.send_telegram(f"Parking watch started ✅ {n} spots. Text 'live' for a snapshot.")
    log("references captured, monitoring")

    last_tl = last_beat = 0.0
    disconnected = False
    while True:
        try:
            # ---- Telegram commands (responsive) ----
            for cmd in tg.poll():
                if cmd in ("live", "view", "pic"):
                    img, free = render_live(last_aligned, spots, S)
                    send_photo(img, f"📷 Live — {free}/{n} free")
                    log(f"'live' command -> sent ({free} free)")
                elif cmd == "status":
                    states = "".join("F" if s["state"] == "empty" else "o" for s in S)
                    cfg.send_telegram(f"Status: {states}  ({sum(s['state']=='empty' for s in S)}/{n} free)")

            # ---- camera health (catches BOTH dead and frozen streams) ----
            if time.time() - cam.last_change > DISCONNECT_SEC:
                if not disconnected:
                    disconnected = True
                    cfg.send_telegram("⚠️ Camera DISCONNECTED / frozen — no live video.")
                    log("camera disconnected/frozen")
                time.sleep(CHECK_EVERY_SEC); continue
            elif disconnected:
                disconnected = False
                cfg.send_telegram("✅ Camera reconnected.")
                log("camera reconnected")

            # ---- detection ----
            raw = cam.read()
            if raw is None:
                time.sleep(1); continue
            frame = stabilize(raw)
            if frame is None:
                time.sleep(CHECK_EVERY_SEC); continue
            last_aligned = frame

            now = time.time()
            stamp = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
            if TIMELAPSE_SEC and now - last_tl >= TIMELAPSE_SEC:
                cv2.imwrite(os.path.join(TL_DIR, f"{datetime.datetime.now():%Y%m%d_%H%M%S}.jpg"), raw)
                last_tl = now

            rows = []
            for i, (x1, y1, x2, y2) in enumerate(spots):
                s = S[i]
                norm, edge = prep(frame[y1:y2, x1:x2])
                if s["state"] == "occupied":
                    change = diff(norm, s["occ_ref"])
                    if change > CHANGE_THRESH and edge < s["occ_edge"] * EDGE_DROP:
                        if s["empty_since"] is None:
                            s["empty_since"] = now
                        elif now - s["empty_since"] >= EMPTY_HOLD_SEC:
                            f2 = frame.copy()
                            cv2.rectangle(f2, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(f2, f"FREE #{i}", (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.imwrite(os.path.join(EVENT_DIR, f"free_{i}_{int(now)}.jpg"), f2)
                            send_photo(f2, f"🅿️ Parking spot #{i} is now FREE!")
                            log(f"SPOT #{i} FREED")
                            s.update(state="empty", empty_ref=norm, empty_edge=edge, empty_since=None)
                    else:
                        s["empty_since"] = None
                        if change < CHANGE_THRESH * 0.5:
                            s["occ_ref"] = 0.9 * s["occ_ref"] + 0.1 * norm
                            s["occ_edge"] = 0.9 * s["occ_edge"] + 0.1 * edge
                else:
                    change = diff(norm, s["empty_ref"])
                    if change > CHANGE_THRESH and edge > s["empty_edge"] / EDGE_DROP:
                        if s["occ_since"] is None:
                            s["occ_since"] = now
                        elif now - s["occ_since"] >= EMPTY_HOLD_SEC:
                            s.update(state="occupied", occ_ref=norm, occ_edge=edge, occ_since=None)
                            if NOTIFY_TAKEN:
                                cfg.send_telegram(f"Spot #{i} is taken again.")
                            log(f"spot #{i} re-occupied")
                    else:
                        s["occ_since"] = None
                rows.append(f"{stamp},{i},{s['state']},{diff(norm, s['occ_ref']):.4f},{edge:.4f},{s['occ_edge']:.4f}")

            with open(LOG_CSV, "a") as f:
                f.write("\n".join(rows) + "\n")

            if now - last_beat >= 600:
                log(f"alive; states={[s['state'][0] for s in S]}"); last_beat = now

            time.sleep(CHECK_EVERY_SEC)

        except Exception as e:
            log(f"loop error: {e!r} — continuing")
            time.sleep(3)


if __name__ == "__main__":
    main()
