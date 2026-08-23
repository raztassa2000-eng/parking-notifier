#!/usr/bin/env python3
"""
Empty-parking-spot notifier for a Tapo camera.

Watches an RTSP stream, checks a set of parking-spot boxes you define,
and sends a Telegram push when a spot becomes (and stays) empty.

First-time setup:
  1) pip install ultralytics opencv-python requests
  2) Fill in the CONFIG section below.
  3) Run once with DRAW_SETUP = True to pick your parking-spot boxes on a
     still frame (click two opposite corners per spot, press 's' to save,
     'q' to quit). It writes the coordinates to spots.json.
  4) Set DRAW_SETUP = False and run again to start monitoring.
"""

import json
import time
import os
import cv2
import requests
try:
    from ultralytics import YOLO   # only needed for the (unused) YOLO approach
except ImportError:
    YOLO = None

# ----------------------------- CONFIG ---------------------------------
# Secrets live in config.py (git-ignored). Copy config_example.py -> config.py
# and fill in your own RTSP URL, Telegram bot token, and chat id.
from config import RTSP_URL, BOT_TOKEN, CHAT_ID

DRAW_SETUP = True        # True = pick spots on a still frame, then set to False
SPOTS_FILE = "spots.json"

# Detection tuning
CHECK_EVERY_SEC   = 3     # how often to analyze a frame
EMPTY_HOLD_SEC    = 15    # spot must stay empty this long before notifying
CONF              = 0.35  # detection confidence threshold
OVERLAP_TO_OCCUPY = 0.15  # fraction of a spot covered by a car to call it "occupied"
VEHICLE_CLASSES   = {"car", "truck", "bus", "motorcycle"}
# ----------------------------------------------------------------------


def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)


def grab_frame(cap):
    ok, frame = cap.read()
    return frame if ok else None


def pick_spots(frame):
    """Click two opposite corners per spot; 's' saves, 'q' quits."""
    spots, pts = [], []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                spots.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                pts.clear()

    cv2.namedWindow("setup")
    cv2.setMouseCallback("setup", on_mouse)
    print("Click two opposite corners for each spot. 's'=save, 'q'=quit.")
    while True:
        disp = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(spots):
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, str(i), (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("setup", disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("s"):
            json.dump(spots, open(SPOTS_FILE, "w"))
            print(f"Saved {len(spots)} spots to {SPOTS_FILE}")
            break
        if k == ord("q"):
            break
    cv2.destroyAllWindows()


def box_overlap_fraction(spot, car):
    """Fraction of the SPOT area covered by the car box."""
    sx1, sy1, sx2, sy2 = spot
    cx1, cy1, cx2, cy2 = car
    ix1, iy1 = max(sx1, cx1), max(sy1, cy1)
    ix2, iy2 = min(sx2, cx2), min(sy2, cy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    spot_area = (sx2 - sx1) * (sy2 - sy1)
    return inter / spot_area if spot_area else 0.0


def main():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise SystemExit("Could not open RTSP stream. Check RTSP_URL / camera account.")

    if DRAW_SETUP:
        frame = None
        for _ in range(10):           # discard first few (often blank) frames
            frame = grab_frame(cap)
        if frame is None:
            raise SystemExit("No frame received for setup.")
        pick_spots(frame)
        cap.release()
        return

    if not os.path.exists(SPOTS_FILE):
        raise SystemExit(f"{SPOTS_FILE} not found. Run once with DRAW_SETUP=True.")
    spots = json.load(open(SPOTS_FILE))
    print(f"Monitoring {len(spots)} spots.")

    model = YOLO("yolov8n.pt")        # small, auto-downloads on first run
    occupied  = [True] * len(spots)   # assume occupied at start (no false "empty")
    empty_since = [None] * len(spots)
    notified_empty = [False] * len(spots)

    send_telegram("Parking notifier started ✅")

    while True:
        frame = grab_frame(cap)
        if frame is None:
            time.sleep(2)
            cap.open(RTSP_URL)        # try to reconnect
            continue

        results = model(frame, conf=CONF, verbose=False)[0]
        cars = []
        for b in results.boxes:
            name = model.names[int(b.cls)]
            if name in VEHICLE_CLASSES:
                cars.append([float(v) for v in b.xyxy[0]])

        now = time.time()
        for i, spot in enumerate(spots):
            covered = max((box_overlap_fraction(spot, c) for c in cars), default=0.0)
            is_occupied = covered >= OVERLAP_TO_OCCUPY

            if is_occupied:
                occupied[i] = True
                empty_since[i] = None
                notified_empty[i] = False
            else:
                if empty_since[i] is None:
                    empty_since[i] = now
                held = now - empty_since[i]
                if held >= EMPTY_HOLD_SEC and not notified_empty[i]:
                    send_telegram(f"🅿️ Parking spot #{i} is now FREE!")
                    notified_empty[i] = True
                    occupied[i] = False

        time.sleep(CHECK_EVERY_SEC)


if __name__ == "__main__":
    main()
