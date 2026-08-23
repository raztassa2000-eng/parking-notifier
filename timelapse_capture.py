#!/usr/bin/env python3
"""Capture a frame from the Tapo camera every INTERVAL seconds into ./timelapse/.
Used to gather real day+night data for tuning the parking detector."""
import cv2, time, os, datetime
import parking_notifier as p

INTERVAL = 120  # seconds between captures
OUTDIR = "timelapse"
os.makedirs(OUTDIR, exist_ok=True)

def grab():
    c = cv2.VideoCapture(p.RTSP_URL, cv2.CAP_FFMPEG)
    frame = None
    for _ in range(15):
        ok, f = c.read()
        if ok:
            frame = f
            break
    c.release()
    return frame

n = 0
while True:
    frame = grab()
    if frame is not None:
        # local timestamp; safe filename
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(OUTDIR, f"{ts}.jpg"), frame)
        n += 1
        print(f"[{ts}] saved frame #{n}", flush=True)
    else:
        print("grab failed, retrying next interval", flush=True)
    time.sleep(INTERVAL)
