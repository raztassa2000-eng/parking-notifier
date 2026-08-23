#!/usr/bin/env python3
"""
Live detection viewer (auto-stabilized, non-freezing).

A background thread continuously reads the RTSP stream and keeps only the LATEST
frame, so the display never falls behind and the stream can't back up and stall.
Frames are aligned to data/reference.jpg so boxes track the spots as the camera
drifts. RED = occupied, GREEN = empty. Read-only, sends no alerts. 'q' to quit.
"""
import cv2, json, time, threading
import numpy as np
import parking_notifier as cfg

SPOTS_FILE    = "spots.json"
REFERENCE_IMG = "data/reference.jpg"
CROP_SIZE     = 64
CHANGE_THRESH = 0.12
EDGE_DROP     = 0.70
MIN_INLIERS   = 25

orb = cv2.ORB_create(1500)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
ref = cv2.imread(REFERENCE_IMG)
ref_kp, ref_des = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
ref_size = (ref.shape[1], ref.shape[0])


class LatestFrame:
    """Continuously reads the stream in a thread, keeps only the newest frame."""
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.running = True
        self.cap = self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok or f is None:
                time.sleep(0.2)
                self.cap.release()
                self.cap = self._open()
                continue
            self.frame = f

    def read(self):
        return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


def stabilize(frame):
    if (frame.shape[1], frame.shape[0]) != ref_size:   # camera resolution changed
        frame = cv2.resize(frame, ref_size)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(g, None)
    if des is None or len(kp) < 2 * MIN_INLIERS:
        return None
    matches = bf.match(ref_des, des)
    if len(matches) < 2 * MIN_INLIERS:
        return None
    src = np.float32([kp[m.trainIdx].pt for m in matches])
    dst = np.float32([ref_kp[m.queryIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5)
    if M is None or inl is None or int(inl.sum()) < MIN_INLIERS:
        return None
    return cv2.warpAffine(frame, M, ref_size)


def prep(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (CROP_SIZE, CROP_SIZE))
    edge = float(cv2.Canny(g, 50, 150).mean()) / 255.0
    norm = g.astype(np.float32); norm -= norm.mean()
    return norm, edge


def diff(a, b):
    return float(np.abs(a - b).mean()) / 255.0


def main():
    spots = json.load(open(SPOTS_FILE))
    cam = LatestFrame(cfg.RTSP_URL)

    # wait for first stabilizable frame -> references
    refs = None
    t0 = time.time()
    while time.time() - t0 < 30:
        f = cam.read()
        if f is not None:
            a = stabilize(f)
            if a is not None:
                refs = [prep(a[y1:y2, x1:x2]) for (x1, y1, x2, y2) in spots]
                break
        time.sleep(0.1)
    if refs is None:
        cam.stop(); raise SystemExit("No stabilized frame in 30s.")

    print("Live view (stabilized, non-freezing). Press 'q' to quit.")
    while True:
        raw = cam.read()
        if raw is None:
            if (cv2.waitKey(30) & 0xFF) == ord('q'):
                break
            continue
        frame = stabilize(raw)
        if frame is None:
            frame = raw   # show unaligned rather than freeze
        disp = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(spots):
            norm, edge = prep(frame[y1:y2, x1:x2])
            rn, re = refs[i]
            empty = diff(norm, rn) > CHANGE_THRESH and edge < re * EDGE_DROP
            color = (0, 255, 0) if empty else (0, 0, 255)
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(disp, f"{i}:{'EMPTY' if empty else 'occ'} e{edge:.2f}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imshow("Parking - live (stabilized, q to quit)", disp)
        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            break
    cam.stop(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
