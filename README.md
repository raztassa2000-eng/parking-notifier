# 🅿️ Parking Notifier

Sends a Telegram photo when a street parking spot frees up, using a Tapo camera.

## How it works
The camera is fixed and the spots never move, so instead of *recognizing cars*
(off-the-shelf YOLO failed badly on this fisheye/night view — found 2 of ~8 cars),
we use **change-detection**: each hand-drawn spot box stores its "occupied" look,
and we alert when its contents change AND go flat (a car leaving reveals asphalt).

## Structure
- `parking_watch.py`   – main detector (resilient, single RTSP connection). RUN THIS.
- `parking_notifier.py`– config/secrets (RTSP URL, Telegram token/chat) + YOLO version (unused).
- `live_view.py`       – live window: red=occupied, green=empty (q to quit). Steals a camera slot.
- `test_departure.py`  – simulate a car leaving to test the alert pipeline.
- `spots.json`         – the 6 hand-drawn spot boxes.
- `data/`              – metrics_log.csv, timelapse/ frames, events/ (free-spot photos).
- `logs/`              – runtime logs.
- `diagnostics/`       – snapshots used while building/tuning.
- `models/`            – YOLO weights (unused; ~190MB, safe to delete).
- `parking_venv/`      – Python environment.

## Run
    cd parking_project
    ./parking_venv/bin/python parking_watch.py        # detector (background it with nohup)
    ./parking_venv/bin/python live_view.py            # optional live window

## Tuning knobs (top of parking_watch.py)
- CHANGE_THRESH (0.12) – how different from "parked" counts as changed
- EDGE_DROP (0.70)     – how much the busyness must fall to call it empty
- EMPTY_HOLD_SEC (7)   – how long it must stay changed before alerting
- NOTIFY_TAKEN (True)  – also ping when a spot fills again (turn off once tuned)

## Notes
- The Mac must stay on the home WiFi (reaches camera locally). Phone can be anywhere.
- Don't leave live_view.py running — it holds one of the camera's ~3 connection slots.
