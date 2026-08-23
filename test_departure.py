#!/usr/bin/env python3
"""
Simulate a car leaving spot #TEST_SPOT and verify the full alert pipeline:
detection logic -> Telegram photo. Uses a saved frame so it does NOT touch the
live camera or the running detector. Sends a clearly-labelled TEST message.
"""
import cv2, json, numpy as np
import parking_watch as pw   # reuse identical prep/diff/thresholds/send_photo

TEST_SPOT = 3
BASE_IMG  = "diagnostics/day_now.jpg"    # a frame where the spot is OCCUPIED

frame = cv2.imread(BASE_IMG)
spots = json.load(open("spots.json"))
x1, y1, x2, y2 = spots[TEST_SPOT]

# occupied reference (as the detector captured it)
occ_ref, occ_edge = pw.prep(frame[y1:y2, x1:x2])

# build a "car left" frame: fill the spot with flat asphalt-grey + faint noise
sim = frame.copy()
patch = np.full((y2 - y1, x2 - x1, 3), 90, np.uint8)          # flat grey
patch = cv2.add(patch, np.random.randint(0, 12, patch.shape, dtype=np.uint8))
sim[y1:y2, x1:x2] = patch

now_norm, now_edge = pw.prep(sim[y1:y2, x1:x2])
change = pw.diff(now_norm, occ_ref)
would_fire = change > pw.CHANGE_THRESH and now_edge < occ_edge * pw.EDGE_DROP

print(f"spot #{TEST_SPOT}: change={change:.3f} (need >{pw.CHANGE_THRESH}), "
      f"edge={now_edge:.3f} (need <{occ_edge*pw.EDGE_DROP:.3f})  -> "
      f"{'WOULD ALERT' if would_fire else 'no alert'}")

if would_fire:
    cv2.rectangle(sim, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(sim, f"FREE #{TEST_SPOT}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    pw.send_photo(sim, f"🧪 TEST: spot #{TEST_SPOT} 'freed' (simulated) — pipeline OK")
    print("Sent TEST photo to Telegram.")
