My fiancé has a parking strategy: send me to check. Every. Single. Time.

The problem? We live in Tel Aviv, where an open spot has the lifespan of a snowflake. So I stopped checking — and taught our street camera to text me the instant one frees up. 🅿️

The naïve plan was simple: point an AI object-detector (YOLO) at the camera, count the cars, done.

It failed. Hard.

Our camera is a cheap fisheye pointed down at the street at night. To a model trained on clean daytime photos, that warped, grainy, high-angle view is basically alien. The heaviest model I could run found 2 cars out of 8. In broad daylight, with a full row: still 2 of 8.

So I stopped asking the hard question — "is that a car?" — and asked an easier one:

"Did this spot just change?"

The camera never moves. The parking bays never move. That's a huge amount of free information. So I hand-drew a box on each spot, stored what "occupied" looks like, and watched for the moment a box's contents change AND go flat — the visual signature of a car pulling out and revealing bare asphalt. No recognition. No GPU. ~100 lines of Python.

It works. My phone now buzzes with a photo the second a spot frees up. (My fiancé is thrilled. I am unemployed as a parking scout.)

Three things this reminded me:

→ The fanciest tool isn't the right tool. A billion-parameter detector lost to a subtraction.

→ Constraints are a gift. "The camera is fixed" turned an AI problem into a simple one.

→ Shipping means surviving reality. v1 died overnight — a power cut, plus the camera's 3-connection limit got saturated. The real work was making it reconnect forever and never fall over.

Next stop: moving it off my laptop onto a $20 Raspberry Pi so it can run 24/7 in the window.

Sometimes "AI" is the wrong answer to your own question. Ask a better question. 🚗💨

#ComputerVision #Python #EdgeAI #TelAviv #EngineeringMindset #BuildInPublic
