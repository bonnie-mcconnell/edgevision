"""Snapshot-based webcam preview with the alert zone drawn on top, works
without GUI support (this project uses opencv-python-headless, which has
no cv2.imshow). Saves a frame to disk each time you press Enter, open
preview.jpg in Photos (or any viewer) and re-open it after each capture
to see current position relative to the zone.

Not part of the actual detection pipeline, just a framing aid. Don't run
this at the same time as the real uvicorn server bc both would try to open
the same webcam device.

    python -m scripts.preview_zone
"""

import cv2

from app.alerts import Zone

# Keep this in sync with DEFAULT_ZONE in app/main.py
ZONE = Zone(name="front_door", x1=200, y1=0, x2=440, y2=480)
OUT_PATH = "preview.jpg"


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Couldn't open webcam (index 0)")

    print(f"Saving snapshots to {OUT_PATH}. Open it in Photos and keep it open -")
    print("re-open/refresh after each capture to see your latest position.")
    try:
        while True:
            command = input("Press Enter to capture (or 'q' to quit): ")
            if command.strip().lower() == "q":
                break

            ok, frame = cap.read()
            if not ok:
                print("Couldn't read a frame, try again")
                continue

            cv2.rectangle(frame, (ZONE.x1, ZONE.y1), (ZONE.x2, ZONE.y2), (0, 255, 255), 2)
            cv2.putText(frame, "alert zone", (ZONE.x1 + 5, ZONE.y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(OUT_PATH, frame)
            print(f"Saved {OUT_PATH} - check it now")
    finally:
        cap.release()


if __name__ == "__main__":
    main()