"""crowbuster-preview — live camera feed for aiming the laptop at the nest.

The camera can only be open in one process at a time, so stop the running
service first:

    systemctl --user stop crowbuster
    python3 preview.py
    # tilt the laptop / adjust position while watching the window
    # press 'q' inside the window to quit
    systemctl --user start crowbuster

The overlay draws:
  - centered crosshair (green)   — point this at the nest
  - rule-of-thirds grid (yellow) — frame the nest at an intersection
  - frame size + fps (top-left)
"""

import time

import cv2

CAMERA_INDEX = 0


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(
            "Cannot open camera. Is the crowbuster service still running?\n"
            "Stop it first:  systemctl --user stop crowbuster"
        )
        return

    print("Camera preview opened. Press 'q' in the window to quit.")
    last_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # rule-of-thirds grid (yellow, thin)
        for x in (w // 3, 2 * w // 3):
            cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)
        for y in (h // 3, 2 * h // 3):
            cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)

        # centered crosshair (green, thicker)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), 1)

        # fps + size
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
        last_t = now
        cv2.putText(
            frame,
            f"{w}x{h}  {fps:.1f} fps  -  press 'q' to quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("crowbuster preview — aim at the nest", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
