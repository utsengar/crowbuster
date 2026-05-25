"""crowbuster — watch the porch, scare crows, save the eggs.

Three-stage detection pipeline (cheapest first):
  1. Motion detection (cv2.absdiff)  — free, ~5ms
  2. YOLOv8n bird detection (local)  — free, ~80ms
  3. Claude vision API crow check    — paid, ~500ms

Design principle: fail toward MORE alarms, not fewer.
Every stage escalates on uncertainty. Network errors → fire speaker.

Sound files in ./sounds/ are crow distress calls from
HME Products: https://www.hmeproducts.com/sounds-download/
See ./sounds/SOURCES.txt for full attribution.
"""

import base64
import os
import random
import time
from datetime import datetime, time as dtime
from pathlib import Path

import anthropic
import cv2
import pygame
from ultralytics import YOLO

# --- Configuration ----------------------------------------------------------
HERE = Path(__file__).parent
SOUNDS_DIR = HERE / "sounds"
CAPTURES_DIR = HERE / "captures"   # frames that triggered (for review/tuning)
LOG_FILE = HERE / "events.log"
HEARTBEAT_FILE = HERE / "heartbeat"

LOOP_INTERVAL = 2                  # seconds between motion checks
COOLDOWN = 30                      # min seconds between speaker triggers
BASELINE_DETERRENT_MINUTES = 30    # play a sound this often regardless (insurance)
HEARTBEAT_SECONDS = 60             # write heartbeat file every N seconds

MOTION_THRESHOLD = 8.0             # mean blurred abs-diff above this = motion
YOLO_BIRD_CONFIDENCE = 0.25        # permissive — false positives just cost an API call
YOLO_FORCE_CHECK_EVERY = 30        # every Nth iteration, run YOLO regardless of motion
                                   #   (catches a crow that lands silently)

MODEL = "claude-haiku-4-5"
CAMERA_INDEX = 0

# Crows are diurnal — skip nighttime to save cost
DAYLIGHT_START = dtime(5, 30)
DAYLIGHT_END = dtime(20, 30)

# --- Setup -----------------------------------------------------------------
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY in your environment")

CAPTURES_DIR.mkdir(exist_ok=True)
client = anthropic.Anthropic()
pygame.mixer.init()
yolo = YOLO("yolov8n.pt")          # downloads ~6MB on first run


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def is_daylight() -> bool:
    now = datetime.now().time()
    return DAYLIGHT_START <= now <= DAYLIGHT_END


def has_motion(prev, curr) -> bool:
    """Stage 1: cheap pixel-diff motion check."""
    if prev is None:
        return True  # first frame — assume motion to seed the pipeline
    gp = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    gc = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    return float(cv2.absdiff(gp, gc).mean()) > MOTION_THRESHOLD


def detect_bird(frame) -> bool:
    """Stage 2: local YOLO 'is there any bird in frame?'"""
    results = yolo(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            if yolo.names[int(box.cls)] == "bird" and float(box.conf) >= YOLO_BIRD_CONFIDENCE:
                return True
    return False


def is_crow(frame) -> bool:
    """Stage 3: Claude vision API — fails toward True (alarm)."""
    _, buf = cv2.imencode(".jpg", frame)
    image_b64 = base64.b64encode(buf).decode("utf-8")
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Is there a crow, raven, or other large dark "
                                "corvid bird in this image? If uncertain, "
                                "answer YES. Reply with only YES or NO."
                            ),
                        },
                    ],
                }
            ],
        )
        for block in response.content:
            if block.type == "text":
                return block.text.strip().upper().startswith("YES")
        log("Claude response had no text block — failing toward alarm")
        return True
    except anthropic.APIError as e:
        log(f"Claude API error: {e} — failing toward alarm")
        return True


def play_distress(reason: str) -> None:
    sounds = list(SOUNDS_DIR.glob("*.mp3"))
    if not sounds:
        log(f"No mp3 files in {SOUNDS_DIR} — cannot play deterrent")
        return
    sound = random.choice(sounds)
    pygame.mixer.music.load(str(sound))
    pygame.mixer.music.play()
    log(f"🚨 SPEAKER FIRED ({reason}) — playing {sound.name}")
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def save_capture(frame, label: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(CAPTURES_DIR / f"{ts}_{label}.jpg"), frame)


def write_heartbeat() -> None:
    HEARTBEAT_FILE.write_text(datetime.now().isoformat(timespec="seconds"))


def main() -> None:
    log(f"crowbuster started (model={MODEL}, loop={LOOP_INTERVAL}s)")
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        log("FATAL: cannot open camera")
        return

    prev_frame = None
    last_trigger = 0.0
    last_baseline = time.time()
    last_heartbeat = 0.0
    iteration = 0

    try:
        while True:
            iteration += 1
            now = time.time()

            if now - last_heartbeat > HEARTBEAT_SECONDS:
                write_heartbeat()
                last_heartbeat = now

            if not is_daylight():
                time.sleep(LOOP_INTERVAL * 5)  # poll less at night
                continue

            # Baseline insurance: play a deterrent every ~30min regardless
            if (
                now - last_baseline > BASELINE_DETERRENT_MINUTES * 60
                and now - last_trigger > COOLDOWN
            ):
                play_distress(reason="baseline")
                last_baseline = now
                last_trigger = now
                time.sleep(LOOP_INTERVAL)
                continue

            ok, frame = cam.read()
            if not ok:
                log("camera read failed; reopening")
                cam.release()
                time.sleep(2)
                cam = cv2.VideoCapture(CAMERA_INDEX)
                continue

            # Stage 1: motion (with periodic forced check for silent landings)
            motion = has_motion(prev_frame, frame)
            forced = iteration % YOLO_FORCE_CHECK_EVERY == 0
            prev_frame = frame
            if not motion and not forced:
                time.sleep(LOOP_INTERVAL)
                continue

            # Stage 2: local YOLO bird check
            if not detect_bird(frame):
                time.sleep(LOOP_INTERVAL)
                continue

            log("bird detected by YOLO — escalating to Claude")

            # Stage 3: Claude crow classification
            if is_crow(frame):
                if now - last_trigger > COOLDOWN:
                    save_capture(frame, "crow")
                    play_distress(reason="crow")
                    last_trigger = now
                else:
                    log("crow seen but still in cooldown")
            else:
                save_capture(frame, "bird_not_crow")
                log("bird detected but not a crow — leaving nesting birds alone")

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        log("shutdown requested")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
