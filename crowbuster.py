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

# Disable NNPACK before torch imports — old CPUs lack the required instructions
os.environ.setdefault("TORCH_NNPACK", "0")

import anthropic
import cv2
import pygame
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()  # picks up ANTHROPIC_API_KEY from .env in the script's directory

# --- Configuration ----------------------------------------------------------
HERE = Path(__file__).parent
SOUNDS_DIR = HERE / "sounds"
CAPTURES_DIR = HERE / "captures"   # frames that triggered (for review/tuning)
LOG_FILE = HERE / "events.log"
HEARTBEAT_FILE = HERE / "heartbeat"

LOOP_INTERVAL = 1                  # seconds between motion checks
TARGET_GONE_AFTER_N_EMPTY = 5      # consecutive YOLO misses before "target left"
PERSISTENT_REFIRE_SECONDS = 180    # re-fire if target stays in frame this long (stubborn crow)
BASELINE_DETERRENT_MINUTES = 30    # play a sound this often regardless (insurance)
MAX_PLAY_SECONDS = 15              # cap sound playback (long files won't stall detection)
HEARTBEAT_SECONDS = 60             # write heartbeat file every N seconds
STATS_INTERVAL_SECONDS = 300       # log pipeline stats every N seconds

MOTION_THRESHOLD = 8.0             # mean blurred abs-diff above this = motion
YOLO_BIRD_CONFIDENCE = 0.25        # permissive — false positives just cost an API call
YOLO_FORCE_CHECK_EVERY = 30        # every Nth iteration, run YOLO regardless of motion
                                   #   (catches a crow that lands silently)

MODEL = "claude-haiku-4-5"
CAMERA_INDEX = 0

# Test mode — detect humans instead of crows so you can stand in front of the
# camera and verify the full pipeline works. Enable with: CROWBUSTER_TEST=1
TEST_MODE = os.environ.get("CROWBUSTER_TEST", "0") == "1"
TARGET_YOLO_CLASS = "person" if TEST_MODE else "bird"
TARGET_DESCRIPTION = "human" if TEST_MODE else "crow"
TARGET_PROMPT = (
    "Is there a human (person) visible in this image? "
    "Reply with only YES or NO."
    if TEST_MODE
    else "Is there a crow, raven, or other large dark corvid bird in this "
    "image? If uncertain, answer YES. Reply with only YES or NO."
)

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


def has_motion(prev, curr) -> tuple[bool, float]:
    """Stage 1: cheap pixel-diff motion check. Returns (motion, diff_value)."""
    if prev is None:
        return True, 0.0  # first frame — assume motion to seed the pipeline
    gp = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    gc = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    diff = float(cv2.absdiff(gp, gc).mean())
    return diff > MOTION_THRESHOLD, diff


def detect_bird(frame) -> tuple[bool, float]:
    """Stage 2: local YOLO. Returns (found, best_confidence) for the target class."""
    results = yolo(frame, verbose=False)
    max_conf = 0.0
    for r in results:
        for box in r.boxes:
            if yolo.names[int(box.cls)] == TARGET_YOLO_CLASS:
                conf = float(box.conf)
                if conf > max_conf:
                    max_conf = conf
    return max_conf >= YOLO_BIRD_CONFIDENCE, max_conf


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
                        {"type": "text", "text": TARGET_PROMPT},
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


_sound_deck: list[Path] = []  # shuffle-bag — cycles through all sounds before repeating


def play_distress(reason: str) -> None:
    global _sound_deck
    if not _sound_deck:
        sounds = sorted(SOUNDS_DIR.glob("*.mp3"))
        if not sounds:
            log(f"No mp3 files in {SOUNDS_DIR} — cannot play deterrent")
            return
        _sound_deck = sounds.copy()
        random.shuffle(_sound_deck)

    sound = _sound_deck.pop()
    pygame.mixer.music.load(str(sound))
    pygame.mixer.music.play()
    log(f"🚨 SPEAKER FIRED ({reason}) — playing {sound.name}")
    start = time.time()
    while pygame.mixer.music.get_busy():
        if time.time() - start > MAX_PLAY_SECONDS:
            pygame.mixer.music.stop()
            log(f"  (truncated playback at {MAX_PLAY_SECONDS}s)")
            break
        time.sleep(0.1)


def save_capture(frame, label: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(CAPTURES_DIR / f"{ts}_{label}.jpg"), frame)


def write_heartbeat() -> None:
    HEARTBEAT_FILE.write_text(datetime.now().isoformat(timespec="seconds"))


def main() -> None:
    mode = f"TEST_MODE (target={TARGET_DESCRIPTION})" if TEST_MODE else "production"
    log(f"crowbuster started [{mode}] (model={MODEL}, loop={LOOP_INTERVAL}s)")
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        log("FATAL: cannot open camera")
        return

    prev_frame = None
    was_motion = False
    target_present = False     # set True after a confirmed crow; clears after N empty YOLO frames
    empty_yolo_count = 0
    last_trigger = 0.0
    last_baseline = time.time()
    last_heartbeat = 0.0
    last_stats = time.time()
    iteration = 0
    stats = {"frames": 0, "motion": 0, "yolo_runs": 0, "birds": 0, "crows": 0}

    try:
        while True:
            iteration += 1
            now = time.time()

            if now - last_heartbeat > HEARTBEAT_SECONDS:
                write_heartbeat()
                last_heartbeat = now

            if now - last_stats > STATS_INTERVAL_SECONDS:
                log(
                    f"stats: frames={stats['frames']} motion={stats['motion']} "
                    f"yolo={stats['yolo_runs']} birds={stats['birds']} "
                    f"crows={stats['crows']}"
                )
                last_stats = now

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
            stats["frames"] += 1

            # Stage 1: motion — log every transition so you see it immediately
            motion, diff = has_motion(prev_frame, frame)
            forced = iteration % YOLO_FORCE_CHECK_EVERY == 0
            prev_frame = frame
            if motion:
                stats["motion"] += 1
                if not was_motion:
                    log(f"⏵ motion started (diff={diff:.1f})")
            elif was_motion:
                log(f"⏸ motion stopped (diff={diff:.1f})")
            was_motion = motion

            if not motion and not forced:
                time.sleep(LOOP_INTERVAL)
                continue

            # Stage 2: YOLO — log every run with timing + confidence
            stats["yolo_runs"] += 1
            t0 = time.time()
            yolo_found, yolo_conf = detect_bird(frame)
            yolo_ms = (time.time() - t0) * 1000

            if not yolo_found:
                empty_yolo_count += 1
                if target_present and empty_yolo_count >= TARGET_GONE_AFTER_N_EMPTY:
                    log(f"⏸ {TARGET_DESCRIPTION} left the frame "
                        f"(after {empty_yolo_count} empty YOLO checks)")
                    target_present = False
                log(f"  → YOLO: no {TARGET_YOLO_CLASS} "
                    f"(top conf={yolo_conf:.2f}, {yolo_ms:.0f}ms)")
                time.sleep(LOOP_INTERVAL)
                continue

            empty_yolo_count = 0
            stats["birds"] += 1
            log(f"  → YOLO: {TARGET_YOLO_CLASS} FOUND "
                f"(conf={yolo_conf:.2f}, {yolo_ms:.0f}ms)")

            # Rising-edge gate: only call Claude + fire speaker on new appearance
            # (or if target has persisted past PERSISTENT_REFIRE_SECONDS)
            if target_present and now - last_trigger < PERSISTENT_REFIRE_SECONDS:
                log(f"  → {TARGET_DESCRIPTION} still in frame — not re-firing")
                time.sleep(LOOP_INTERVAL)
                continue

            # Stage 3: Claude — log timing and verdict
            t0 = time.time()
            confirmed = is_crow(frame)
            claude_ms = (time.time() - t0) * 1000
            log(f"  → Claude: {'YES' if confirmed else 'no'} "
                f"({claude_ms:.0f}ms)")

            if confirmed:
                stats["crows"] += 1
                target_present = True
                save_capture(frame, TARGET_DESCRIPTION)
                play_distress(reason=TARGET_DESCRIPTION)
                last_trigger = now
            else:
                save_capture(frame, f"{TARGET_YOLO_CLASS}_not_{TARGET_DESCRIPTION}")

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        log("shutdown requested")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
