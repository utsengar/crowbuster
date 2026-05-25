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
import contextlib
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

# Suppress noisy startup messages before importing pygame / torch
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
os.environ.setdefault("TORCH_NNPACK", "0")


@contextlib.contextmanager
def _silenced_stderr():
    """Swallow C-level stderr (e.g. NNPACK warnings) during a wrapped call."""
    devnull = open(os.devnull, "w")
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        devnull.close()

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
ALARM_SOUND = SOUNDS_DIR / "alarm.wav"  # special human-summoning alarm for habituated crows

LOOP_INTERVAL = 1                  # seconds between motion checks
TARGET_GONE_AFTER_N_EMPTY = 5      # consecutive YOLO misses before "target left"
PERSISTENT_REFIRE_SECONDS = 210    # re-fire if target stays in frame this long (stubborn crow)
HABITUATION_THRESHOLD = 2          # consecutive persistent-refires before sounding human alarm

# Turn the display off while crowbuster is running, back on when it exits.
# Disable by setting CROWBUSTER_NO_SCREEN_CONTROL=1 (headless boxes, multi-user.target).
CONTROL_SCREEN = os.environ.get("CROWBUSTER_NO_SCREEN_CONTROL", "0") != "1"
SCREEN_DISPLAY = os.environ.get("DISPLAY", ":0")
BASELINE_DETERRENT_MINUTES = 30    # play a sound this often regardless (insurance)
MAX_PLAY_SECONDS = 45              # cap sound playback (long files won't stall detection)
MAX_CAPTURES = 500                 # keep at most this many capture jpgs (~25-50 MB)
CAPTURE_PRUNE_EVERY = 20           # check captures/ folder size every Nth save
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
    with _silenced_stderr():
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


def _play_file(path: Path, log_line: str) -> None:
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    log(log_line)
    start = time.time()
    while pygame.mixer.music.get_busy():
        if time.time() - start > MAX_PLAY_SECONDS:
            pygame.mixer.music.stop()
            log(f"  (truncated playback at {MAX_PLAY_SECONDS}s)")
            break
        time.sleep(0.1)


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
    _play_file(sound, f"🚨 SPEAKER FIRED ({reason}) — playing {sound.name}")


def play_alarm(reason: str) -> None:
    """Play the dedicated human-summoning alarm. Falls back to a regular distress
    sound if alarm.wav isn't present yet."""
    if ALARM_SOUND.exists():
        _play_file(ALARM_SOUND, f"🆘 HUMAN ALARM ({reason}) — playing {ALARM_SOUND.name}")
    else:
        log(f"alarm sound not found at {ALARM_SOUND.name} — falling back to distress")
        play_distress(reason)


_save_count = 0


def save_capture(frame, label: str) -> None:
    global _save_count
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CAPTURES_DIR / f"{ts}_{label}.jpg"
    cv2.imwrite(str(path), frame)
    _save_count += 1
    if _save_count % CAPTURE_PRUNE_EVERY == 0:
        prune_captures()


def prune_captures() -> None:
    """Keep only the newest MAX_CAPTURES jpgs in captures/."""
    files = sorted(CAPTURES_DIR.glob("*.jpg"))  # ascending = oldest first
    excess = len(files) - MAX_CAPTURES
    if excess > 0:
        for f in files[:excess]:
            f.unlink(missing_ok=True)
        log(f"pruned {excess} old captures (folder capped at {MAX_CAPTURES})")


def write_heartbeat() -> None:
    HEARTBEAT_FILE.write_text(datetime.now().isoformat(timespec="seconds"))


def _set_screen(state: str) -> None:
    """Turn the display on/off via xset DPMS. Best-effort, silent if X isn't reachable."""
    if not CONTROL_SCREEN:
        return
    env = {**os.environ, "DISPLAY": SCREEN_DISPLAY}
    # Use loginctl/find current Xauthority — works whether we're in the X session or over SSH.
    xauth = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
    if xauth.exists():
        env["XAUTHORITY"] = str(xauth)
    try:
        result = subprocess.run(
            ["xset", "dpms", "force", state],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            log(f"display turned {state}")
        else:
            log(f"display control failed ({state}): {result.stderr.strip() or 'unknown'}")
    except FileNotFoundError:
        log("xset not installed — skipping screen control")
    except subprocess.TimeoutExpired:
        log(f"display control timed out ({state})")


def print_banner() -> None:
    """A friendly startup banner so you know exactly what's loaded."""
    BOLD, DIM, CYAN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[0m"
    mode = (
        f"{YELLOW}TEST MODE{RESET} {DIM}(target=human){RESET}"
        if TEST_MODE
        else f"{CYAN}production{RESET} {DIM}(target=crow){RESET}"
    )
    sound_count = len(list(SOUNDS_DIR.glob("*.mp3")))
    capture_count = len(list(CAPTURES_DIR.glob("*.jpg")))
    alarm_status = "ready" if ALARM_SOUND.exists() else f"missing ({ALARM_SOUND.name})"

    print(f"""
  {CYAN}▸{RESET} {BOLD}crowbuster{RESET} {DIM}— operation eggsafe{RESET}
  {DIM}watching the porch · saving the eggs{RESET}

  {DIM}mode      {RESET}{mode}
  {DIM}model     {RESET}{MODEL}
  {DIM}loop      {RESET}every {LOOP_INTERVAL}s
  {DIM}refire    {RESET}after {PERSISTENT_REFIRE_SECONDS}s if target persists
  {DIM}playback  {RESET}up to {MAX_PLAY_SECONDS}s per sound
  {DIM}sounds    {RESET}{sound_count} mp3 file{'s' if sound_count != 1 else ''} loaded
  {DIM}alarm     {RESET}{alarm_status} {DIM}(plays after {HABITUATION_THRESHOLD} refires){RESET}
  {DIM}captures  {RESET}{capture_count} existing (max {MAX_CAPTURES})
  {DIM}daylight  {RESET}{DAYLIGHT_START.strftime('%H:%M')} – {DAYLIGHT_END.strftime('%H:%M')}

  {CYAN}▸{RESET} {DIM}pipeline starting{RESET}
""")


def main() -> None:
    print_banner()
    log(f"crowbuster started [{'TEST' if TEST_MODE else 'production'}]")
    prune_captures()  # clean up any backlog from a long-stopped previous run

    # Translate SIGTERM into KeyboardInterrupt so the finally block restores the screen.
    def _on_sigterm(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)

    _set_screen("off")

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        log("FATAL: cannot open camera")
        _set_screen("on")
        return

    # Prime prev_frame so the first iteration computes a real diff (no fake "motion started 0.0")
    ok, prev_frame = cam.read()
    if not ok:
        prev_frame = None
    was_motion = False
    target_present = False     # set True after a confirmed crow; clears after N empty YOLO frames
    empty_yolo_count = 0
    consecutive_refires = 0    # persistent-refires in a row; resets when target leaves
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
                    consecutive_refires = 0  # reset habituation counter
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
                trigger_kind = "persistent-refire" if target_present else "rising-edge"
                if trigger_kind == "persistent-refire":
                    consecutive_refires += 1
                target_present = True
                save_capture(frame, TARGET_DESCRIPTION)

                if consecutive_refires >= HABITUATION_THRESHOLD:
                    save_capture(frame, "habituated")
                    play_alarm(
                        reason=f"{TARGET_DESCRIPTION} won't leave — "
                        f"{consecutive_refires} refires in a row"
                    )
                else:
                    play_distress(reason=f"{TARGET_DESCRIPTION}, {trigger_kind}")
                last_trigger = now
            else:
                save_capture(frame, f"{TARGET_YOLO_CLASS}_not_{TARGET_DESCRIPTION}")

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        log("shutdown requested")
    finally:
        cam.release()
        _set_screen("on")


if __name__ == "__main__":
    main()
