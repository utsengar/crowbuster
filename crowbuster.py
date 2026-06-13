"""crowbuster — watch the porch, scare predators, save the eggs.

Three-stage detection pipeline (cheapest first):
  1. Motion detection (cv2.absdiff)  — free, ~5ms
  2. YOLOv8n class detection (local) — free, ~80ms
  3. Claude vision refinement (opt)  — paid, ~500ms, only for ambiguous classes

Multi-target: each predator gets a config entry in TARGETS below. YOLO runs
once per frame and routes detections to per-target state machines.

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
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
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

# Turn the display off while crowbuster is running, back on when it exits.
# Disable by setting CROWBUSTER_NO_SCREEN_CONTROL=1 (headless boxes, multi-user.target).
CONTROL_SCREEN = os.environ.get("CROWBUSTER_NO_SCREEN_CONTROL", "0") != "1"
SCREEN_DISPLAY = os.environ.get("DISPLAY", ":0")
MAX_PLAY_SECONDS = 45              # cap sound playback (long files won't stall detection)
MAX_CAPTURES = 500                 # keep at most this many capture jpgs (~25-50 MB)
CAPTURE_PRUNE_EVERY = 20           # check captures/ folder size every Nth save
HEARTBEAT_SECONDS = 60             # write heartbeat file every N seconds
STATS_INTERVAL_SECONDS = 300       # log pipeline stats every N seconds

MOTION_THRESHOLD = 3.5             # mean blurred abs-diff above this = motion
                                   #   tuned down from 8.0 after prod logs showed
                                   #   real birds weren't budging the score enough.
                                   #   Small/distant birds barely shift the mean —
                                   #   prefer false motion (cheap, just runs YOLO)
                                   #   over missing a landing (dead eggs).
YOLO_FORCE_CHECK_EVERY = 30        # every Nth iteration, run YOLO regardless of motion
                                   #   (catches a target that lands silently)

MODEL = "claude-haiku-4-5"
CAMERA_INDEX = 0

# Daylight window — used by targets with active_hours="daylight"
DAYLIGHT_START = dtime(5, 30)
DAYLIGHT_END = dtime(20, 30)

# --- Targets ----------------------------------------------------------------
# One entry per predator. To add a new animal:
#   1. Pick a YOLO class from yolo.names (run the model + print names — common
#      ones: bird=14, cat=15, dog=16, bear=21).
#   2. Drop deterrent mp3s into sounds/<key>/ (or keep at sounds/ for crow).
#   3. Add a dict entry below. Restart the service.
#
# Each target runs an independent state machine: motion → YOLO → optional
# Claude → fire. They share one camera and one YOLO inference per frame.
#
# Fields:
#   yolo_class                 — what YOLO calls it
#   label                      — display name in logs / capture filenames
#   min_confidence             — YOLO confidence threshold (0.0–1.0)
#   sounds_dir                 — directory of *.mp3 deterrents (shuffle deck)
#   use_claude                 — run Claude vision to refine YOLO? Only useful
#                                when the YOLO class is broader than the actual
#                                target (e.g. bird→crow). For specific classes
#                                like cat/dog, set False — saves API cost.
#   claude_prompt              — required if use_claude=True
#   active_hours               — "daylight" (5:30–20:30) or "always" (24/7)
#   persistent_refire_seconds  — re-fire if target lingers this long
#   habituation_threshold      — consecutive refires before HUMAN ALARM
TARGETS: dict[str, dict] = {
    "crow": {
        "yolo_class": "bird",
        "label": "crow",
        "min_confidence": 0.25,
        "sounds_dir": HERE / "sounds",      # flat layout — existing crow mp3s
        "use_claude": True,                 # YOLO "bird" includes the resident
                                            # parents; Claude refines to corvid
        "claude_prompt": (
            "Is there a crow, raven, or other large dark corvid bird in this "
            "image? If uncertain, answer YES. Reply with only YES or NO."
        ),
        "active_hours": "daylight",
        "persistent_refire_seconds": 210,
        "habituation_threshold": 2,
    },
    "cat": {
        "yolo_class": "cat",
        "label": "cat",
        "min_confidence": 0.25,
        "sounds_dir": HERE / "sounds" / "cat",
        "use_claude": False,                # YOLO "cat" is specific enough
        "claude_prompt": None,
        "active_hours": "always",           # cats come at midnight
        "persistent_refire_seconds": 210,
        "habituation_threshold": 2,
    },
}

# Test mode — replace TARGETS with a single "person" target so you can stand
# in front of the camera and verify the full pipeline (including Claude).
# Refire/empty thresholds are tightened so habituation alarm is reachable in
# ~30 seconds of standing in frame.
TEST_MODE = os.environ.get("CROWBUSTER_TEST", "0") == "1"
if TEST_MODE:
    TARGET_GONE_AFTER_N_EMPTY = 3
    TARGETS = {
        "person": {
            "yolo_class": "person",
            "label": "human",
            "min_confidence": 0.25,
            "sounds_dir": HERE / "sounds",
            "use_claude": True,
            "claude_prompt": (
                "Is there a human (person) visible in this image? "
                "Reply with only YES or NO."
            ),
            "active_hours": "always",
            "persistent_refire_seconds": 10,
            "habituation_threshold": 2,
        }
    }

# Push notifications via ntfy.sh — fires on every confirmed detection
# (rising-edge fire = default priority; HUMAN ALARM = urgent priority).
# Same target lingering = one ping, not many. Leave topic unset to disable.
#   1. Pick a hard-to-guess topic name (anyone who knows it can read your alerts).
#   2. Install the "ntfy" app on your phone, subscribe to the topic.
#   3. Set CROWBUSTER_NTFY_TOPIC=<your-topic> in .env or the environment.
# Self-hosted: set CROWBUSTER_NTFY_SERVER=https://ntfy.yourdomain.com
NTFY_TOPIC = os.environ.get("CROWBUSTER_NTFY_TOPIC", "").strip()
NTFY_SERVER = os.environ.get("CROWBUSTER_NTFY_SERVER", "https://ntfy.sh").rstrip("/")

# Periodic health-check pings to ntfy so you know the system is alive even
# when no predators have shown up in a while. Default: every 12h. Absence of
# the ping is the signal — if you don't see a "🟢 operational" notification
# for >24h, the laptop/Wi-Fi/sshd has probably died and you need to go
# investigate physically.
HEARTBEAT_PING_HOURS = float(os.environ.get("CROWBUSTER_HEARTBEAT_HOURS", "12"))
CAMERA_FAILURE_ALERT_AFTER = 5  # consecutive camera-reopen failures before urgent ntfy

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


def detect_targets(frame) -> dict[str, float]:
    """Stage 2: local YOLO, multi-class. Returns {target_key: best_confidence}
    for every configured TARGET whose yolo_class shows up above its threshold.
    One inference pass — costs the same whether we have 1 or 5 targets."""
    with _silenced_stderr():
        results = yolo(frame, verbose=False)
    found: dict[str, float] = {}
    for r in results:
        for box in r.boxes:
            cls_name = yolo.names[int(box.cls)]
            conf = float(box.conf)
            for key, cfg in TARGETS.items():
                if cls_name == cfg["yolo_class"] and conf >= cfg["min_confidence"]:
                    if conf > found.get(key, 0.0):
                        found[key] = conf
    return found


def is_target_via_claude(frame, prompt: str) -> bool:
    """Stage 3 (optional, per-target): Claude vision refinement.
    Fails toward True (alarm) on any error."""
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
                        {"type": "text", "text": prompt},
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


@dataclass
class TargetState:
    """Per-target runtime state. One instance per entry in TARGETS.
    Owns its own shuffle-deck of deterrent sounds so multiple targets don't
    fight over a shared deck."""
    key: str
    cfg: dict
    target_present: bool = False
    empty_yolo_count: int = 0
    consecutive_refires: int = 0
    last_trigger: float = 0.0
    sound_deck: list[Path] = field(default_factory=list)
    _missing_sounds_warned: bool = False

    def is_active_now(self) -> bool:
        if self.cfg["active_hours"] == "always":
            return True
        return is_daylight()

    def play_distress(self, reason: str) -> None:
        if not self.sound_deck:
            sounds = sorted(self.cfg["sounds_dir"].glob("*.mp3"))
            if not sounds:
                # Log once per "empty deck" event — re-checks every cycle so
                # files dropped in mid-run get picked up automatically.
                if not self._missing_sounds_warned:
                    log(f"no mp3 files in {self.cfg['sounds_dir']} — "
                        f"cannot fire {self.cfg['label']} deterrent")
                    self._missing_sounds_warned = True
                return
            self.sound_deck = sounds.copy()
            random.shuffle(self.sound_deck)
            self._missing_sounds_warned = False
        sound = self.sound_deck.pop()
        _play_file(sound, f"🚨 SPEAKER FIRED ({reason}) — playing {sound.name}")

    def play_alarm(self, reason: str) -> None:
        """Human-summoning alarm. Falls back to this target's distress sounds
        if alarm.wav isn't present. Phone notification is sent at the call
        site, not here — see send_alert()."""
        if ALARM_SOUND.exists():
            _play_file(ALARM_SOUND,
                       f"🆘 HUMAN ALARM ({reason}) — playing {ALARM_SOUND.name}")
        else:
            log(f"alarm sound not found at {ALARM_SOUND.name} — falling back to distress")
            self.play_distress(reason)


def send_alert(title: str, message: str, frame=None, priority: str = "default",
               tag: str = "owl") -> None:
    """Push a notification to the configured ntfy topic, with the triggering
    frame attached as a JPEG. No-op if no topic is configured. Best-effort —
    a notification failure must never break the detection loop.

    Priority: "default" for routine detections (your phone settings decide
    whether to make sound), "urgent" for HUMAN ALARM (breaks through DND)."""
    if not NTFY_TOPIC:
        return
    headers = {
        "Title": title,
        "Priority": priority,
        "Tags": tag,
        "Message": message,
    }
    body = b""
    if frame is not None:
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            body = buf.tobytes()
            headers["Filename"] = f"detect-{int(time.time())}.jpg"
    try:
        req = urllib.request.Request(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        log(f"📲 phone alert sent to ntfy/{NTFY_TOPIC} ({priority})")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        log(f"ntfy alert failed: {e}")


# --- Health-check pings -----------------------------------------------------
# Three signals tell you the system is OK or in trouble:
#   1. Startup ping  — confirms crowbuster came up cleanly (after reboot/crash)
#   2. Heartbeat ping — periodic "alive" check-in; absence = something is wrong
#   3. Shutdown ping — confirms clean exit (ctrl+c, systemd stop, or crash)
# Plus an urgent alert on persistent camera failure (script is alive but blind).

def _format_uptime(seconds: float) -> str:
    hours = seconds / 3600
    if hours < 1:
        return f"{seconds / 60:.0f}m"
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def send_startup_ping() -> None:
    """Confirm crowbuster is up. Low priority — informational."""
    targets = ", ".join(TARGETS.keys()) or "(none)"
    mode = "TEST" if TEST_MODE else "production"
    send_alert(
        title="🟢 crowbuster started",
        message=f"Pipeline online [{mode}]. Watching: {targets}.",
        priority="low",
        tag="green_circle",
    )


def send_heartbeat_ping(stats: dict, uptime_seconds: float) -> None:
    """Periodic 'still alive' check-in. Low priority so it doesn't interrupt
    you; the signal is the regular cadence, not any one ping. Stops arriving
    when the laptop/Wi-Fi/sshd dies — that's how you know to investigate."""
    detail = " · ".join(
        f"{k} {stats.get(f'{k}_found', 0)}/{stats.get(f'{k}_fired', 0)}"
        for k in TARGETS
    )
    send_alert(
        title="🟢 crowbuster operational",
        message=f"Up {_format_uptime(uptime_seconds)}. "
                f"Frames={stats.get('frames', 0)} YOLO={stats.get('yolo_runs', 0)}. "
                f"Per-target found/fired: {detail}.",
        priority="low",
        tag="heart",
    )


def send_shutdown_ping(reason: str) -> None:
    """Notify on exit so you know if systemd kept it down. Default priority."""
    send_alert(
        title="🟡 crowbuster stopped",
        message=f"Process exiting: {reason}. "
                f"If you didn't stop it, check systemd / Wi-Fi / disk space.",
        priority="default",
        tag="yellow_circle",
    )


def send_camera_failure_alert(consecutive: int) -> None:
    """Camera is dead — pipeline can't see anything. URGENT (DND override)."""
    send_alert(
        title="🚨 crowbuster — CAMERA DEAD",
        message=f"{consecutive} consecutive camera read failures. "
                f"Detection is BLIND. Check the USB cam / /dev/video0.",
        priority="urgent",
        tag="warning",
    )


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


_screen_keeper_stop = threading.Event()
_screen_keeper_thread: threading.Thread | None = None


def _xset(*args: str) -> bool:
    """Run an xset command, best-effort. Returns True on success."""
    env = {**os.environ, "DISPLAY": SCREEN_DISPLAY}
    # Try common Xauthority locations so this works whether launched from inside
    # the X session or over SSH.
    for candidate in (
        os.environ.get("XAUTHORITY"),
        f"/run/user/{os.getuid()}/gdm/Xauthority",
        f"/home/{os.environ.get('USER', '')}/.Xauthority",
    ):
        if candidate and Path(candidate).exists():
            env["XAUTHORITY"] = candidate
            break
    try:
        return subprocess.run(
            ["xset", *args], env=env, capture_output=True, text=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _screen_keeper_loop() -> None:
    """Re-issue dpms-force-off every 30s — survives X events that wake the screen."""
    while not _screen_keeper_stop.is_set():
        _xset("dpms", "force", "off")
        _screen_keeper_stop.wait(30)


def screen_off() -> None:
    """Turn the display off and keep it off until screen_on() runs."""
    if not CONTROL_SCREEN:
        return
    # Disable the screensaver + DPMS timing so nothing fights us
    _xset("s", "off")
    _xset("s", "noblank")
    _xset("-dpms")
    ok = _xset("dpms", "force", "off")
    log(f"display turned off{'' if ok else ' (xset failed — screen may stay on)'}")
    # Start the keeper thread to re-assert off if something wakes the screen
    global _screen_keeper_thread
    _screen_keeper_stop.clear()
    _screen_keeper_thread = threading.Thread(target=_screen_keeper_loop, daemon=True)
    _screen_keeper_thread.start()


def screen_on() -> None:
    """Stop the keeper thread, restore screensaver/DPMS, turn the screen back on."""
    if not CONTROL_SCREEN:
        return
    _screen_keeper_stop.set()
    if _screen_keeper_thread is not None:
        _screen_keeper_thread.join(timeout=2)
    _xset("+dpms")          # restore DPMS power management
    _xset("s", "default")   # restore default screensaver timeout
    _xset("dpms", "force", "on")
    log("display turned on")


def print_banner() -> None:
    """A friendly startup banner so you know exactly what's loaded."""
    BOLD, DIM, CYAN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[0m"
    mode = f"{YELLOW}TEST MODE{RESET}" if TEST_MODE else f"{CYAN}production{RESET}"
    capture_count = len(list(CAPTURES_DIR.glob("*.jpg")))
    alarm_status = "ready" if ALARM_SOUND.exists() else f"missing ({ALARM_SOUND.name})"
    phone_status = f"ntfy/{NTFY_TOPIC}" if NTFY_TOPIC else "disabled (set CROWBUSTER_NTFY_TOPIC)"

    target_lines = []
    for key, cfg in TARGETS.items():
        sdir = cfg["sounds_dir"]
        sound_count = len(list(sdir.glob("*.mp3"))) if sdir.exists() else 0
        claude_note = "Claude-refined" if cfg["use_claude"] else "YOLO-only"
        target_lines.append(
            f"  {DIM}· {RESET}{BOLD}{key}{RESET} "
            f"{DIM}— yolo={cfg['yolo_class']}, {cfg['active_hours']}, "
            f"{claude_note}, {sound_count} sound{'s' if sound_count != 1 else ''}, "
            f"refire {cfg['persistent_refire_seconds']}s{RESET}"
        )

    print(f"""
  {CYAN}▸{RESET} {BOLD}crowbuster{RESET} {DIM}— operation eggsafe{RESET}
  {DIM}watching the porch · saving the eggs{RESET}

  {DIM}mode      {RESET}{mode}
  {DIM}model     {RESET}{MODEL}
  {DIM}loop      {RESET}every {LOOP_INTERVAL}s
  {DIM}playback  {RESET}up to {MAX_PLAY_SECONDS}s per sound
  {DIM}alarm     {RESET}{alarm_status}
  {DIM}phone     {RESET}{phone_status}
  {DIM}captures  {RESET}{capture_count} existing (max {MAX_CAPTURES})
  {DIM}daylight  {RESET}{DAYLIGHT_START.strftime('%H:%M')} – {DAYLIGHT_END.strftime('%H:%M')}

  {DIM}targets:{RESET}
{chr(10).join(target_lines)}

  {CYAN}▸{RESET} {DIM}pipeline starting{RESET}
""")


def main() -> None:
    print_banner()
    log(f"crowbuster started [{'TEST' if TEST_MODE else 'production'}]")
    prune_captures()  # clean up any backlog from a long-stopped previous run
    send_startup_ping()

    # Translate SIGTERM into KeyboardInterrupt so the finally block restores the screen.
    def _on_sigterm(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)

    screen_off()

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        log("FATAL: cannot open camera")
        send_shutdown_ping("FATAL: cannot open camera at startup")
        screen_on()
        return

    # Per-target state machines — independent presence/refire/cooldown bookkeeping
    states: dict[str, TargetState] = {
        key: TargetState(key=key, cfg=cfg) for key, cfg in TARGETS.items()
    }

    # Prime prev_frame so the first iteration computes a real diff
    ok, prev_frame = cam.read()
    if not ok:
        prev_frame = None
    was_motion = False
    last_heartbeat = 0.0
    last_stats = time.time()
    start_time = time.time()
    last_heartbeat_ping = time.time()      # don't fire one right at startup
    consecutive_camera_failures = 0
    camera_alert_sent = False              # latch — alert once per outage, not every retry
    iteration = 0
    stats: dict[str, int] = {"frames": 0, "motion": 0, "yolo_runs": 0}
    for key in TARGETS:
        stats[f"{key}_found"] = 0
        stats[f"{key}_fired"] = 0

    try:
        while True:
            iteration += 1
            now = time.time()

            if now - last_heartbeat > HEARTBEAT_SECONDS:
                write_heartbeat()
                last_heartbeat = now

            # Periodic "alive" ping to ntfy. Default 12h. Absence is the signal —
            # if you stop seeing these for >24h, the system died and you need
            # to investigate physically (the script can't alert when the box is
            # offline; this is the only reliable indicator of that case).
            if now - last_heartbeat_ping > HEARTBEAT_PING_HOURS * 3600:
                send_heartbeat_ping(stats, now - start_time)
                last_heartbeat_ping = now

            if now - last_stats > STATS_INTERVAL_SECONDS:
                parts = [
                    f"frames={stats['frames']}",
                    f"motion={stats['motion']}",
                    f"yolo={stats['yolo_runs']}",
                ]
                for key in TARGETS:
                    parts.append(f"{key}_found={stats[f'{key}_found']}")
                    parts.append(f"{key}_fired={stats[f'{key}_fired']}")
                log("stats: " + " ".join(parts))
                last_stats = now

            # Skip the loop entirely only if NO target is active right now.
            # With "always" targets (e.g. cat), this rarely fires.
            if not any(s.is_active_now() for s in states.values()):
                time.sleep(LOOP_INTERVAL * 5)
                continue

            ok, frame = cam.read()
            if not ok:
                consecutive_camera_failures += 1
                log(f"camera read failed (#{consecutive_camera_failures}); reopening")
                if (consecutive_camera_failures >= CAMERA_FAILURE_ALERT_AFTER
                        and not camera_alert_sent):
                    send_camera_failure_alert(consecutive_camera_failures)
                    camera_alert_sent = True
                cam.release()
                time.sleep(2)
                cam = cv2.VideoCapture(CAMERA_INDEX)
                continue
            # Camera back — reset the alert latch so a future outage notifies again
            if consecutive_camera_failures > 0:
                log(f"camera recovered after {consecutive_camera_failures} failures")
                consecutive_camera_failures = 0
                camera_alert_sent = False
            stats["frames"] += 1

            # Stage 1: motion — shared across all targets
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

            # Stage 2: YOLO — single inference, multi-class routing
            stats["yolo_runs"] += 1
            t0 = time.time()
            yolo_results = detect_targets(frame)  # {key: best_conf}
            yolo_ms = (time.time() - t0) * 1000

            if not yolo_results:
                log(f"  → YOLO: no target classes ({yolo_ms:.0f}ms)")

            # Per-target processing — every configured target gets a turn,
            # whether or not YOLO found its class this frame.
            for key, cfg in TARGETS.items():
                state = states[key]

                # Outside this target's active hours → clear any stale presence
                # so we don't fire stale persistent-refires when it reactivates.
                if not state.is_active_now():
                    if state.target_present:
                        log(f"⏸ {cfg['label']} window closed — clearing state")
                        state.target_present = False
                        state.consecutive_refires = 0
                        state.empty_yolo_count = 0
                    continue

                conf = yolo_results.get(key, 0.0)
                if conf == 0.0:
                    # Not detected this frame
                    state.empty_yolo_count += 1
                    if (state.target_present
                            and state.empty_yolo_count >= TARGET_GONE_AFTER_N_EMPTY):
                        log(f"⏸ {cfg['label']} left the frame "
                            f"(after {state.empty_yolo_count} empty YOLO checks)")
                        state.target_present = False
                        state.consecutive_refires = 0
                    continue

                # Detected this frame
                state.empty_yolo_count = 0
                stats[f"{key}_found"] += 1
                log(f"  → YOLO: {cfg['yolo_class']} FOUND "
                    f"({cfg['label']}, conf={conf:.2f}, {yolo_ms:.0f}ms)")

                # Rising-edge gate
                if (state.target_present
                        and now - state.last_trigger < cfg["persistent_refire_seconds"]):
                    log(f"  → {cfg['label']} still in frame — not re-firing")
                    continue

                # Stage 3: optional Claude refinement
                if cfg["use_claude"]:
                    t0 = time.time()
                    confirmed = is_target_via_claude(frame, cfg["claude_prompt"])
                    claude_ms = (time.time() - t0) * 1000
                    log(f"  → Claude({cfg['label']}): "
                        f"{'YES' if confirmed else 'no'} ({claude_ms:.0f}ms)")
                    if not confirmed:
                        save_capture(frame, f"{cfg['yolo_class']}_not_{cfg['label']}")
                        continue

                # Fire
                trigger_kind = "persistent-refire" if state.target_present else "rising-edge"
                if trigger_kind == "persistent-refire":
                    state.consecutive_refires += 1
                state.target_present = True
                stats[f"{key}_fired"] += 1
                save_capture(frame, cfg["label"])

                if state.consecutive_refires >= cfg["habituation_threshold"]:
                    # Stubborn target — louder phone alert, then play the
                    # human-summoning sound. send_alert fires first so the
                    # phone buzzes before audio playback blocks the loop.
                    save_capture(frame, "habituated")
                    send_alert(
                        title=f"🆘 HUMAN ALARM — {cfg['label']} won't leave",
                        message=f"{state.consecutive_refires} refires in a row. Go outside.",
                        frame=frame,
                        priority="urgent",
                        tag="rotating_light",
                    )
                    state.play_alarm(
                        reason=f"{cfg['label']} won't leave — "
                               f"{state.consecutive_refires} refires in a row"
                    )
                elif trigger_kind == "rising-edge":
                    # New arrival — log the catch on your phone (default priority,
                    # your phone decides whether to play a sound). Persistent-refires
                    # of the same target intentionally do NOT alert again — you've
                    # already been told it's there; no need to buzz twice.
                    claude_note = f", Claude {claude_ms:.0f}ms" if cfg["use_claude"] else ""
                    send_alert(
                        title=f"crowbuster — {cfg['label']} detected",
                        message=f"YOLO conf {conf:.2f}{claude_note}.",
                        frame=frame,
                        priority="default",
                        tag="owl",
                    )
                    state.play_distress(reason=f"{cfg['label']}, {trigger_kind}")
                else:
                    state.play_distress(reason=f"{cfg['label']}, {trigger_kind}")
                state.last_trigger = now

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        log("shutdown requested")
        send_shutdown_ping("Ctrl+C / SIGTERM")
    except Exception as e:
        log(f"FATAL: unhandled exception: {type(e).__name__}: {e}")
        send_shutdown_ping(f"crash: {type(e).__name__}: {e}")
        raise
    finally:
        cam.release()
        screen_on()


if __name__ == "__main__":
    main()
