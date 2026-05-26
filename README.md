# crowbuster

A computer-vision sentry that watches a bird nest, identifies crows in real time, and plays distress calls through a speaker to scare them off.

> Built after two consecutive crow attacks wiped out the eggs of a small bird family nesting on our front porch. The third clutch isn't going down without a fight.

## How it works

A three-stage detection pipeline, cheapest checks first. Each stage is a filter; only frames that pass make it to the next.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Webcam     │──▶│ 1. Motion    │──▶│ 2. YOLOv8n   │──▶│ 3. Claude API   │
│  every 1s    │    │ (cv2 absdiff)│    │  "is it a    │    │  "is it a       │
│              │    │  ~5ms, free  │    │   bird?"     │    │   crow?"        │
│              │    │              │    │ ~140ms, free │    │ ~600ms, paid    │
└──────────────┘    └──────┬───────┘    └──────┬───────┘    └────────┬────────┘
                            │ no                │ no                  │ yes
                            ▼                   ▼                     ▼
                          skip                skip          🚨 Play distress
                                                              call through
                                                              Bluetooth speaker
```

### Rising-edge triggering

The speaker fires when a target **first appears** in the frame, not continuously while it's there:

- Crow lands → 🚨 SPEAKER FIRED
- Crow stays in frame → silence (Claude API is *not* called — saves cost)
- Crow leaves (5+ empty YOLO frames in a row) → state resets
- Crow returns → 🚨 SPEAKER FIRED again

Stubborn-crow insurance: if a crow refuses to leave and stays in frame longer than `PERSISTENT_REFIRE_SECONDS` (default: 3 minutes), the speaker fires again.

### Design principle: fail toward more alarms, not fewer

A false positive plays an extra speaker sound. A false negative means dead eggs. So every stage in the pipeline escalates on uncertainty:

- Motion borderline → run YOLO anyway every 30th frame (catches silent landings)
- YOLO confidence low (0.25 threshold) → still escalate to Claude
- Claude API errors, network out, or unclear response → **fire the speaker anyway**

### Habituated-crow escalation

Crows learn fast. After ~10–20 exposures to the same stimulus, they figure out the speaker is harmless and ignore it. crowbuster detects this and escalates:

1. **First detection** — random distress sound (rising-edge fire)
2. **3.5 minutes later, crow still there** — different distress sound (persistent-refire #1)
3. **7 minutes in, still there** — `🆘 HUMAN ALARM` plays `sounds/alarm.wav` — a sound *you* will recognize and respond to by physically going outside

Add `sounds/alarm.wav` to enable. The script falls back to a regular distress sound if it's missing. The counter resets the moment the crow leaves the frame, so a determined-but-mobile crow won't trigger the alarm; only a truly habituated one that refuses to budge does.

### Safeguards that work even if detection is broken

- **Heartbeat file** — script writes `./heartbeat` every 60s. If the file goes stale, you know the script died.
- **Auto camera reopen** — if the camera read fails mid-loop, the script reopens it.
- **`@reboot` cron** — script auto-restarts on boot.
- **Captured frames** — every triggering frame is saved to `./captures/` with a timestamp, so you can review what tripped the system and retune.
- **Shuffle-bag audio rotation** — cycles through every mp3 in `./sounds/` before any repeats, then reshuffles. Defeats crow habituation faster than pure random.

### What you'll see in the logs

```
[15:20:01] crowbuster started [production] (model=claude-haiku-4-5, loop=1s)
[15:20:03] ⏵ motion started (diff=12.4)
[15:20:03]   → YOLO: bird FOUND (conf=0.87, 142ms)
[15:20:04]   → Claude: YES (612ms)
[15:20:04] 🚨 SPEAKER FIRED (crow) — playing 116-Crow & Hawk Fight.mp3
[15:20:08]   → YOLO: bird FOUND (conf=0.91, 138ms)
[15:20:08]   → crow still in frame — not re-firing
[15:20:35] ⏸ motion stopped (diff=2.1)
[15:20:45] ⏸ crow left the frame (after 5 empty YOLO checks)
[15:25:00] stats: frames=300 motion=42 yolo=18 birds=4 crows=2
```

Every 5 minutes a stats line summarizes pipeline activity, so you can verify the script is alive even when there's nothing to report.

## Hardware

- A spare laptop with a webcam (this runs on a 2012-era ThinkPad)
- A Bluetooth speaker placed near the nest
- Wi-Fi
- No Raspberry Pi, soldering, or enclosure required

## Setup

### 1. Install system dependencies (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv \
                    ffmpeg libsdl2-mixer-2.0-0
```

### 2. Install Python dependencies

```bash
git clone https://github.com/<your-fork>/crowbuster.git
cd crowbuster
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Heads up: `ultralytics` pulls in PyTorch, which is a ~500MB download. First run also auto-downloads the YOLOv8n model (~6MB). On a CPU-only old laptop, you can reclaim ~3GB by reinstalling torch as CPU-only:
> ```bash
> pip uninstall -y torch torchvision nvidia-*
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 3. Sound files

A few crow distress mp3s ship in `./sounds/`. Add more for better variety — the shuffle-bag picks them up automatically. Good sources:

- [HME Products — Free Predator Calls](https://www.hmeproducts.com/sounds-download/)
- [xeno-canto.org](https://www.xeno-canto.org) — search "Corvus brachyrhynchos alarm" (Creative Commons licensed)

Mix in hawk and owl calls — crows fear those too, and variety defeats habituation. See `sounds/SOURCES.txt` for attribution.

**Audio length:** Aim for 5–15 second clips. Crows are gone within seconds of the first burst; long sustained sounds just block detection and train crows to ignore the noise. The script caps playback at `MAX_PLAY_SECONDS` (default 15s) so longer files won't stall the pipeline, but trimming them is cleaner.

### 4. Set your Anthropic API key

Grab one at [console.anthropic.com](https://console.anthropic.com) → API Keys, then:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

`crowbuster.py` auto-loads `.env` via `python-dotenv`. `.env` is gitignored, so your key stays out of the repo. If you'd rather use a shell env var, that works too — `os.environ` takes precedence.

### 5. Pair the Bluetooth speaker

```bash
bluetoothctl
power on
scan on
# find the speaker MAC in the list
pair XX:XX:XX:XX:XX:XX
connect XX:XX:XX:XX:XX:XX
trust XX:XX:XX:XX:XX:XX
exit
```

Set it as default audio output in your sound settings.

### 6. Test the pipeline end-to-end

Before pointing it at a real nest, verify each stage works using test mode — it swaps the target from "crow" to "human" so you can stand in front of the camera:

```bash
CROWBUSTER_TEST=1 python3 crowbuster.py
```

You'll see `[TEST_MODE (target=human)]` in the startup log. Step in/out of frame and confirm:

| Stage | What you'll see | Proves |
|---|---|---|
| Motion | `⏵ motion started (diff=12.4)` | Camera + frame diff working |
| YOLO | `→ YOLO: person FOUND (conf=0.87, 142ms)` | Local model loaded, person class detected |
| Claude | `→ Claude: YES (612ms)` | API key + network working |
| Speaker | `🚨 SPEAKER FIRED (human)` | Bluetooth speaker + audio path working |

Walk out of frame, wait 5+ seconds, walk back in — speaker should fire again (rising-edge). If it fires while you're stationary, that's the persistent-refire safety kicking in after 3 minutes.

### 7. Run for real

```bash
python3 crowbuster.py
```

Logs go to stdout and `events.log`. Triggering frames go to `captures/`. Ctrl+C to stop.

## Configuration

Tweak the constants at the top of `crowbuster.py`:

| Setting | Default | Notes |
|---|---|---|
| `LOOP_INTERVAL` | `1` | Seconds between motion checks. Lower = faster reaction, more CPU. |
| `TARGET_GONE_AFTER_N_EMPTY` | `5` | Consecutive YOLO misses before considering the target gone (resets rising-edge). |
| `PERSISTENT_REFIRE_SECONDS` | `210` | Re-fire if a target stays in frame this long. Stubborn-crow insurance. |
| `HABITUATION_THRESHOLD` | `2` | Persistent-refires in a row before playing `sounds/alarm.wav` to summon a human. |
| `MAX_PLAY_SECONDS` | `45` | Truncate long audio files; keeps detection loop responsive. |
| `MAX_CAPTURES` | `500` | Cap on `captures/` folder size (~25–50 MB). Oldest pruned first. |
| `CAPTURE_PRUNE_EVERY` | `20` | Check folder size every Nth save (avoids per-save filesystem stat). |
| `MOTION_THRESHOLD` | `3.5` | Mean blurred abs-diff cutoff. Lower = more sensitive. Tuned down from `8.0` after a 2-day prod log showed 0 birds detected — small/distant birds barely shift the mean. If empty-porch frames start triggering YOLO too often, raise toward `5–6`; if you still miss landings, drop toward `2.5`. |
| `YOLO_BIRD_CONFIDENCE` | `0.25` | Permissive on purpose — false positives are cheap. |
| `YOLO_FORCE_CHECK_EVERY` | `30` | Run YOLO every Nth iteration even without motion (catches silent landings). |
| `STATS_INTERVAL_SECONDS` | `300` | How often to log pipeline activity summary. |
| `HEARTBEAT_SECONDS` | `60` | How often to update `./heartbeat`. |
| `MODEL` | `claude-haiku-4-5` | Upgrade to `claude-opus-4-7` if accuracy is poor. |
| `CAMERA_INDEX` | `0` | Built-in webcam. `1`, `2`, ... for USB cameras. |
| `DAYLIGHT_START` / `_END` | `5:30` / `20:30` | Sleep through the night — crows don't hunt then. |
| `TEST_MODE` | env var | Set `CROWBUSTER_TEST=1` to detect humans for testing. |
| `CONTROL_SCREEN` | env var | Set `CROWBUSTER_NO_SCREEN_CONTROL=1` to disable. By default, the script turns the display off at startup, disables the screensaver, and re-asserts the off state every 30s in a background thread (so the screensaver can't wake it). On exit (Ctrl+C, SIGTERM, or crash) the screensaver + DPMS are restored and the screen turned back on. When the script isn't running, the laptop behaves normally. |

> **Test-mode timing override:** when `CROWBUSTER_TEST=1`, `PERSISTENT_REFIRE_SECONDS` drops to 10 and `TARGET_GONE_AFTER_N_EMPTY` drops to 3. This makes the full pipeline (rising-edge → persistent-refire → habituated-crow alarm) reachable in ~30 seconds of standing in frame, instead of ~7 minutes. Production timings are unchanged.

## Cost

With rising-edge triggering, Claude is only called on new appearances, so API costs are minimal in normal operation.

| Stage | Cost |
|---|---|
| Motion + YOLO (when no bird present) | $0 |
| YOLO triggers + Claude (per bird visit) | ~$0.002 |
| Typical daily bird visits | ~10–50 |
| Estimated total | **~$1–3 / month** |

If even that's too much, raise `LOOP_INTERVAL` (slower scan) or remove the Claude stage entirely — but then YOLO will fire the speaker on any bird, scaring the nesting birds you're trying to protect. Don't do that.

## Development

### Auto-sync from a dev machine to the run host

Edit on your laptop, push to the run host (a separate Linux box like an old ThinkPad) automatically:

```bash
cp .env.example .env       # then edit .env with your remote host details
brew install fswatch       # macOS
./sync.sh
```

Every file change triggers an `rsync --delete` to the remote host within a second. `sync.sh` reads `REMOTE_USER`, `REMOTE_HOST`, and `REMOTE_PATH` from `.env` (or from your shell env).

### Run forever (auto-restart on crash, start on boot)

The repo ships a systemd user service that:

- Starts crowbuster at boot
- Restarts automatically on any crash (up to 50 times per 10 minutes)
- Keeps running when you log out (via `loginctl enable-linger`)
- Captures stdout + stderr in the journal

One-shot install on the run host:

```bash
./install-service.sh
```

That copies `crowbuster.service` to `~/.config/systemd/user/`, enables it, starts it, and enables user lingering so the service survives logouts.

Day-to-day operations:

```bash
systemctl --user status crowbuster        # current state
journalctl --user -u crowbuster -f        # tail the live log
systemctl --user restart crowbuster       # restart after editing the script
systemctl --user stop crowbuster          # pause it
systemctl --user disable --now crowbuster # uninstall the service
```

The service runs from the `.venv` so you don't need to activate it manually. Edit `crowbuster.service` if you want to change the restart policy, add environment variables, or run from a different path.

### Monitor the running service from a dev machine

Once the service is installed, you rarely need to be physically at the run host. SSH from your dev machine for everything.

**Single-shot health check** — is it running and producing fresh heartbeats?

```bash
ssh <user>@<run-host> 'systemctl --user is-active crowbuster && cat ~/crowbuster/heartbeat'
```

Expected output: `active` followed by a timestamp from the last 60 seconds. If `is-active` returns something other than `active`, or the heartbeat is more than ~2 minutes stale, the service is in trouble.

**Tail the live log:**

```bash
ssh <user>@<run-host> 'journalctl --user -u crowbuster -f'
```

**Recent activity / stats:**

```bash
ssh <user>@<run-host> 'journalctl --user -u crowbuster --since "1 hour ago" | grep -E "FIRED|stats|ALARM"'
```

**Quick capture review** — copy the most recent triggered frames back to your dev machine:

```bash
ssh <user>@<run-host> 'ls -t ~/crowbuster/captures/ | head -5' \
  | xargs -I{} scp <user>@<run-host>:~/crowbuster/captures/{} ~/Desktop/
```

**Recommended aliases** — drop these into your dev machine's `~/.zshrc` (or `~/.bashrc`):

```bash
# Reads REMOTE_USER and REMOTE_HOST from your environment so the same
# aliases work for any run host. Set them once in your shell rc:
export REMOTE_USER=your-username
export REMOTE_HOST=192.168.1.x

alias cb-status='ssh $REMOTE_USER@$REMOTE_HOST "systemctl --user is-active crowbuster && cat ~/crowbuster/heartbeat"'
alias cb-logs='ssh $REMOTE_USER@$REMOTE_HOST "journalctl --user -u crowbuster -f"'
alias cb-restart='ssh $REMOTE_USER@$REMOTE_HOST "systemctl --user restart crowbuster"'
alias cb-fires='ssh $REMOTE_USER@$REMOTE_HOST "journalctl --user -u crowbuster --since today | grep FIRED"'
alias cb-screen-on='ssh $REMOTE_USER@$REMOTE_HOST "DISPLAY=:0 xset dpms force on"'
```

After `source ~/.zshrc`, you can run `cb-status`, `cb-logs`, `cb-fires`, etc. from anywhere on your dev machine.

### Reviewing captures

`captures/` fills up with timestamped jpgs labeled `crow` (Claude confirmed) or `bird_not_crow` (Claude said no). Scroll through periodically:

```bash
ls -lt captures/ | head -20
```

If you see crows tagged `bird_not_crow`, the Claude prompt or model needs tuning. If you see lots of empty-frame triggers, raise `MOTION_THRESHOLD`. If you see crows you missed entirely, lower it.

The folder caps itself at `MAX_CAPTURES` (default 500) — oldest files are pruned automatically. You'll see `pruned N old captures` in the log when this happens.

**Browse captures from a dev machine in a web browser** — to skim recent frames visually without copying anything locally, run a one-line HTTP server on the run host:

```bash
ssh <user>@<run-host>
cd ~/crowbuster/captures
python3 -m http.server 8000
```

Then open `http://<run-host>:8000` in any browser on the same network. Click a jpg to view inline; refresh to pick up newly written frames. Ctrl+C the SSH session when done.

> Only run this on a trusted LAN — `http.server` has no authentication.

### Performance on old hardware

On a 2012 ThinkPad (4-core Intel, 4GB RAM, no GPU):
- YOLO inference: ~140ms per frame
- Claude API round-trip: ~600ms
- Idle CPU: ~5–10%
- Sustained RAM: ~800MB

Plenty of headroom for the box to keep doing other things. To squeeze more performance: switch to multi-user boot (`sudo systemctl set-default multi-user.target`) and skip the GUI entirely.

## Aiming the camera at the nest

Use VLC (or any video capture tool) to view the webcam feed while you tilt the laptop into position. The camera can only be open in one process at a time, so stop the service first:

```bash
systemctl --user stop crowbuster   # release the camera
vlc v4l2:///dev/video0             # or open VLC and pick Media → Open Capture Device
# aim the laptop while watching the feed, then close VLC
systemctl --user start crowbuster  # bring crowbuster back up
```

Tips for framing:
- Place the nest near one of the rule-of-thirds intersections rather than dead-center — Claude classifies better when the bird is in context
- Keep the nest in the upper third so most of the frame is empty porch, which keeps the motion baseline calm
- Aim slightly *above* the nest — crows approach from above, so you want to see the landing

## Troubleshooting

### 🆘 The screen is stuck off and I can't get it back

This is the most common worry. Recovery options, easiest first:

1. **Press any key on the laptop's physical keyboard.** Same as waking from a screensaver. Always works as long as X is alive.
2. **Remote panic button from another machine on the network:**
   ```bash
   ssh utkarsh@192.168.5.33 'DISPLAY=:0 xset dpms force on'
   ```
   Aliasing this on your dev machine is recommended:
   ```bash
   # Add to ~/.zshrc:
   alias eva-screen-on='ssh utkarsh@192.168.5.33 "DISPLAY=:0 xset dpms force on"'
   ```
3. **Ctrl+C the running script.** The `finally` block fires `xset dpms force on` automatically.
4. **Kill the script over SSH** — same effect:
   ```bash
   pkill -f crowbuster.py
   ```
5. **Reboot.** DPMS state doesn't persist across reboots.

To never have the script touch the display:
```bash
# Add to .env:
CROWBUSTER_NO_SCREEN_CONTROL=1
```

`xset dpms force off` is a runtime X server state — exactly what a screensaver does. It is not persisted, not a system config change, and survives no power cycle. You are never trapped.

### No sound plays when a target is detected

1. Verify mp3s exist:
   ```bash
   ls -la ~/crowbuster/sounds/*.mp3
   ```
   The shuffle-bag needs at least one. If empty, drop some in.

2. Confirm audio is routed to the Bluetooth speaker (not the laptop speaker):
   ```bash
   paplay ~/crowbuster/sounds/<any-file>.mp3
   ```
   Should come from the BT speaker. If not, set it as default output:
   ```bash
   pactl set-default-sink <bluez_sink_name>
   pactl list short sinks   # find the right name
   ```

3. Check the BT connection is alive:
   ```bash
   bluetoothctl info <speaker-mac>
   ```

4. The mpg123 `id3.c:process_comment` error is harmless — the audio plays even when it appears. Strip metadata to silence the warning:
   ```bash
   sudo apt install -y eyed3
   eyeD3 --remove-all sounds/*.mp3
   ```

### Camera not opening / `FATAL: cannot open camera`

1. Test outside the script:
   ```bash
   python3 -c "import cv2; c=cv2.VideoCapture(0); ok,_=c.read(); print(ok); c.release()"
   ```
   Should print `True`.

2. If `False`, check the camera isn't held by another process:
   ```bash
   fuser /dev/video0   # shows PID using it
   ```

3. Permission check — your user should be in the `video` group:
   ```bash
   groups   # look for "video"
   sudo usermod -aG video $USER   # add if missing; log out + back in
   ```

4. Try a different `CAMERA_INDEX` (some laptops list the same camera as both 0 and 1).

### `ANTHROPIC_API_KEY` not set / API errors on every fire

Either the `.env` file isn't being picked up, or the key in it is wrong:

```bash
cd ~/crowbuster
cat .env | grep ANTHROPIC   # key should start with sk-ant-
python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; print('key set:', bool(os.environ.get('ANTHROPIC_API_KEY')))"
```

If `key set: False`, the file is missing or malformed. Recreate it from `.env.example`.

### False positives: speaker keeps firing on an empty porch

1. Look at the most recent capture to see what triggered it:
   ```bash
   ls -lt captures/ | head -5
   ```
   Then `scp` it back to look at it. Common culprits: a coat on a chair, a poster of a person, your reflection in a window, the laundry on a line.

2. If YOLO is hallucinating a `bird` on an empty frame, raise its confidence threshold:
   ```python
   YOLO_BIRD_CONFIDENCE = 0.40   # was 0.25
   ```

3. If motion is firing too often (camera shake, lighting changes), raise:
   ```python
   MOTION_THRESHOLD = 6.0   # was 3.5
   ```
   The default is intentionally permissive — false motion just costs a YOLO call (~140ms, free). Only raise if YOLO is being woken constantly on an empty frame.

### False negatives: real crow visited but no fire happened

1. Check `events.log` for the time window — did motion fire? did YOLO escalate?
2. If motion didn't fire, lower `MOTION_THRESHOLD` (try `2.5`, then `2.0`). Default is `3.5`; small or distant birds against a still porch may only nudge the mean diff by 2–3. Even at the default, `YOLO_FORCE_CHECK_EVERY=30` runs YOLO every 30th frame regardless — so a totally silent landing should still get caught within ~30s. If you're seeing `motion=0` for long stretches in the stats line during clearly-active daylight hours, drop the threshold.
3. If motion fired but YOLO didn't find a bird, lower `YOLO_BIRD_CONFIDENCE`.
4. If both fired but Claude said no, try upgrading the model:
   ```python
   MODEL = "claude-opus-4-7"   # more accurate, ~5× the cost
   ```

### `NNPACK could not initialize` warnings flood the log

Suppressed by default on newer code (the YOLO call is wrapped in `_silenced_stderr`). If you still see them, you're likely on an older revision — `git pull` to update.

### Disk filling up

- `captures/` is capped at `MAX_CAPTURES` (default 500). Lower it if needed.
- `events.log` grows ~75 MB/year. Truncate without restarting:
  ```bash
  : > ~/crowbuster/events.log
  ```
- PyTorch CUDA libs eat ~3 GB on a CPU-only laptop. Reclaim:
  ```bash
  source .venv/bin/activate
  pip uninstall -y torch torchvision nvidia-* triton cuda-*
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### The script crashed, how do I see what happened?

```bash
tail -100 ~/crowbuster/events.log
```

If the process died without writing the stack trace, you may need to look at stdout where the script was launched, or the cron output file if running via `@reboot`.

### Verifying the script is running (from another machine)

```bash
ssh utkarsh@192.168.5.33 'cat ~/crowbuster/heartbeat'
```

The timestamp should be within the last 60 seconds. Stale = script died, restart with:
```bash
ssh utkarsh@192.168.5.33 'cd ~/crowbuster && nohup .venv/bin/python crowbuster.py >> events.log 2>&1 &'
```

## Credits

- Crow distress audio: [HME Products](https://www.hmeproducts.com/sounds-download/)
- Local bird detection: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- Vision classification: [Claude](https://www.anthropic.com/claude) (`claude-haiku-4-5`)
- Built with the help of [Claude Code](https://claude.com/claude-code)

## License

MIT — see `LICENSE`. Sound files in `./sounds/` retain their original rights from HME Products.
