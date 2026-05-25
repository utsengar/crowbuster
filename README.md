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

- **Baseline deterrent** — plays a random distress sound every ~30 minutes regardless of detection. Crows learn the location = noise = bad. Free insurance against the entire pipeline failing.
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
| `BASELINE_DETERRENT_MINUTES` | `30` | Random distress sound every N minutes regardless of detection. |
| `MAX_PLAY_SECONDS` | `45` | Truncate long audio files; keeps detection loop responsive. |
| `MAX_CAPTURES` | `500` | Cap on `captures/` folder size (~25–50 MB). Oldest pruned first. |
| `CAPTURE_PRUNE_EVERY` | `20` | Check folder size every Nth save (avoids per-save filesystem stat). |
| `MOTION_THRESHOLD` | `8.0` | Lower = more sensitive. Tune by watching `captures/`. |
| `YOLO_BIRD_CONFIDENCE` | `0.25` | Permissive on purpose — false positives are cheap. |
| `YOLO_FORCE_CHECK_EVERY` | `30` | Run YOLO every Nth iteration even without motion (catches silent landings). |
| `STATS_INTERVAL_SECONDS` | `300` | How often to log pipeline activity summary. |
| `HEARTBEAT_SECONDS` | `60` | How often to update `./heartbeat`. |
| `MODEL` | `claude-haiku-4-5` | Upgrade to `claude-opus-4-7` if accuracy is poor. |
| `CAMERA_INDEX` | `0` | Built-in webcam. `1`, `2`, ... for USB cameras. |
| `DAYLIGHT_START` / `_END` | `5:30` / `20:30` | Sleep through the night — crows don't hunt then. |
| `TEST_MODE` | env var | Set `CROWBUSTER_TEST=1` to detect humans for testing. |

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

### Run on boot

```bash
crontab -e
# add:
@reboot sleep 30 && cd /home/$USER/crowbuster && \
  .venv/bin/python crowbuster.py >> events.log 2>&1
```

### Reviewing captures

`captures/` fills up with timestamped jpgs labeled `crow` (Claude confirmed) or `bird_not_crow` (Claude said no). Scroll through periodically:

```bash
ls -lt captures/ | head -20
```

If you see crows tagged `bird_not_crow`, the Claude prompt or model needs tuning. If you see lots of empty-frame triggers, raise `MOTION_THRESHOLD`. If you see crows you missed entirely, lower it.

The folder caps itself at `MAX_CAPTURES` (default 500) — oldest files are pruned automatically. You'll see `pruned N old captures` in the log when this happens.

### Performance on old hardware

On a 2012 ThinkPad (4-core Intel, 4GB RAM, no GPU):
- YOLO inference: ~140ms per frame
- Claude API round-trip: ~600ms
- Idle CPU: ~5–10%
- Sustained RAM: ~800MB

Plenty of headroom for the box to keep doing other things. To squeeze more performance: switch to multi-user boot (`sudo systemctl set-default multi-user.target`) and skip the GUI entirely.

## Credits

- Crow distress audio: [HME Products](https://www.hmeproducts.com/sounds-download/)
- Local bird detection: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- Vision classification: [Claude](https://www.anthropic.com/claude) (`claude-haiku-4-5`)
- Built with the help of [Claude Code](https://claude.com/claude-code)

## License

MIT — see `LICENSE`. Sound files in `./sounds/` retain their original rights from HME Products.
