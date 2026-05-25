# crowbuster

A computer-vision sentry that watches a bird nest, identifies crows in real time, and plays distress calls through a speaker to scare them off.

> Built after two consecutive crow attacks wiped out the eggs of a small bird family nesting on our front porch. The third clutch isn't going down without a fight.

## How it works

A three-stage pipeline, cheapest checks first. Each stage is a filter; only frames that pass make it to the next.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Webcam     │──▶│ 1. Motion     │──▶│ 2. YOLOv8n   │──▶│ 3. Claude API   │
│  every 2s    │    │  (cv2 absdiff)│    │   "is it a   │    │   "is it a      │
│              │    │   ~5ms, free  │    │    bird?"    │    │    crow?"       │
│              │    │               │    │  ~80ms, free │    │  ~500ms, paid   │
└──────────────┘    └──────┬───────┘    └──────┬───────┘    └────────┬────────┘
                            │ no                │ no                  │ yes
                            ▼                   ▼                     ▼
                          skip                skip          🚨 Play distress
                                                              call through
                                                              Bluetooth speaker
```

### Design principle: fail toward more alarms, not fewer

A false positive plays an extra speaker sound. A false negative means dead eggs. So every stage in the pipeline escalates on uncertainty:

- Motion borderline → run YOLO anyway every 30th frame
- YOLO confidence low (0.25 threshold) → still escalate to Claude
- Claude API errors or unclear response → **fire the speaker anyway**

### Safeguards that work even if detection is broken

- **Baseline deterrent** — plays a random distress sound every ~30 minutes regardless of detection. Crows learn the location = noise = bad. Free insurance.
- **Heartbeat file** — script writes `./heartbeat` every 60s. If the file goes stale, you know the script died.
- **Auto camera reopen** — if the camera read fails, the script reopens it.
- **`@reboot` cron** — script auto-restarts on boot.
- **Captured frames** — every triggering frame is saved to `./captures/` with a timestamp, so you can review what tripped the system and retune.

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

> Heads up: `ultralytics` pulls in PyTorch, which is a ~500MB download. First run will also auto-download the YOLOv8n model (~6MB).

### 3. Drop crow distress sounds in `./sounds/`

Download mp3s from [HME Products — Free Predator Calls](https://www.hmeproducts.com/sounds-download/) (or any source you like) and put a few in `./sounds/`. See `sounds/SOURCES.txt` for attribution notes.

### 4. Set your Anthropic API key

Grab one at [console.anthropic.com](https://console.anthropic.com) → API Keys, then:

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

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

### 6. Run it

```bash
python3 crowbuster.py
```

Logs go to stdout and `events.log`. Triggering frames go to `captures/`. Ctrl+C to stop.

## Configuration

Tweak the constants at the top of `crowbuster.py`:

| Setting | Default | Notes |
|---|---|---|
| `LOOP_INTERVAL` | `2` | Seconds between motion checks. |
| `COOLDOWN` | `30` | Min seconds between speaker triggers. |
| `BASELINE_DETERRENT_MINUTES` | `30` | Insurance sound interval. |
| `MOTION_THRESHOLD` | `8.0` | Lower = more sensitive. Tune by watching `captures/`. |
| `YOLO_BIRD_CONFIDENCE` | `0.25` | Permissive on purpose — false positives are cheap. |
| `YOLO_FORCE_CHECK_EVERY` | `30` | Run YOLO every Nth iteration even without motion (catches silent landings). |
| `MODEL` | `claude-haiku-4-5` | Upgrade to `claude-opus-4-7` if accuracy is poor. |
| `CAMERA_INDEX` | `0` | Built-in webcam. `1`, `2`, ... for USB cameras. |
| `DAYLIGHT_START` / `_END` | `5:30` / `20:30` | Sleep through the night — crows don't hunt then. |

## Cost

With the three-stage pipeline, you typically only call the API a handful of times per day (just when a bird actually lands in frame).

| Stage | Cost |
|---|---|
| Motion + YOLO (when no bird present) | $0 |
| YOLO triggers, Claude classifies | ~$0.0015 per call |
| Estimated total | **~$1–3 / month** |

If even that's too much, switch to local-only mode by removing the Claude call entirely — YOLO will fire the speaker on any bird, scaring everything (including the nesting birds). Don't do that.

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

`captures/` fills up with timestamped jpgs labeled `crow` (Claude said yes) or `bird_not_crow` (Claude said no). Scroll through periodically:

```bash
ls -lt captures/ | head -20
```

If you see crows tagged `bird_not_crow`, the Claude prompt or model needs tuning. If you see lots of empty-frame triggers, raise `MOTION_THRESHOLD`. If you see crows you missed entirely, lower it.

## Credits

- Crow distress audio: [HME Products](https://www.hmeproducts.com/sounds-download/)
- Local bird detection: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- Vision classification: [Claude](https://www.anthropic.com/claude) (`claude-haiku-4-5`)
- Built with the help of [Claude Code](https://claude.com/claude-code)

## License

MIT — see `LICENSE`. Sound files in `./sounds/` retain their original rights.
