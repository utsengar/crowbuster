# Why cats are different from crows

This project started as a crow deterrent — predator-call audio played through a porch speaker to scare away corvids attacking a nest. When cat detection was added (PR #4), it turned out the playbook does **not** transfer. This doc captures the discovery so future contributors don't reach for "just play another distress call" by reflex.

## TL;DR

|  | Crows | Cats |
|---|---|---|
| Detection | YOLO `bird` + Claude refinement | YOLO `cat` (no Claude — skips API cost) |
| Audio that works | Predator interaction (hawk fights, crow distress) | **Ultrasonic 18-22kHz only** |
| Audio that BACKFIRES | — | **Crow distress** (signals prey to a cat) |
| Habituation timeline | weeks | days for audible cues; slower for ultrasonic |
| Primary defense | Speaker (audio is the workhorse) | **Phone notification** (audio is opportunistic) |

## Why the crow strategy works

Crows recognize hawks as predators. Recordings of crow distress calls and crow-vs-hawk fights trigger instinctive flight responses because the audio carries semantic meaning to a crow's brain: "a predator is here and other crows are dying." Pure loud noise without that meaning habituates within a week. The HME predator-call files in `sounds/*.mp3` were selected for exactly this — see `sounds/SOURCES.txt`.

## Why the same strategy fails for cats

### Crow distress audio *attracts* cats
A bird's distress call signals "vulnerable prey nearby" to a cat. We were about to play 15-second clips of dying-bird audio out a porch speaker. That would have *increased* cat traffic, not decreased it. Caught this by reviewing predator-prey research before deployment — the original PR description was about to ship dog-bark and crow-style files for cats.

### Recorded dog barks habituate in days
Dogs are real cat predators, so a recording briefly works. But cats are pattern-recognition machines: within a few exposures they observe "the bark plays, no dog appears, nothing chases me," then map the area as safe. Once mapped, the bark is meaningless background noise. Same habituation curve as the crow audio, but **faster** because cats are individual learners (vs crows' social-mobbing dynamics that compound the signal across the flock).

### Audible sirens and alarms also fail
Sirens, whistles, anything in the standard audible range — cats startle once, then ignore. The cat-deterrent products on Amazon that play "scary noises" through audible speakers all have the same one-star-after-two-weeks reviews.

## What actually works for cats

### Ultrasonic 18-22kHz
Cats hear up to ~64kHz. Sustained tones become physically uncomfortable for them starting around 18kHz. Commercial ultrasonic deterrents (FERA-tested) reduce cat visits by ~46%. The mechanism is more habituation-resistant than audible cues because the frequencies are *physically* uncomfortable, not just unfamiliar — cats avoid the area instead of mapping it as safe.

**Our file**: `sounds/cat/ultrasonic-20khz.mp3` — a pure 20kHz sine, 8 seconds, generated locally with sox. See [Generation](#generation) below.

### Hardware caveat: Bluetooth speakers roll off
Most consumer Bluetooth speakers attenuate above 18-20kHz. The 20kHz tone we ship MAY come out at meaningfully reduced SPL depending on the speaker — there's no good way to know without testing on the actual hardware. Fallback: ship an 18kHz file as well; easier for speakers to reproduce, still painful to cats.

### Territorial cat-hiss audio (not yet in the repo)
Cat-on-cat threat audio — angry hissing, yowling, fight sequences — should work by triggering territorial avoidance rather than predator avoidance. Different psychology than dog-bark mimicry. Not included yet because we haven't sourced clean samples; would layer on top of ultrasonic, not replace it.

## How this changes the system design

For crows, the defense chain is:

```
Detect → Play predator audio → (rare) habituation → human alarm
```

Audio is the workhorse; the human alarm is fallback.

For cats, the chain effectively inverts:

```
Detect → Phone notification (primary) → Ultrasonic best-effort from speaker → Human goes outside and intervenes
```

Audio is opportunistic. The phone ping is the **real** defense, because a cat will not be reliably scared by anything coming out of a Bluetooth speaker. Humans physically present are the only deterrent cats consistently respect.

This is why the cat target in `crowbuster.py`:

- Skips Claude refinement (`use_claude: False`) — YOLO's `cat` class is specific; no disambiguation needed
- Fires the phone alert immediately on rising-edge
- Runs `active_hours: "always"` — cats are nocturnal and we can't afford to skip the night
- Habituation escalation still plays `sounds/alarm.wav` + urgent phone ping (same as crow path)

## Generation

To regenerate or tune the ultrasonic file:

```bash
# 20kHz, 8 seconds (current default)
sox -n -r 48000 -c 1 sounds/cat/ultrasonic-20khz.mp3 synth 8 sine 20000 vol 0.9

# 18kHz fallback for speakers that roll off above 19kHz
sox -n -r 48000 -c 1 sounds/cat/ultrasonic-18khz.mp3 synth 8 sine 18000 vol 0.9

# Sweep 18→22kHz, may catch a wider response curve
sox -n -r 48000 -c 1 sounds/cat/ultrasonic-sweep.mp3 synth 10 sine 18000-22000 vol 0.9
```

The script's shuffle deck picks one file at random per fire. Shipping multiple = variety + speaker-range hedging.

## Open questions / future work

- **Speaker reproduction**: does the porch Bluetooth speaker actually deliver 20kHz? You can't hear it — confirm by watching whether cats react over the next few weeks. If they don't, drop to 18kHz and retest.
- **Territorial-hiss layer**: source 2-3 short hiss/yowl clips from freesound.org, drop into `sounds/cat/`. Should improve effectiveness vs ultrasonic alone, especially against habituated outdoor cats that have learned the porch is "safe."
- **Motion-activated sprinkler**: per FERA and HSUS research, this is the gold-standard cat deterrent. Could be triggered by crowbuster's cat detection via a smart plug or relay. Out of scope for the audio-only branch but the right next move if ultrasonic underwhelms.
- **Effectiveness data**: track whether `_cat.jpg` captures decrease over the weeks after deployment. The events.log + captures dir already produce the data; needs a small analysis script.

## Further reading

- Human hearing tops out around 16-20kHz (adults). Cat hearing tops out around 64kHz.
- Sustained-tone discomfort threshold for cats: ~18kHz
- Project's crow audio philosophy: README "How it works" section
