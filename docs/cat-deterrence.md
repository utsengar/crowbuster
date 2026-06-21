# Why cats are different from crows

This project started as a crow deterrent — predator-call audio played through a porch speaker to scare away corvids attacking a nest. When cat detection was added (PR #4), it turned out the playbook does **not** transfer. This doc captures the discovery so future contributors don't reach for "just play another distress call" by reflex.

> **Update (2026-06-21):** the ultrasonic strategy was tried in production and didn't work — the porch Bluetooth speaker doesn't reproduce 20 kHz at any meaningful SPL. We've switched to a dog-bark file with eyes open about habituation. See "Why we abandoned ultrasonic in practice" below. Claude refinement was also added to the cat path after a phantom-cat HUMAN ALARM (YOLO solo was insufficient).

## TL;DR

|  | Crows | Cats |
|---|---|---|
| Detection | YOLO `bird` + Claude refinement | YOLO `cat` + Claude refinement |
| Audio that works | Predator interaction (hawk fights, crow distress) | Dog bark (short-term, habituates fast) |
| Audio that BACKFIRES | — | **Crow distress** (signals prey to a cat) |
| Audio we tried and dropped | — | Ultrasonic 20 kHz (speaker can't reproduce it) |
| Habituation timeline | weeks | days for audible cues |
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

## Why we abandoned ultrasonic in practice

We tried `sounds/cat/ultrasonic-20khz.mp3` (pure 20 kHz sine, 8 s) for several weeks. The "Hardware caveat" below was the failure mode that won: the porch Bluetooth speaker doesn't reproduce frequencies that high at usable SPL, so the cat heard essentially nothing and kept showing up. We have no good way to measure speaker output above human hearing, so we couldn't tune our way out of it. The file has been removed from the repo. The `sox` generation snippets are kept below in case someone wants to retry on known-good hardware (a tweeter-rated wired speaker, an outdoor PA, etc.).

The principle still holds — sustained 18-22 kHz tones do work on cats, when the speaker can actually emit them. We just don't have that speaker.

## What we use now: dog bark (audible)

`sounds/cat/dog-bark.mp3` — an 18-second "dog barking at door" sample from freesound.org (attribution in `sounds/SOURCES.txt`). We're shipping this with eyes open about habituation: it will work for days, then degrade as the cat learns the bark isn't followed by a dog. That's fine because **the speaker was never the primary defense for cats** — the phone notification is. The bark buys us a small window of actual deterrence on top of getting our attention to physically intervene.

When the bark stops working, the next moves are layered audio (territorial hiss, see below) and hardware (motion-activated sprinkler).

### What didn't work as audio (history)

- **Ultrasonic 20 kHz**: speaker hardware limit, see above.
- **Crow-style distress** (e.g. our crow files): would *attract* cats by signalling prey. Not used.
- **Audible sirens/alarms**: cats startle once, then ignore. Not used.

### Territorial cat-hiss audio (not yet in the repo)
Cat-on-cat threat audio — angry hissing, yowling, fight sequences — should work by triggering territorial avoidance rather than predator avoidance. Different psychology than dog-bark mimicry. Not included yet because we haven't sourced clean samples; should layer on top of dog bark, not replace it.

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

- Uses Claude refinement (`use_claude: True`) — YOLO's `cat` class on its own fired phantom HUMAN ALARMs on raccoons, dark plush items, and shadows on textiles. Claude vetoes those before the refire counter can escalate to the human alarm.
- Fires the phone alert immediately on rising-edge
- Runs `active_hours: "always"` — cats are nocturnal and we can't afford to skip the night
- Habituation escalation still plays `sounds/alarm.wav` + urgent phone ping (same as crow path). With `habituation_threshold: 1`, the very first persistent refire is the human alarm — that's why the Claude check matters.

## Generation (ultrasonic — kept for historical reference)

If someone retries ultrasonic on hardware that can actually reproduce >18 kHz (a tweeter-rated wired speaker, an outdoor PA, etc.), regenerate with:

```bash
# 20kHz, 8 seconds
sox -n -r 48000 -c 1 sounds/cat/ultrasonic-20khz.mp3 synth 8 sine 20000 vol 0.9

# 18kHz fallback for speakers that roll off above 19kHz
sox -n -r 48000 -c 1 sounds/cat/ultrasonic-18khz.mp3 synth 8 sine 18000 vol 0.9

# Sweep 18→22kHz, may catch a wider response curve
sox -n -r 48000 -c 1 sounds/cat/ultrasonic-sweep.mp3 synth 10 sine 18000-22000 vol 0.9
```

The script's shuffle deck picks one file at random per fire. With the audible dog bark in play, dropping an ultrasonic file in would mix the two — fine if you want to A/B them.

## Open questions / future work

- **Bark habituation curve**: track whether cat captures pick back up 1-2 weeks after deploying the bark. Expected; that's when the territorial-hiss layer becomes the next move.
- **Territorial-hiss layer**: source 2-3 short hiss/yowl clips from freesound.org, drop into `sounds/cat/`. Should layer on top of dog bark, not replace it. Different psychology (territorial avoidance vs predator avoidance).
- **Motion-activated sprinkler**: per FERA and HSUS research, this is the gold-standard cat deterrent. Could be triggered by crowbuster's cat detection via a smart plug or relay. Out of scope for the audio-only branch but the right next move if audio underwhelms.
- **Effectiveness data**: track whether `_cat.jpg` captures decrease over the weeks after deployment. The events.log + captures dir already produce the data; needs a small analysis script.

## Further reading

- Human hearing tops out around 16-20kHz (adults). Cat hearing tops out around 64kHz.
- Sustained-tone discomfort threshold for cats: ~18kHz
- Project's crow audio philosophy: README "How it works" section
