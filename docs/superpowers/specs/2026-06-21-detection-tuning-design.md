# Detection tuning — cat audio swap, crow threshold, VLM prompt overhaul

**Date:** 2026-06-21
**Status:** approved, ready for implementation plan

## Problem

Three issues observed in production:

1. **Cat ultrasonic isn't deterring anything.** The porch Bluetooth speaker likely rolls off above ~19 kHz, so the 20 kHz tone we ship comes out below an effective SPL for cats. The doc (`docs/cat-deterrence.md`) anticipated this as the "speaker reproduction" open question — confirmed in the field.
2. **Crow false positives.** YOLO `bird` class fires at 0.25 confidence on small/light birds (sparrows, doves), then Claude lets them through because the current prompt explicitly biases toward YES.
3. **Phantom-cat HUMAN ALARM.** Cat path has no Claude refinement, runs `habituation_threshold: 1`, and `persistent_refire_seconds: 30`. One sustained YOLO false-positive (raccoon, dark plush, shadow) for 30 s triggers the emergency human alarm. Happened today with no cat in frame.

## Goals

- Replace the cat audio strategy with dog barking (user-supplied file).
- Reduce crow false-positive rate without losing genuine distant/occluded crows.
- Stop phantom-cat HUMAN ALARMs by adding Claude refinement to the cat path.
- Make Claude's reasoning auditable so future false positives are diagnosable from `events.log` and `captures/`.

## Non-goals

- Per-target bounding-box crops for Claude input. (Deferred — revisit after a week of reason-logged data tells us whether prompts alone are enough.)
- Territorial-hiss audio layer for cats. (Out of scope; still in the cat-deterrence open-question list.)
- Motion-activated sprinkler integration. (Out of scope.)
- Touching the existing motion/heartbeat/ntfy/screen-control machinery.

## Design

### 1. Cat audio: ultrasonic → dog bark

- Delete `sounds/cat/ultrasonic-20khz.mp3`. (Shuffle-deck logic in `TargetState.play_distress` cycles through every mp3 in `sounds_dir`; leaving ultrasonic in means it plays half the time and defeats the override.)
- Add user-supplied dog bark mp3 to `sounds/cat/` (filename: `dog-bark.mp3` unless user names it differently).
- Update `docs/cat-deterrence.md`:
  - Mark ultrasonic strategy as **abandoned**, with one-paragraph "what we learned" — Bluetooth speakers roll off as anticipated; the open question is now closed (negative result).
  - Update the TL;DR table: "Audio that works" column gains "Dog bark — short-term, until habituation."
  - Add a new section noting expected habituation in days; the doc already warned about this, so we flag that the real defense remains the phone notification.
  - Update the "How this changes the system design" section to reflect `use_claude: True` for cats (see §3).
- Keep the `sox` generation snippets in the doc — they remain useful if someone wants to retry ultrasonic with a wired/known-good speaker later.
- Update `sounds/SOURCES.txt` with attribution for the new bark file (user to confirm source).

### 2. Crow YOLO threshold: 0.25 → 0.35

One-line config change in `crowbuster.py`:

```python
TARGETS["crow"]["min_confidence"] = 0.35
```

Adjust the trailing comment to reflect new value and rationale: *"Tuned up from 0.25 after small-bird false-fires. Claude refinement still backstops anything that gets past 0.35."*

Cat threshold stays at **0.25**. The cat path's new safety net is the Claude prompt (§3), not the YOLO score — keeping the threshold low ensures we don't miss a low-confidence-but-genuine cat in poor lighting.

### 3. VLM refinement — prompt overhaul + reason logging

#### New crow prompt

```
A motion-triggered camera flagged a possible crow on a residential porch.
Look at the image and answer: is there clearly a crow, raven, or other large
all-dark corvid?

Crows are large (pigeon-sized or bigger), uniformly black or very dark gray,
with a heavy straight bill.

Not crows: pigeons, doves, sparrows, finches, robins, jays, and any small or
light-colored bird.

Respond in this exact format:
YES — <up to 10 words why>
or
NO — <up to 10 words why>
```

#### New cat prompt (cat path gains `use_claude: True`)

```
A motion-triggered camera flagged a possible cat on a residential porch.
Look at the image and answer: is there clearly a cat?

Cats have four legs, fur, a feline body silhouette, and a tail.

Common false positives in this camera: raccoons, dark plush items, shadows on
textiles, dogs.

Respond in this exact format:
YES — <up to 10 words why>
or
NO — <up to 10 words why>
```

Removed from both: the previous *"If uncertain, answer YES"* clause. The `fail-toward-alarm` principle remains, but is now enforced at the **error path** (API failure, malformed response) rather than baked into the prompt itself.

#### Function signature change

```python
def is_target_via_claude(frame, prompt: str) -> tuple[bool, str]:
    """Returns (decision, reason). Fails toward (True, '<error msg>') on
    API or parsing failure — the alarm bias lives here, not in the prompt."""
```

- `max_tokens`: 10 → 50.
- Parse: take the first text block, uppercase + strip, decision is `startswith("YES")`. Reason is whatever follows the first em dash, hyphen, comma, or whitespace after `YES`/`NO`, trimmed. If we can't parse a clean reason, set it to `""` (decision stands).
- Error paths return `(True, "API error: <ExceptionName>")` or `(True, "no text block")`.

#### Logging & captures

- Detection log line:
  - Before: `→ Claude(cat): YES (456ms)`
  - After:  `→ Claude(cat): YES — clearly a cat, four legs and tail visible (456ms)`
- Capture filenames for **rejected** YOLO detections include a slug of the reason:
  - Before: `20260621_143012_bird_not_crow.jpg`
  - After:  `20260621_143012_bird_not_crow__small_light_brown_sparrow.jpg`
  - Slug = lowercase, alphanumeric + underscores, truncated at ~40 chars. Filesystem-safe.
- Confirmed-detection captures (`{label}.jpg`, `habituated.jpg`) keep their existing names — the reason is in the log line on the same timestamp; no need to duplicate it into the filename.

#### Config wiring

```python
TARGETS["cat"]["use_claude"] = True
TARGETS["cat"]["claude_prompt"] = "<new cat prompt above>"
TARGETS["crow"]["claude_prompt"] = "<new crow prompt above>"
```

Cat's `habituation_threshold: 1` and `persistent_refire_seconds: 30` stay — those reflect cat psychology (stalkers, fast-escalate), not detection logic. The Claude check stops phantom alarms before they reach the refire counter.

#### Test mode

Update the `TEST_MODE` "person" target prompt to the same `YES — <reason>` / `NO — <reason>` format. Otherwise the test pipeline doesn't exercise the new reason-parsing path and a parser regression would slip past manual smoke testing.

### 4. Docs

- `README.md` "How it works" / pipeline section: update to note cat path is now Claude-refined.
- `docs/cat-deterrence.md`: changes covered in §1 and §3.

## Implementation order

1. Edit `crowbuster.py`: threshold bump, prompt rewrites, `is_target_via_claude` signature change, cat path `use_claude=True`, log line + capture filename updates.
2. Drop user-supplied `dog-bark.mp3` into `sounds/cat/`, delete `ultrasonic-20khz.mp3`, update `sounds/SOURCES.txt`.
3. Update `docs/cat-deterrence.md` and `README.md`.
4. Manual smoke test: `CROWBUSTER_TEST=1 python crowbuster.py` and walk in front of camera — verify the test prompt parses with the new format and the reason hits the log.
5. Deploy to the production box via `sync.sh`, restart the service, confirm startup ping and first heartbeat.

## Risk & rollback

- **Risk: new prompts reject genuine crows/cats.** Mitigation: reason field in logs makes this immediately visible. Rollback = revert prompt strings (one commit). YOLO threshold revert is one line.
- **Risk: parsing the reason breaks on unexpected model output.** Mitigation: parser falls back to `(True, "")` on any failure, so detection still fires. Visible in logs as a missing reason.
- **Risk: dog bark habituation kicks in within days.** Acknowledged in `cat-deterrence.md`; the real defense is the phone notification, which is unaffected. Reassess after two weeks.

## Open questions (deferred, not blocking)

- Should we crop to YOLO bbox before sending to Claude? Defer until reason-logged data tells us where Claude is failing.
- Should the Claude reason be sent in the ntfy notification body? Currently no — keeps notifications terse. Reconsider if reasons reveal patterns worth alerting on.
