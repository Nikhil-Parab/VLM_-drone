# Master Development Plan: SVLM-Driven Perception & Control Pipeline
**Phase: Laptop-first prototyping. Core/thread constraints deliberately ignored per current direction — revisit at Pi5 migration (Phase 8).**

---

## 0. The Core Architectural Decision

You want one SVLM to act as both **visual characteristic classifier** (find/lock arbitrary targets, down to minute features) **and drone controller** (decide movement). That's the right ambition, but it needs to be split into two distinct responsibilities inside the same conceptual "SVLM does everything" system, because a single model call cannot reliably do both well at once:

| Responsibility | What it needs to be good at | Recommended approach |
|---|---|---|
| **Grounding** — finding an arbitrary described target, including tiny/fine-grained features | Precise spatial localization (bounding box/point) for open-vocabulary text | A dedicated grounding model (Stage 1 below) — small captioning-style VLMs are weak at bbox regression |
| **Semantic control** — deciding "what to do about it" (direction, urgency, lock/lost/re-acquire) | Structured judgment under uncertainty, reasoning about relative position | The SVLM, prompted for structured discrete/categorical output — NOT raw motor floats |

This isn't a downgrade of your vision — it's what makes it actually reliable. The SVLM is still the brain making every meaningful decision; it just delegates precise pixel-level localization to a component built for that, the same way your own brain doesn't calculate exact muscle-fiber tension when you decide "grab that."

---

## 1. Target-Grounding Layer (replaces YOLO-COCO for open-vocabulary targets)

**Why this has to change:** YOLOv8n only knows COCO's 80 classes. "The smallest of things, even a minute feature of someone or some object" is fundamentally impossible for it — there's no amount of tuning that gets a COCO detector to find "the mole above someone's left eyebrow" or "the cracked corner of that phone case." This requires open-vocabulary grounding.

**Since you're laptop-first and ignoring compute constraints for now:**
- Evaluate **Grounding DINO** (already flagged as a half-finished thread in your own project notes — "Grounding DINO benchmark: box_threshold → threshold fix") or **OWL-ViT** as the grounding backbone. Both take a free-text phrase and return candidate boxes, which is exactly the capability YOLO structurally lacks.
- Feed the grounding model the *literal target description* extracted from the voice command (e.g., "the red mug's handle," "the small logo on his shirt") rather than a COCO class name.
- Build an explicit **confidence/ambiguity handling step**: grounding models return multiple candidate boxes with scores. Don't auto-lock on the top box blindly — surface top-3 candidates to the SVLM (Stage 2) for a semantic sanity check ("does this crop actually match the description?") before committing to a lock. This catches false grounds before they propagate into a bad follow decision.
- **Set honest expectations now, not after testing:** even state-of-the-art grounding models struggle with genuinely tiny sub-features (a nose on a small/distant person, a single button on a jacket) — resolution limits are physical, not a prompt-engineering problem. Plan an evaluation phase (Phase 4 below) specifically to characterize *how* small/fine a feature this pipeline can reliably ground, rather than assuming unlimited fidelity.

---

## 2. Lock-On & Re-ID Layer (extends what you already have)

Your existing `reid_engine.py` (ORB + HSV histogram embedding, ~5ms) is a good foundation and needs minimal change in spirit — just a new entry point:

- `lock_target()` currently computes an embedding from a YOLO-detected crop. Change the *source* of that crop to the grounding layer's output instead of YOLO's, everything downstream stays the same.
- Keep the fast embedding-based `verify()` loop for frame-to-frame continuity (every 0.5s) exactly as-is — this is cheap and doesn't need to change regardless of how exotic the target description is, since it's comparing visual similarity, not re-running grounding every frame.
- Re-grounding (re-running the full grounding model, not just the fast embedding check) should only happen when `verify()` reports a likely ID switch or the tracker reports the target lost — grounding models are heavier than ORB matching, so don't run them per-frame.

---

## 3. SVLM Semantic Control Layer (the new core piece)

This is the actual "SVLM as drone controller" you're describing, designed to be reliable:

**Input to the SVLM each decision cycle:**
- Current frame (or locked-target crop + surrounding context)
- The target description / lock state
- A compact telemetry summary (current heading/speed/altitude — plain text, not raw floats, e.g. "moving forward, moderate speed, target left-of-center")

**Output — force a constrained, structured schema, not free text:**
```
{
  "target_visible": true/false,
  "horizontal": "left" | "center" | "right",
  "vertical": "up" | "center" | "down",
  "depth": "too_close" | "good_distance" | "far" | "very_far",
  "urgency": "hold" | "gentle" | "moderate" | "aggressive",
  "confidence": "low" | "medium" | "high"
}
```
This is the critical design choice: **discrete categorical bins, not continuous numbers.** A small VLM asked "what's the exact heading in degrees" will hallucinate precision it doesn't have. A small VLM asked "left, center, or right" is answering something it can actually perceive reliably. Confidence in the model's output is what lets you trust it in a control loop.

**Deterministic execution layer (thin, fast, not a VLM call):**
- Maps the SVLM's categorical output → actual velocity vector via a lookup table you tune (e.g., `urgency=aggressive` + `horizontal=left` → specific yaw/roll rate).
- Runs every frame using this table, *not* waiting on the SVLM every frame — the SVLM's decision persists and gets refreshed asynchronously (same async pattern your `reid_executor`/`cls_executor` already use), while the Kalman tracker smooths motion between SVLM updates exactly like it already smooths between YOLO detections today.
- This means: SVLM decides *intent* every ~1-2 seconds (or whenever meaningfully new), the execution layer converts intent → smooth continuous motion every frame. This is the same "slow brain, fast reflex" split your current architecture already uses for perception — you're extending the same pattern to control, not inventing a new one.

---

## 4. Compound Multi-Parameter Voice Commands

"Move right, then up, at this speed, to this height" in one utterance needs the parser upgraded, but — consistent with your own earlier, already-validated decision to keep command parsing rule-based rather than VLM-driven (SmolVLM hallucinated JSON headers when tried for this before) — this should stay a **deterministic slot-filling parser**, not a second VLM call:

- Extend the current regex/keyword parser into a **multi-slot grammar**: recognize multiple direction tokens, a speed modifier ("slowly," "quickly," or an explicit value), and an altitude/height target, all in a single parsed command object rather than a single `action` string.
- Command object shape becomes: `{"directions": ["right", "up"], "speed": "medium", "target_altitude_m": 5.0, "duration_or_distance": ...}` instead of today's single-action packet.
- This is a parser/grammar engineering task, not a model task — keep it fast and deterministic, same reasoning as before.

---

## 5. Altitude-Gated Activation State Machine

New states layered onto your existing mission state machine (`FOLLOW`/`HOVER`/`SEARCH`/`SCAN`/`RTL`):

```
ARMED → CLIMBING → [altitude threshold reached] → CAMERA_ACTIVE → AWAITING_TARGET → LOCKED → FOLLOWING
```
- `CLIMBING → CAMERA_ACTIVE` transition triggers on telemetry altitude crossing your configured threshold — this is a pure state-machine addition, no model involvement.
- Camera/recording/evaluation loop only starts consuming CPU/inference budget once this transition fires — meaningful for later Pi optimization even though you're ignoring that now, since it means the heavy grounding+SVLM pipeline isn't running during climb-out at all.

---

## 6. Safety & Execution Guardrails (non-negotiable regardless of prototype stage)

Since this pipeline ends in actual flight control, build these in from the start rather than retrofitting later — they're cheap now and expensive to bolt on after the control loop exists:

- **Bounds checking on every execution-layer output** — hard caps on max velocity/yaw-rate regardless of what the SVLM's "aggressive" urgency maps to, independent of and layered on top of whatever Pixhawk/MAVSDK-side limits exist.
- **Watchdog on SVLM decision staleness** — if the SVLM's last decision is older than N seconds (call timed out, executor stuck), the execution layer must fall back to `hold_position()`, never continue extrapolating a stale "aggressive forward" intent indefinitely.
- **Manual override always wins** — any keyboard/voice `hover`/`rtl` command must be able to interrupt an in-progress SVLM-driven follow instantly, no queued/pending state allowed to block it.
- **Simulation-first validation** — validate the full SVLM-decides→execution-layer→drone-controller loop extensively in `Sim2DDroneController` before it ever touches `MAVSDKDroneController`/real Pixhawk hardware. The categorical-bin design in Section 3 is testable and gradeable in sim (you can log every SVLM decision alongside ground-truth target position and score directional accuracy offline) — do that scoring before trusting it near real flight.

---

## 7. Evaluation Harness (build this before wiring into live flight control)

Because the whole system's reliability hinges on SVLM output quality, build an offline test harness before Phase 6's live integration:

- Record video clips of targets (including deliberately fine-grained/small ones) with manually-labeled ground-truth position/direction.
- Replay them through the grounding layer + SVLM decision layer, score: grounding hit-rate at varying target sizes, SVLM directional-bin accuracy against ground truth, lock persistence/ID-switch rate.
- This tells you *empirically* — not from vibes — how small a feature this pipeline can actually track, and where the SVLM's categorical decisions are and aren't trustworthy, before you fly anything based on it.

---

## 8. Codebase Cleanup (Safe, Non-Functional Changes Only)

Before layering the new grounding/SVLM-control layers on top of the existing pipeline, it's worth clearing out accumulated cruft — but this must be done with a strict non-negotiable rule: **remove only what is provably unreferenced. If there's any doubt whether something is dead, leave it and flag it instead of deleting it.** A cleanup pass that breaks a working pipeline right before a major architecture change is the worst possible time for that to happen.

Categories worth auditing, roughly in order of how confidently they can be identified as safe to remove:

1. **Duplicate/superseded project folders.** Your project has migrated at least once already (`drone_engine/` → `enginev2/`, per the latest progress notes). Confirm which folder is actually being run (`enginev2/` per the run commands) and whether `drone_engine/` is now a stale fork nobody imports from. If confirmed unused, archive it outside the active repo rather than deleting outright — cheap insurance, and it costs nothing to keep as a zip alongside `code.zip`.
2. **Duplicate context/log files.** Multiple copies of `context.md` and prior task-snapshot markdowns exist across folders (`anticontext/context.md`, root-level `context.md`, `vLLM-production-task_snapshot.md`). These are documentation, not code — safe to consolidate into one canonical location, but don't delete history; merge into a single running changelog instead of discarding.
3. **Dead code paths from superseded optimization passes.** Specifically check for lingering references to patterns explicitly replaced earlier in this project — e.g., any remaining Jaccard-word-overlap ReID comparison code now that embedding-based ReID replaced it, or any remaining per-call `torch.set_num_threads()` toggling inside `_low_priority_inference()` now that `thread_budget.py` is the single source of truth (flagged as a likely hang cause earlier — if not yet removed, this is both a cleanup item and a bug fix). Grep for the specific function/variable names of anything a prior phase said it "replaced" or "removed," since old scaffolding sometimes survives as unreachable code even after its caller changes.
4. **Unused imports and orphaned helper functions.** Standard static-analysis cleanup (unused imports, functions with zero call sites across the repo) — safe once confirmed via actual call-site search, not just visual inspection, since Python's dynamic nature means a function can be referenced via string/dict dispatch (e.g., the command-action dispatch table) rather than a direct call.
5. **`__pycache__`, stale `.venv` artifacts, and any old benchmark output files** (e.g. leftover JSON dumps from the SmolVLM latency benchmark) that aren't referenced by any script — pure housekeeping, essentially zero risk.

**Process, not just a list:** have Cursor/Antigravity produce a *dry-run report first* — every file/function it proposes removing, with the evidence (grep results showing zero references) — for your review, before it actually deletes anything. Do not let it delete-then-report. This matches the same "measure before changing" discipline used throughout this project's optimization passes.

---

## 9. Phasing & Sequencing

| Phase | Deliverable | Depends on |
|---|---|---|
| 0 | Codebase cleanup — dry-run report, then removal of confirmed-dead files/code only (Section 8) | — (can run anytime, ideally before Phase 1) |
| 1 | Stand up Grounding DINO/OWL-ViT for open-vocabulary target grounding, offline, no drone integration yet | — |
| 2 | Wire grounding output into existing `reid_engine.py` lock/verify loop (swap crop source only) | Phase 1 |
| 3 | Design + prompt-engineer the SVLM structured-decision schema (Section 3), test offline against recorded clips | Phase 1-2 |
| 4 | Build the evaluation harness (Section 7), characterize grounding accuracy vs. target size and SVLM decision accuracy | Phase 3 |
| 5 | Build the deterministic execution layer (categorical decision → velocity vector), integrate with `Sim2DDroneController` only | Phase 3-4 |
| 6 | Extend voice parser to multi-slot compound commands (Section 4) | Independent — can run parallel to 1-5 |
| 7 | Add altitude-gated activation state machine (Section 5) + full safety guardrails (Section 6) | Phase 5-6 |
| 8 | Pi5 migration: quantize/distill grounding model + SVLM, revisit thread/core budget, re-run evaluation harness on target hardware before trusting it there | Everything above, validated in sim first |

Phases 1-4 are pure perception/decision-quality work and can proceed entirely offline on recorded footage — no live drone needed. Don't skip ahead to Phase 5+ until Phase 4's evaluation numbers give you real confidence in grounding/decision accuracy; wiring an unvalidated decision loop straight into flight control is where prototypes turn into crashes.
