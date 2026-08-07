# CHANGELOG

Full development history, moved here from the in-code docstrings so the
source files stay lean for Pi transfer. Code comments now point here
instead of repeating this narrative.

## SVLM Controller Master Plan — Phases 0–7 (laptop-first)

Implemented per `svlm_controller_master_plan.md`. Phase 8 (Pi5 migration) deferred.

### Phase 0 — Cleanup dry-run
- `dev_tools/cleanup_dry_run.md`: audit of stale folders, dead code, safe removals.
- No files auto-deleted; `drone_engine/` already absent from tree.

### Phase 1 — Open-vocabulary grounding (`grounding_engine.py`)
- **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`) primary backend.
- **OWL-ViT** optional fallback (`backend="owlvit"`).
- `threshold=` / `box_threshold=` compat for transformers ≥4.55.
- Top-K candidates + SmolVLM yes/no **sanity check** before lock (`pick_best_grounded_candidate`).

### Phase 2 — Grounding → ReID lock path
- `drone_tracking_engine.py --detector grounding` feeds grounding boxes into existing
  `reid.lock_target()` / ORB `verify()` loop (crop source swapped, downstream unchanged).
- Re-ground on embedding ID-switch when grounding backend active.

### Phase 3 — SVLM semantic control (`svlm_controller.py`)
- Pipe-delimited `<CTRL>` tag schema (visible, h, v, d, u, c categorical bins).
- `SVLMController.decide()` async via dedicated `ctrl_executor`.
- `build_telemetry_text()` for plain-language drone state in prompt.

### Phase 4 — Evaluation harness (`eval_harness.py`)
- Offline replay: grounding hit-rate (IoU vs JSON labels), optional SVLM directional accuracy.
- CLI: `python eval_harness.py --video clip.mp4 --phrase "red mug" [--labels gt.json] [--svlm]`

### Phase 5 — Execution layer (`execution_layer.py`)
- Lookup table: categorical decision → `move_toward` velocity (sim pixel space).
- **Guardrails**: max speed cap, low-confidence downgrades aggressive moves,
  **staleness watchdog** (hold if decision older than 4s), **manual override** for hover/RTL.

### Phase 6 — Compound voice commands (`voice_command_interface.py`)
- Multi-slot parser: `directions[]`, `speed`, `target_altitude_m`, `grounding_phrase`.
- Example: `"move right then up slowly to 5 meters"` → structured UDP packet.
- Still rule-based (not VLM JSON) — same rationale as original design.

### Phase 7 — Altitude activation + sim telemetry
- State machine: `ARMED → CLIMBING → CAMERA_ACTIVE → AWAITING_TARGET` (skip with `--skip-activation`).
- `Sim2DDroneController`: `altitude_m`, `set_altitude_target()`, climb rate in `step_physics`.
- Perception pipeline gated until activation altitude reached (default 3m sim).

### Engine CLI additions
```bash
python drone_tracking_engine.py --detector grounding --svlm-control --skip-activation
python drone_tracking_engine.py --detector yolo                    # legacy COCO path (default)
python drone_tracking_engine.py --activation-altitude 5.0          # custom climb gate
```

### thread_budget fix (during integration)
- `reapply_cv2_and_affinity()` after YOLO load — no second `torch.set_num_threads()`.
- `repair_thread_budget_if_drifted()` no longer calls `torch.set_num_threads()` mid-run.

---

## Windows OpenCV MSMF Backend Fix (`MF_E_INVALIDREQUEST`)