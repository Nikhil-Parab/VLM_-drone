# Phase 0 — Codebase cleanup dry-run report

**Generated for:** `svlm_controller_master_plan.md` Section 8  
**Rule:** Remove only what is provably unreferenced. Flag uncertain items; do not delete.

## 1. Duplicate / superseded folders

| Path | Status | Evidence | Recommendation |
|---|---|---|---|
| `drone_engine/` (git: deleted) | **Stale** | Git status shows all files deleted; run commands point to `enginev2/` | Already removed from tree; keep `anticontext/code.zip` as archive |
| `enginev2/` | **Active** | README, CHANGELOG, all run commands | Canonical runtime folder |

## 2. Duplicate context / log files

| Path | Status | Recommendation |
|---|---|---|
| `anticontext/context.md` | Canonical project log | Keep |
| `svlm_controller_master_plan.md` (repo root) | Active spec | Keep |
| `vLLM-production-task.snapshot.md` (Downloads) | External handoff | Not in repo — no action |
| Root `context.md` | Not found in repo | N/A |

## 3. Dead code paths (grep verification)

| Item | Status | Evidence |
|---|---|---|
| `VLMThreadLimiter` | **Removed** | Grep: zero matches in enginev2 |
| Jaccard ReID compare | **Removed** | `reid_engine.py` uses ORB+HSV only |
| Per-call `torch.set_num_threads()` in VLM paths | **Removed** | Grep: only `thread_budget.py` |
| `small_object_classifier` missing `VLMThreadLimiter` import | **Fixed** | Class removed entirely |

## 4. Unused imports / orphans

| File | Finding | Action |
|---|---|---|
| `drone_tracking_engine.py` | `verify_thread_budget` imported but unused (repair used instead) | Safe to remove import in future pass |
| `grounding_engine.py` | `re` imported, `time` imported — time unused | Minor; leave for now |

## 5. Housekeeping (zero-risk)

| Item | Action |
|---|---|
| `__pycache__/` | Add to `.gitignore` if not already |
| Old benchmark JSON | Not in enginev2 — none to delete |

## Proposed removals (awaiting approval)

**None auto-deleted.** All items above are either already cleaned or flagged only.

## New modules added (master plan Phases 1–7)

- `grounding_engine.py` — Phase 1–2
- `svlm_controller.py` — Phase 3
- `execution_layer.py` — Phase 5
- `eval_harness.py` — Phase 4
- Extended `voice_command_interface.py` — Phase 6
- Extended `drone_tracking_engine.py` — Phases 2, 3, 5, 7
