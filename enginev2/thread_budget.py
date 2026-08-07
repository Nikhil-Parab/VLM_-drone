"""
thread_budget.py — single source of truth for CPU core / thread limits.

When running on CPU (--device cpu or Pi deployment), the process must not
spread across all host cores.  This module caps:
  - BLAS/OpenMP env vars (must be set before numpy/torch import)
  - torch intra-op and inter-op thread pools (set ONCE at startup)
  - OpenCV's internal thread pool
  - OS process CPU affinity (Linux sched_setaffinity + psutil on Windows/macOS)

Import this module before cv2/numpy/torch in entry-point scripts.
"""
from __future__ import annotations

import os
import sys
import threading

# Confirms this module body runs before drone_tracking_engine imports cv2/numpy/torch.
print("[thread_budget] module load — BLAS env caps before numeric imports", flush=True)

DEFAULT_MAX_CORES = 2

_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Guard: torch.set_num_threads() must fire exactly once per process (startup only).
_TORCH_INTRA_THREADS_SET_COUNT = 0
_BUDGET_APPLIED_MAX_CORES: int | None = None


def early_max_cores() -> int:
    """Parse --max-cores from sys.argv without importing argparse-heavy deps."""
    for i, arg in enumerate(sys.argv):
        if arg == "--max-cores" and i + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[i + 1]))
            except ValueError:
                pass
        if arg.startswith("--max-cores="):
            try:
                return max(1, int(arg.split("=", 1)[1]))
            except ValueError:
                pass
    return DEFAULT_MAX_CORES


def apply_env_thread_caps(max_cores: int) -> None:
    """Force BLAS/OpenMP libraries to honour max_cores (call before numpy/torch)."""
    n = str(max(1, int(max_cores)))
    for var in _ENV_VARS:
        os.environ[var] = n


def _set_torch_intra_threads_once(max_cores: int) -> None:
    """Set torch intra-op threads to max_cores."""
    global _TORCH_INTRA_THREADS_SET_COUNT
    import torch

    _TORCH_INTRA_THREADS_SET_COUNT += 1
    torch.set_num_threads(max_cores)


def get_torch_intra_threads_set_count() -> int:
    """Test/validation hook — should remain 1 for a full engine run."""
    return _TORCH_INTRA_THREADS_SET_COUNT


def _read_budget_state(expected_cores: int) -> dict:
    """Read-only snapshot of current thread/affinity settings."""
    import cv2
    import torch

    state = {
        "expected": expected_cores,
        "torch_intra": torch.get_num_threads(),
        "cv2_threads": None,
        "sched_affinity": None,
        "psutil_affinity": None,
        "torch_ok": False,
        "cv2_ok": False,
        "affinity_ok": False,
        "ok": False,
    }

    try:
        interop = torch.get_num_interop_threads()
    except Exception:
        interop = "?"
    state["torch_interop"] = interop

    if hasattr(os, "sched_getaffinity"):
        state["sched_affinity"] = set(os.sched_getaffinity(0))

    try:
        import psutil

        aff = psutil.Process().cpu_affinity()
        state["psutil_affinity"] = set(aff) if aff is not None else None
    except Exception:
        state["psutil_affinity"] = None

    try:
        state["cv2_threads"] = cv2.getNumThreads()
    except Exception:
        state["cv2_threads"] = "unknown"

    expected_set = set(range(expected_cores))
    state["torch_ok"] = state["torch_intra"] == expected_cores
    state["cv2_ok"] = (
        state["cv2_threads"] == "unknown"
        or state["cv2_threads"] == expected_cores
    )

    if state["sched_affinity"] is not None:
        state["affinity_ok"] = state["sched_affinity"] == expected_set
    elif state["psutil_affinity"] is not None:
        state["affinity_ok"] = state["psutil_affinity"] == expected_set
    else:
        state["affinity_ok"] = True  # platform without affinity API

    state["ok"] = state["torch_ok"] and state["cv2_ok"] and state["affinity_ok"]
    return state


def _pin_process_affinity(max_cores: int) -> tuple[str, set[int] | str]:
    """Pin the current process to the first N logical CPUs. Returns (method, cores)."""
    n = max(1, int(max_cores))
    target = set(range(n))

    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, target)
            if hasattr(os, "sched_getaffinity"):
                return "sched_setaffinity", set(os.sched_getaffinity(0))
            return "sched_setaffinity", target
        except OSError as exc:
            print(f"[BUDGET] sched_setaffinity failed: {exc}")

    try:
        import psutil

        proc = psutil.Process()
        available = proc.cpu_affinity()
        if not available:
            total = psutil.cpu_count(logical=True) or n
            available = list(range(total))
        pinned = available[: min(n, len(available))]
        proc.cpu_affinity(pinned)
        return "psutil", set(proc.cpu_affinity())
    except Exception as exc:
        print(f"[BUDGET] psutil cpu_affinity failed: {exc} "
              f"(thread pools still capped; OS may schedule on more cores)")

    return "none", "n/a"


def boost_main_thread_priority() -> None:
    """Raise main/UI thread scheduling priority (cheap; safe to call once at startup)."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        THREAD_PRIORITY_ABOVE_NORMAL = 1
        ok = kernel32.SetThreadPriority(
            kernel32.GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL
        )
        print(f"[BUDGET] main thread priority -> ABOVE_NORMAL ({'ok' if ok else 'failed'})")
    else:
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            PRIO_PROCESS = 0
            tid = threading.get_native_id()
            libc.setpriority(PRIO_PROCESS, tid, -5)
            print(f"[BUDGET] main thread nice -> -5 (tid={tid})")
        except Exception as exc:
            print(f"[BUDGET] main thread priority boost skipped: {exc}")


def set_background_thread_priority() -> None:
    """
    Lower priority on VLM worker threads so they yield to the main loop under
    2-core affinity.  Intended as ThreadPoolExecutor initializer — cheap vs
    torch.set_num_threads() pool rebuilds.
    """
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        THREAD_PRIORITY_BELOW_NORMAL = -1
        kernel32.SetThreadPriority(
            kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL
        )
    else:
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            PRIO_PROCESS = 0
            tid = threading.get_native_id()
            libc.setpriority(PRIO_PROCESS, tid, 10)
        except Exception:
            try:
                os.nice(5)
            except OSError:
                pass


def set_thread_budget(max_cores: int) -> None:
    """
    Apply the full CPU budget once at startup: env vars, torch, OpenCV, affinity.
    """
    global _BUDGET_APPLIED_MAX_CORES
    max_cores = max(1, int(max_cores))
    _BUDGET_APPLIED_MAX_CORES = max_cores
    apply_env_thread_caps(max_cores)

    import cv2
    import torch

    _set_torch_intra_threads_once(max_cores)
    try:
        torch.set_num_interop_threads(min(2, max_cores))
    except RuntimeError:
        pass

    cv2.setNumThreads(max_cores)

    method, affinity = _pin_process_affinity(max_cores)
    print(f"[BUDGET] Process pinned via {method} -> {affinity}")

    verify_thread_budget(max_cores, context="startup")


def verify_thread_budget(expected_cores: int, context: str = "") -> None:
    """Read-only budget check — never calls SET/re-pin paths."""
    state = _read_budget_state(expected_cores)
    tag = "OK" if state["ok"] else "MISMATCH"
    print(
        f"[BUDGET:{context}] torch_intra={state['torch_intra']} "
        f"torch_interop={state.get('torch_interop', '?')} "
        f"sched_affinity={state['sched_affinity']} "
        f"psutil_affinity={state['psutil_affinity']} "
        f"cv2_threads={state['cv2_threads']} expected={expected_cores} [{tag}]"
    )
    if state["cv2_threads"] not in ("unknown", expected_cores) and state["cv2_threads"] > expected_cores:
        print(
            f"[BUDGET:{context}] WARNING: cv2 thread pool ({state['cv2_threads']}) "
            f"exceeds budget ({expected_cores})"
        )


def reapply_cv2_and_affinity(max_cores: int, context: str = "post-model") -> None:
    """
    Re-apply cv2 threads + OS affinity after a library (e.g. ultralytics YOLO)
    resets thread pools — WITHOUT calling torch.set_num_threads().
    """
    import cv2

    max_cores = max(1, int(max_cores))
    cv2.setNumThreads(max_cores)
    state = _read_budget_state(max_cores)
    if not state["affinity_ok"]:
        method, affinity = _pin_process_affinity(max_cores)
        print(f"[BUDGET:{context}] re-pinned affinity via {method} -> {affinity}")
    if not state["cv2_ok"]:
        print(f"[BUDGET:{context}] cv2 threads re-applied -> {max_cores}")


def repair_thread_budget_if_drifted(expected_cores: int, context: str = "runtime") -> bool:
    """
    Read-only check first; re-apply SET paths only when drift is detected.
    Returns True if a repair was performed.

    IMPORTANT — torch.set_num_threads() is intentionally NOT called here.
    On MKL/Windows (and some BLAS backends on Linux/Pi) torch.set_num_threads()
    triggers a real teardown and rebuild of the native thread pool, which can
    stall for tens to hundreds of milliseconds — the primary cause of the
    process "freeze" symptom. The OS-level affinity pin and BLAS env vars are
    the only levers we can safely pull mid-run without paying that rebuild cost.
    If torch_intra drifts (should never happen after the per-call VLMThreadLimiter
    was removed), log a warning but do NOT re-apply it; a process restart is the
    correct remedy if the pool somehow gets corrupted.
    """
    state = _read_budget_state(expected_cores)
    if state["ok"]:
        return False

    # Log drift details so the operator knows something changed
    print(
        f"[BUDGET:{context}] DRIFT detected "
        f"(torch_ok={state['torch_ok']} cv2_ok={state['cv2_ok']} "
        f"affinity_ok={state['affinity_ok']}) — repairing where safe"
    )

    import cv2

    max_cores = max(1, int(expected_cores))
    repaired = False

    # BLAS env vars: always safe to re-set (affects future BLAS library init only,
    # not the running pool — but keeps the setting correct for any subprocess forks).
    apply_env_thread_caps(max_cores)

    if not state["torch_ok"]:
        print(
            f"[BUDGET:{context}] WARNING: torch_intra drifted to {state['torch_intra']} "
            f"(expected {max_cores}) — NOT repairing mid-run (MKL pool rebuild stalls). "
            f"Restart the engine if this persists."
        )
        repaired = True

    if not state["cv2_ok"]:
        cv2.setNumThreads(max_cores)
        print(f"[BUDGET:{context}] repaired cv2_threads -> {max_cores}")
        repaired = True

    if not state["affinity_ok"]:
        method, affinity = _pin_process_affinity(max_cores)
        print(f"[BUDGET:{context}] repaired affinity via {method} -> {affinity}")
        repaired = True

    verify_thread_budget(expected_cores, context=f"{context}-after-repair")
    return repaired


# Apply BLAS env caps as soon as this module is imported (before numpy/torch load).
apply_env_thread_caps(early_max_cores())
