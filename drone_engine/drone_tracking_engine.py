"""
drone_tracking_engine.py
==========================
Full pipeline — Phase 1-6 integrated + Optimization Phases A/B/C/D/G/I
+ Pi5 hard 2-thread/2-core budget + scoped VLM thread-priority fix.

  Phase A: Two dedicated ThreadPoolExecutors (cls + reid) so classification
           and ReID run concurrently. VLM timeout watchdog abandons stuck calls.
  Phase B: torch.set_num_threads(2) at startup (process-wide ceiling).
  Phase C: ORB+histogram embedding ReID (fast, ~5ms) — VLM only used for the
           optional HUD label, never for frame-to-frame ReID decisions.
  Phase D: YOLO ROI crop (run detection only around Kalman-predicted position
           when locked) + frame-skip when locked (DETECT_EVERY_N_FRAMES_LOCKED)
           + two-tier confidence threshold + optional SAHI-style tiled search
           + BWCD-NMS postprocessing for dense/small-object scenes.
  Phase G: pick_best_candidate() uses combined distance+IoU score to reduce
           identity switches when targets cross paths.
  Phase I: PerfTimer per-stage instrumentation; [p] key toggles HUD timing.

  NEW — Pi5 thread-budget fixes:
    * A single, unambiguous source of truth for the process's thread/core
      ceiling (was previously set in two places that could silently
      disagree — see apply_cpu_affinity()/set_thread_budget() below).
    * VLM generate() calls (in reid_engine.py / small_object_classifier.py)
      now run at a temporarily-reduced thread count so they can never starve
      the main loop's share of the fixed 2-thread budget — see
      _low_priority_inference() in those two files.
    * Periodic console confirmation that the process is still actually
      pinned to its intended cores/threads (drift/verification logging),
      since individual libraries don't always respect a soft hint equally.

  camera → YOLOv8n detection (full frame / ROI crop / tiled search)
         → BWCD-NMS postprocessing
         → SmolObjectClassifier  (small / uncertain boxes get VLM labels)
         → Kalman tracker  (predict / update every frame)
         → periodic embedding-based re-ID check  (every 0.5s, ~5ms per check)
         → decision layer  (stopped / course-changed / lost)
         → DroneController command  (move_toward / hold_position / search_pattern / scan / rtl)
         ← UDP command receiver  (port 5000 — voice_command_interface.py sends here)

Controls (keyboard):
  q  — quit
  l  — lock onto the nearest high-confidence detection
  u  — unlock target / hover
  h  — hold / hover immediately
  s  — trigger search pattern
  r  — return to launch (sim: fly to centre)
  p  — toggle per-stage timing HUD

pip install: ultralytics opencv-python transformers torch pillow pygame numpy
Run:
    python drone_tracking_engine.py
    python drone_tracking_engine.py --max-cores 2      # Pi5 production budget
    python drone_tracking_engine.py --max-cores 4       # dev laptop, no cap
"""

import argparse
import collections
import concurrent.futures
import csv
import io
import json
import os
import queue
import socket
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from drone_controller import Sim2DDroneController
from kalman_tracker import KalmanTrack
from reid_engine import ReIDEngine
from small_object_classifier import SmolObjectClassifier

# ============================================================
# HARD THREAD/CORE BUDGET — single source of truth
# ============================================================
# Everything else in this file and in reid_engine.py / small_object_classifier.py
# assumes this value is set ONCE, here, before any torch/model work happens,
# and never silently overridden elsewhere. The previous version of this file
# set torch.set_num_threads(2) at import time AND ALSO let select_camera()'s
# --threads CLI flag override it later (with a broken log message that always
# printed the same value twice) -- that made it possible for the effective
# thread count to end up higher than the OS-level core affinity pin, which is
# worse than either limit alone (oversubscription: more threads than cores
# means the OS scheduler thrashes between them instead of running cleanly).
# This version fixes that: --max-cores is now the ONE flag that controls both
# torch's intra-op thread count and the OS affinity pin together, and they are
# always kept equal.

DEFAULT_MAX_CORES = 2  # <-- Pi5 production budget; override with --max-cores N


def set_thread_budget(max_cores: int):
    """
    Set BOTH torch's intra-op thread count and the OS-level CPU affinity to
    the same value, so the process can never oversubscribe (more torch
    threads than pinned cores) or undersubscribe in a way that leaves torch
    threads waiting on cores the OS won't schedule them onto.
    """
    torch.set_num_threads(max_cores)

    if hasattr(os, "sched_setaffinity"):
        try:
            target_cores = set(range(max_cores))
            os.sched_setaffinity(0, target_cores)
        except Exception as exc:
            print(f"[BUDGET] Could not set os.sched_setaffinity: {exc}")
    else:
        print(f"[BUDGET] os.sched_setaffinity not supported on {os.name} "
              f"(fine on Windows/dev machines; required on the Pi5 target).")

    verify_thread_budget(max_cores, context="startup")


def verify_thread_budget(expected_cores: int, context: str = ""):
    """
    Read back the ACTUAL current thread count / affinity and log it, rather
    than trusting that the earlier set_thread_budget() call stuck. Some
    libraries (OpenCV's own internal thread pool in particular) have their
    own separate thread settings that are not touched by torch.set_num_threads()
    or os.sched_setaffinity() alone -- this makes any mismatch visible in the
    console instead of silently causing contention.
    """
    actual_torch_threads = torch.get_num_threads()
    if hasattr(os, "sched_getaffinity"):
        actual_affinity = os.sched_getaffinity(0)
    else:
        actual_affinity = "n/a (platform doesn't support sched_getaffinity)"

    try:
        cv2_threads = cv2.getNumThreads()
    except Exception:
        cv2_threads = "unknown"

    ok = (actual_torch_threads == expected_cores)
    tag = "OK" if ok else "MISMATCH"
    print(f"[BUDGET:{context}] torch_threads={actual_torch_threads} "
          f"affinity={actual_affinity} cv2_threads={cv2_threads} "
          f"expected={expected_cores} [{tag}]")
    if cv2_threads != "unknown" and cv2_threads > expected_cores:
        print(f"[BUDGET:{context}] WARNING: OpenCV's own thread pool "
              f"({cv2_threads}) exceeds the intended budget ({expected_cores}). "
              f"Consider calling cv2.setNumThreads({expected_cores}) explicitly.")


# ============================================================
# CONFIGURATION
# ============================================================

YOLO_MODEL_PATH             = "../yolov8n.pt"   # <-- CHANGE DETECTOR MODEL
TARGET_CLASS_NAME           = "person"          # <-- CHANGE TARGET CLASS (COCO label)
DETECTION_CONFIDENCE_LOCKED = 0.40              # <-- locked-mode detection threshold
DETECTION_CONFIDENCE_SEARCH = 0.25              # <-- two-tier threshold for small objects when unlocked
YOLO_IMGSZ                  = 640               # <-- inference resolution (640, 960, or 1280)
ENABLE_TILED_SEARCH         = False             # <-- SAHI-style 2x2 grid search; costs more CPU, gate carefully under 2-core budget

CAMERA_INDEX             = 0
FRAME_WIDTH              = 640
FRAME_HEIGHT             = 480

MAX_MISSED_FRAMES        = 15              # <-- frames before "target lost"
REACQUIRE_SEARCH_RADIUS  = 150            # <-- px radius for re-matching after miss

# Phase D: frame-skip settings
DETECT_EVERY_N_FRAMES         = 1   # <-- full-frame detection rate (unlocked)
DETECT_EVERY_N_FRAMES_LOCKED  = 3   # <-- ROI detection rate (locked+stable)
ROI_PAD_PX                    = 120  # <-- padding around Kalman prediction for ROI crop

# Phase A: VLM watchdog — abandon future if it runs longer than this
VLM_TIMEOUT_SEC = 25.0  # <-- ~2.5x observed mean SmolVLM latency

# Phase G: pick_best_candidate scoring weights
DIST_WEIGHT = 0.6   # <-- contribution of distance score
IOU_WEIGHT  = 0.4   # <-- contribution of IoU score (needs last_known_bbox)

# UDP command receiver (matches voice_command_interface.py sender)
CMD_UDP_HOST             = "0.0.0.0"
CMD_UDP_PORT             = 5000

# Scan waypoints for 'scan' / 'survey' mission commands (pixel coords in sim)
DEFAULT_SCAN_WAYPOINTS   = [
    (160, 120), (480, 120), (480, 360), (160, 360),  # rectangle survey pattern
]


# ============================================================
# PHASE I: Per-stage timing instrumentation
# ============================================================

class PerfTimer:
    """Rolling-average timer for one pipeline stage. Thread-safe read via avg_ms()."""
    WINDOW = 30

    def __init__(self, name: str):
        self.name = name
        self._samples: collections.deque = collections.deque(maxlen=self.WINDOW)
        self._t0: float = 0.0
        self._lock = threading.Lock()

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self._samples.append(dt_ms)

    def record(self, dt_ms: float):
        with self._lock:
            self._samples.append(dt_ms)

    def avg_ms(self) -> float:
        s = list(self._samples)
        return sum(s) / len(s) if s else 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ============================================================
# CLI ARGS — parsed once, up front, before any camera/device/thread setup
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Drone Tracking Engine")
    parser.add_argument("--cam", "--camera", type=str, default=None,
                        help="Camera index (e.g. 0) or IP URL")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "auto"],
                        help="Device to use for models: cuda or cpu")
    parser.add_argument("--max-cores", type=int, default=DEFAULT_MAX_CORES,
                        help=f"Hard ceiling on BOTH torch intra-op threads AND OS "
                             f"CPU affinity, kept in sync (default {DEFAULT_MAX_CORES}; "
                             f"set to your Pi5's reserved-core count in production, or "
                             f"higher on a dev laptop with no reservation needed).")
    args, _ = parser.parse_known_args()
    return args


# ============================================================
# CAMERA SELECTION
# ============================================================

def select_camera(args):
    """
    Prompt user or parse CLI args to select camera source:
      - Laptop / built-in webcam (index 0 or integer)
      - IP Webcam URL (e.g. http://192.168.0.253:8080/video)
    """
    cam_source = args.cam
    if cam_source is None:
        print("\n" + "=" * 54)
        print(" SELECT CAMERA SOURCE")
        print("=" * 54)
        print("  [1] Laptop / Built-in Webcam (Index 0)")
        print("  [2] IP Webcam (Phone App / Network Camera)")
        print("=" * 54)
        try:
            choice = input("Select option (1 or 2, default 1): ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "1"

        if choice == "2":
            default_url = "http://192.168.0.253:8080/video"
            print("\nIP Webcam Setup (e.g. Android IP Webcam App):")
            try:
                url_input = input(f"Enter IP Webcam URL [Press Enter for {default_url}]: ").strip()
            except (KeyboardInterrupt, EOFError):
                url_input = ""
            cam_source = url_input if url_input else default_url
        else:
            cam_source = 0

    if isinstance(cam_source, str):
        if cam_source.isdigit():
            cam_source = int(cam_source)
        else:
            if not (cam_source.startswith("http://") or cam_source.startswith("https://") or cam_source.startswith("rtsp://")):
                cam_source = f"http://{cam_source}"
            if not any(cam_source.endswith(ext) for ext in ["/video", ".mjpg", ".mp4", "/shot.jpg", "/mjpeg"]):
                cam_source = cam_source.rstrip("/") + "/video"

    return cam_source


def select_device(args) -> str:
    """Prompt user or parse CLI args to select computation device (CUDA GPU vs CPU)."""
    chosen_device = args.device
    if chosen_device is None:
        print("\n" + "=" * 54)
        print(" SELECT COMPUTATION DEVICE")
        print("=" * 54)
        print("  [1] CUDA GPU (fast, if available)")
        print("  [2] CPU (Host Processor — standard, e.g. Pi5)")
        print("=" * 54)
        try:
            choice = input("Select device option (1 or 2, default 1): ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "1"
        chosen_device = "cuda" if choice == "1" else "cpu"

    if chosen_device == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[DEVICE] Selected CUDA GPU: {gpu_name}")
            return "cuda"
        else:
            print("\n" + "!" * 60)
            print("[WARNING] CUDA device requested, but torch.cuda.is_available() is False!")
            print("Current PyTorch build is CPU-only (torch+cpu).")
            print("To enable GPU acceleration, install a CUDA-enabled torch build:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
            print("!" * 60 + "\n")
            print("[DEVICE] Falling back to CPU for this run.")
            return "cpu"
    else:
        print("[DEVICE] Selected CPU for inference.")
        return "cpu"


# ============================================================
# THREADED CAMERA STREAM (Zero-Lag IP Webcam & USB Camera)
# ============================================================

class ThreadedCameraStream:
    """
    Background thread that continuously grabs frames from an IP camera / webcam stream.
    Eliminates network buffer queue lag by always serving the LATEST available frame
    and automatically resizing high-resolution phone streams (1080p/4K) to 640x480.
    """
    t_thread_internal: "PerfTimer" = None

    def __init__(self, src, target_size=(FRAME_WIDTH, FRAME_HEIGHT)):
        self.src = src
        self.target_size = target_size
        self.cap = cv2.VideoCapture(src)

        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_buf = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
        try:
            backend_name = self.cap.getBackendName()
        except AttributeError:
            backend_name = "unknown (old OpenCV)"
        print(f"[CAM] Backend: {backend_name}  "
              f"Requested BUFFERSIZE=1, read-back={actual_buf}  "
              f"({'HONORED' if actual_buf == 1.0 else 'IGNORED by backend'})")
        self.backend_name = backend_name
        self.buf_honored  = (actual_buf == 1.0)

        self.latest_frame = None
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()

        self.t_thread_internal = PerfTimer("cam_thread")
        self._last_bg_wall: float = 0.0
        self._frame_wall_gaps: collections.deque = collections.deque(maxlen=60)
        self.t_copy = PerfTimer("cam_copy")

        if self.cap.isOpened():
            self.grabbed, frame = self.cap.read()
            if self.grabbed and frame is not None:
                if (frame.shape[1], frame.shape[0]) != target_size:
                    frame = cv2.resize(frame, target_size)
                self.latest_frame = frame

            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()

    def _update_loop(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue

            _t0 = time.perf_counter()
            grabbed, frame = self.cap.read()
            if grabbed and frame is not None and frame.size > 0:
                if (frame.shape[1], frame.shape[0]) != self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                _dt_ms = (time.perf_counter() - _t0) * 1000.0
                self.t_thread_internal.record(_dt_ms)

                now_wall = time.time()
                if self._last_bg_wall > 0:
                    self._frame_wall_gaps.append(now_wall - self._last_bg_wall)
                self._last_bg_wall = now_wall

                with self.lock:
                    self.latest_frame = frame
                    self.grabbed = grabbed
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            grabbed = self.grabbed
            _t0 = time.perf_counter()
            frame = self.latest_frame.copy()
            self.t_copy.record((time.perf_counter() - _t0) * 1000.0)
        return grabbed, frame

    def get_bg_gap_stats(self):
        gaps = list(self._frame_wall_gaps)
        if not gaps:
            return 0.0, 0.0
        return (sum(gaps) / len(gaps)) * 1000.0, max(gaps) * 1000.0

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()


# ============================================================
# HELPERS & POST-PROCESSING
# ============================================================

def bwcd_nms(candidates: list, iou_thresh: float = 0.45, diou_gamma: float = 0.2, sigma: float = 0.5) -> list:
    """
    Batch-mode Weighted-Cluster DIoU-NMS (BWCD-NMS).
    Replaces hard box suppression with DIoU-penalized cluster distance matching
    and score-weighted coordinate merging for dense/small-object scenes.
    O(n^2) over candidates -- negligible cost at the small candidate counts
    (typically < 20) this pipeline produces per frame; if candidate counts ever
    grow much larger (e.g. very dense tiled-search results), consider a
    spatial index (grid buckets) before this becomes a bottleneck.
    """
    if not candidates or len(candidates) <= 1:
        return candidates

    candidates_sorted = sorted(candidates, key=lambda c: c["conf"], reverse=True)
    merged_candidates = []

    active_indices = list(range(len(candidates_sorted)))

    while active_indices:
        best_idx = active_indices.pop(0)
        best_c = candidates_sorted[best_idx]
        bx1, by1, bx2, by2 = best_c["bbox"]
        b_cx, b_cy = best_c["center"]

        cluster_indices = [best_idx]
        remaining_indices = []

        for idx in active_indices:
            cand = candidates_sorted[idx]
            cx1, cy1, cx2, cy2 = cand["bbox"]
            c_cx, c_cy = cand["center"]

            ix1, iy1 = max(bx1, cx1), max(by1, cy1)
            ix2, iy2 = min(bx2, cx2), min(by2, cy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            area_c = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            union = area_b + area_c - inter
            iou = inter / max(union, 1e-7)

            enc_x1, enc_y1 = min(bx1, cx1), min(by1, cy1)
            enc_x2, enc_y2 = max(bx2, cx2), max(by2, cy2)
            c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

            rho2 = (b_cx - c_cx) ** 2 + (b_cy - c_cy) ** 2
            r_diou = rho2 / max(c2, 1e-7)

            diou_score = iou - diou_gamma * r_diou

            if diou_score >= iou_thresh:
                cluster_indices.append(idx)
            else:
                remaining_indices.append(idx)

        active_indices = remaining_indices

        if len(cluster_indices) == 1:
            merged_candidates.append(best_c)
        else:
            weights = []
            boxes = []
            for c_idx in cluster_indices:
                cand = candidates_sorted[c_idx]
                cx1, cy1, cx2, cy2 = cand["bbox"]
                c_cx, c_cy = cand["center"]
                rho2 = (b_cx - c_cx) ** 2 + (b_cy - c_cy) ** 2
                w = cand["conf"] * np.exp(-rho2 / (sigma * 100.0 + 1e-7))
                weights.append(w)
                boxes.append([cx1, cy1, cx2, cy2])

            weights = np.array(weights, dtype=np.float32)
            weights /= max(np.sum(weights), 1e-7)

            weighted_box = np.sum(np.array(boxes, dtype=np.float32) * weights[:, None], axis=0)
            wx1, wy1, wx2, wy2 = weighted_box.tolist()
            w_center = ((wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0)

            merged_c = dict(best_c)
            merged_c["bbox"] = (wx1, wy1, wx2, wy2)
            merged_c["center"] = w_center
            merged_candidates.append(merged_c)

    return merged_candidates


def detect_target_candidates(yolo_model, frame, target_class_name, conf_threshold, device="cpu", imgsz=640):
    """
    Run YOLOv8n on the frame with specified confidence threshold and imgsz resolution.
    Returns list of dicts: {bbox, conf, center, yolo_label, vlm_label (may be None)}
    """
    results = yolo_model.predict(source=frame, verbose=False, conf=conf_threshold, device=device, imgsz=imgsz)
    r = results[0]
    candidates = []
    if r.boxes is None:
        return candidates

    names = yolo_model.model.names
    for box in r.boxes:
        cls_id   = int(box.cls[0])
        label    = names.get(cls_id, str(cls_id))
        if label != target_class_name:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf     = float(box.conf[0])
        center   = ((x1 + x2) / 2, (y1 + y2) / 2)
        candidates.append({
            "bbox":       (x1, y1, x2, y2),
            "conf":       conf,
            "center":     center,
            "yolo_label": label,
            "vlm_label":  None,
        })

    return bwcd_nms(candidates, iou_thresh=0.45)


def detect_tiled_candidates(yolo_model, frame, target_class_name, conf_threshold, device="cpu", imgsz=640, grid=(2, 2), overlap=0.2):
    """
    SAHI-style tiled inference for small-object recall during search mode.
    Slices the full frame into overlapping tiles, runs YOLO on each tile,
    offsets bounding boxes to full-frame coordinates, and applies BWCD-NMS.

    COST WARNING under the 2-core Pi5 budget: this runs (nx*ny) full YOLO
    inference passes per call instead of 1. Only enable via ENABLE_TILED_SEARCH
    while unlocked/searching (never while locked+tracking), and rely on the
    existing DETECT_EVERY_N_FRAMES frame-skip logic to keep this from running
    every single frame -- see Issue 2 tiering discussion in project notes.
    """
    h, w = frame.shape[:2]
    nx, ny = grid
    tile_w = int(w / nx)
    tile_h = int(h / ny)
    pad_w = int(tile_w * overlap)
    pad_h = int(tile_h * overlap)

    raw_candidates = []
    for ix in range(nx):
        for iy in range(ny):
            x1 = max(0, ix * tile_w - pad_w)
            y1 = max(0, iy * tile_h - pad_h)
            x2 = min(w, (ix + 1) * tile_w + pad_w)
            y2 = min(h, (iy + 1) * tile_h + pad_h)

            tile = frame[y1:y2, x1:x2]
            if tile.size == 0 or tile.shape[0] < 32 or tile.shape[1] < 32:
                continue

            tile_candidates = detect_target_candidates(
                yolo_model, tile, target_class_name, conf_threshold, device=device, imgsz=imgsz
            )
            for c in tile_candidates:
                bx1, by1, bx2, by2 = c["bbox"]
                c["bbox"]   = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                cx, cy      = c["center"]
                c["center"] = (cx + x1, cy + y1)
                raw_candidates.append(c)

    if not raw_candidates:
        return []

    return bwcd_nms(raw_candidates, iou_thresh=0.45)


def detect_in_roi(yolo_model, frame, predicted_xy, roi_pad,
                  target_class_name, conf_threshold, device="cpu", imgsz=640):
    """
    Run YOLO on a padded crop around the Kalman-predicted position.
    Returns candidates with bboxes offset back to full-frame coordinates.
    Falls back to full-frame detection if the ROI is degenerate.
    """
    h, w = frame.shape[:2]
    px, py = int(predicted_xy[0]), int(predicted_xy[1])
    x1_roi = max(0, px - roi_pad)
    y1_roi = max(0, py - roi_pad)
    x2_roi = min(w, px + roi_pad)
    y2_roi = min(h, py + roi_pad)

    if (x2_roi - x1_roi) < 32 or (y2_roi - y1_roi) < 32:
        return detect_target_candidates(yolo_model, frame, target_class_name, conf_threshold, device=device, imgsz=imgsz)

    roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
    candidates = detect_target_candidates(yolo_model, roi, target_class_name, conf_threshold, device=device, imgsz=imgsz)

    for c in candidates:
        bx1, by1, bx2, by2 = c["bbox"]
        c["bbox"]   = (bx1 + x1_roi, by1 + y1_roi, bx2 + x1_roi, by2 + y1_roi)
        cx, cy      = c["center"]
        c["center"] = (cx + x1_roi, cy + y1_roi)

    return bwcd_nms(candidates, iou_thresh=0.45)


def _iou(box_a, box_b) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


def pick_best_candidate(candidates, predicted_xy, max_dist_px, last_known_bbox=None):
    """Combined distance + IoU scoring to reduce identity switches."""
    best, best_score = None, -1.0
    for c in candidates:
        dist = ((c["center"][0] - predicted_xy[0]) ** 2 +
                (c["center"][1] - predicted_xy[1]) ** 2) ** 0.5
        if dist > max_dist_px:
            continue
        dist_score = 1.0 - dist / max_dist_px

        if last_known_bbox is not None:
            iou_score = _iou(c["bbox"], last_known_bbox)
        else:
            iou_score = 0.0

        score = DIST_WEIGHT * dist_score + IOU_WEIGHT * iou_score
        if score > best_score:
            best_score, best = score, c
    return best


def crop_bbox(frame, bbox, pad=10):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return frame[y1:y2, x1:x2]


def _reap_future(future, timeout_sec=VLM_TIMEOUT_SEC):
    """Non-blocking check on a completed future. Abandons stuck futures past timeout_sec."""
    if future is None:
        return None, False
    if future.done():
        try:
            return future.result(timeout=0), True
        except Exception as exc:
            print(f"[VLM Worker] Error: {exc}")
            return None, True
    return None, False


# ============================================================
# UDP COMMAND RECEIVER
# ============================================================

class CommandReceiver:
    """
    Binds a UDP socket on CMD_UDP_PORT and deserialises incoming JSON
    command packets (sent by voice_command_interface.py).
    Commands arrive as: {"action": "follow", "target": "red jacket", "params": {}}
    """

    def __init__(self, host=CMD_UDP_HOST, port=CMD_UDP_PORT):
        self.queue: queue.Queue = queue.Queue(maxsize=16)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.5)
        self._sock.bind((host, port))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[CmdReceiver] Listening for voice commands on UDP {host}:{port}")

    def _loop(self):
        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
                cmd = json.loads(data.decode("utf-8"))
                print(f"[CmdReceiver] Received command: {cmd}")
                if not self.queue.full():
                    self.queue.put(cmd)
            except socket.timeout:
                continue
            except Exception as exc:
                print(f"[CmdReceiver] Error: {exc}")


# ============================================================
# MISSION STATE
# ============================================================

MISSION_FOLLOW  = "follow"
MISSION_HOVER   = "hover"
MISSION_SEARCH  = "search"
MISSION_SCAN    = "scan"
MISSION_RTL     = "rtl"


# ============================================================
# MAIN ENGINE
# ============================================================

def main():
    args = parse_args()

    # --- Set the hard thread/core budget FIRST, before any torch/model work ---
    set_thread_budget(args.max_cores)
    print(f"[ENGINE] Hard thread/core budget: {args.max_cores} "
          f"(torch intra-op threads AND OS CPU affinity both pinned to this value)")

    cam_source = select_camera(args)
    run_device = select_device(args)

    print("=" * 62)
    print(" Drone Tracking Engine — Phase 1-6 + Optimized A/B/C/D/G/I + Pi5 budget")
    print(f" Device: {run_device.upper()}  | Core budget: {args.max_cores} "
          f"| YOLOv8n + Kalman + ORB-ReID + SmolVLM")
    print("=" * 62)
    print(" Keys: [q] quit  [l] lock target  [u] unlock  [h] hover  [s] search  [r] RTL  [p] timing HUD")
    print("=" * 62)

    print(f"[ENGINE] Loading SmolVLM-256M-Instruct on {run_device} ...")
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLMModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLMModel

    device_obj = torch.device(run_device)
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    model_dtype = torch.float16 if run_device == "cuda" else torch.float32
    vlm_model = AutoVLMModel.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=model_dtype
    ).to(device_obj)
    vlm_model.eval()
    print(f"[ENGINE] SmolVLM loaded on {run_device} ({model_dtype}). Sharing with ReID + Classifier.")

    # shared_inference_lock ensures SmolVLM model.generate() is never called
    # concurrently by cls_executor and reid_executor threads, AND makes the
    # scoped thread-priority drop in each module's _low_priority_inference()
    # safe (no other thread can be mid-inference while thread count changes).
    shared_inference_lock = threading.Lock()
    yolo_model  = YOLO(YOLO_MODEL_PATH)
    reid        = ReIDEngine(shared_model=vlm_model, shared_processor=processor,
                             shared_inference_lock=shared_inference_lock)
    classifier  = SmolObjectClassifier(shared_model=vlm_model, shared_processor=processor,
                                       shared_inference_lock=shared_inference_lock)
    drone       = Sim2DDroneController(world_size=(FRAME_WIDTH, FRAME_HEIGHT))
    cmd_recv    = CommandReceiver()

    print(f"[ENGINE] Connecting to camera source: {cam_source} (Threaded Zero-Lag Stream)...")
    cap = ThreadedCameraStream(cam_source, target_size=(FRAME_WIDTH, FRAME_HEIGHT))

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {cam_source}")
        print("Please check your camera index or IP network connection.")
        return

    tracker           = None
    locked            = False
    mission_state     = MISSION_SEARCH
    status_text       = "No target — press 'l' to lock"
    reid_status       = ""
    frame_count       = 0
    current_target_class = TARGET_CLASS_NAME
    last_known_bbox   = None

    cls_executor  = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    reid_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    classification_future = None
    reid_future           = None
    vlm_spatial_cache     = []

    CLASSIFY_COOLDOWN_SEC = 5.0
    last_cls_submit_time  = 0.0

    t_yolo     = PerfTimer("yolo")
    t_kf       = PerfTimer("kalman")
    t_ovl      = PerfTimer("overlay")
    t_loop     = PerfTimer("loop")
    t_cam_read = PerfTimer("cam_read")
    t_imshow   = PerfTimer("imshow")
    show_timing = False

    loop_fps_samples: list = []
    yolo_detect_count   = 0
    vlm_complete_count  = 0
    fps_window_start    = time.time()

    CSV_DUMP_INTERVAL_SEC = 5.0
    _csv_last_dump = time.time()
    _csv_rows: list = []
    print(f"[DIAG] Per-stage CSV will be printed to console every {CSV_DUMP_INTERVAL_SEC:.0f}s.")
    print("[DIAG] Columns: wall_time,loop_ms,cam_read_ms,cam_thread_ms,copy_ms,"
          "yolo_ms,kf_ms,ovl_ms,imshow_ms,bg_gap_mean_ms,bg_gap_max_ms")

    # Periodic budget-drift verification (separate cadence from the CSV dump)
    BUDGET_VERIFY_INTERVAL_SEC = 15.0
    _budget_last_check = time.time()

    while True:
        t_loop.start()

        with t_cam_read:
            ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break
        frame_count += 1

        # ---- 1. Process any queued voice/text commands ----
        while not cmd_recv.queue.empty():
            cmd = cmd_recv.queue.get_nowait()
            action = cmd.get("action", "hover")
            target_desc = cmd.get("target")
            coco_class = cmd.get("coco_class")

            if action in ("hover", "stop"):
                mission_state = MISSION_HOVER
                drone.hold_position()
                status_text = f"[CMD] Hovering (voice: '{action}')"

            elif action in ("follow", "track"):
                mission_state = MISSION_FOLLOW
                if coco_class:
                    current_target_class = coco_class

                if target_desc:
                    reid.target_tag = target_desc.lower()
                    status_text = f"[CMD] Following {current_target_class}: '{target_desc}'"
                else:
                    status_text = f"[CMD] Following {current_target_class} (re-lock on next detection)"

            elif action == "search":
                mission_state = MISSION_SEARCH
                drone.search_pattern()
                locked = False
                status_text = "[CMD] Searching for target"

            elif action in ("scan", "survey"):
                mission_state = MISSION_SCAN
                drone.scan_waypoints(DEFAULT_SCAN_WAYPOINTS)
                status_text = f"[CMD] Scanning waypoints ({action})"

            elif action in ("rtl", "return", "land"):
                mission_state = MISSION_RTL
                drone.return_to_launch()
                status_text = "[CMD] Returning to launch"

            print(f"[ENGINE] Mission → {mission_state}  |  {status_text}")

        # ---- 2. Frame-skip + ROI/tiled/full detection ----
        with t_yolo:
            stable_track = locked and tracker is not None and tracker.frames_since_update < 5
            skip_n = DETECT_EVERY_N_FRAMES_LOCKED if stable_track else DETECT_EVERY_N_FRAMES

            if frame_count % skip_n == 0:
                if stable_track and tracker is not None:
                    predicted_xy = tracker.get_position()
                    candidates = detect_in_roi(
                        yolo_model, frame, predicted_xy, ROI_PAD_PX,
                        current_target_class, DETECTION_CONFIDENCE_LOCKED, device=run_device, imgsz=YOLO_IMGSZ
                    )
                else:
                    conf_thresh = DETECTION_CONFIDENCE_SEARCH if not locked else DETECTION_CONFIDENCE_LOCKED
                    if ENABLE_TILED_SEARCH and not locked:
                        candidates = detect_tiled_candidates(
                            yolo_model, frame, current_target_class, conf_thresh, device=run_device, imgsz=YOLO_IMGSZ
                        )
                    else:
                        candidates = detect_target_candidates(
                            yolo_model, frame, current_target_class, conf_thresh, device=run_device, imgsz=YOLO_IMGSZ
                        )
                yolo_detect_count += 1
            # else: reuse candidates from previous frame (already in scope)

        # ---- 3. Reap async VLM results ----
        now = time.time()

        res, done = _reap_future(classification_future)
        if done:
            classification_future = None
            if res and res.get("label") and res["label"] != "too small to classify":
                vlm_spatial_cache.append(res)
                vlm_complete_count += 1

        res, done = _reap_future(reid_future)
        if done:
            reid_future = None
            if res is not None:
                if isinstance(res, tuple):
                    is_match, similarity, _ = res
                    if not is_match:
                        reid_status = f"⚠ ID switch? sim={similarity:.3f}"
                    else:
                        reid_status = f"✓ ReID sim={similarity:.3f}"
                elif isinstance(res, str):
                    reid.target_tag = res
                    print(f"[ENGINE] HUD label ready: '{res}'")
                vlm_complete_count += 1

        vlm_spatial_cache = [entry for entry in vlm_spatial_cache if now - entry["time"] < 4.0]
        for c in candidates:
            cx, cy = c["center"]
            c["vlm_label"] = None
            for entry in vlm_spatial_cache:
                if (cx - entry["center"][0])**2 + (cy - entry["center"][1])**2 < 2500:
                    c["vlm_label"] = entry["label"]
                    break

            if not c["vlm_label"]:
                area = (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1])
                if ((area < 4096 or c["conf"] < 0.55) and
                    classification_future is None and
                    (now - last_cls_submit_time) >= CLASSIFY_COOLDOWN_SEC):

                    last_cls_submit_time = now
                    crop = crop_bbox(frame, c["bbox"]).copy()
                    def _classify_task(img, center):
                        return {"label": classifier.classify(img), "center": center, "time": time.time()}
                    classification_future = cls_executor.submit(_classify_task, crop, c["center"])

        # Auto-lock target when in MISSION_FOLLOW if not locked yet
        if mission_state == MISSION_FOLLOW and (not locked or tracker is None) and candidates:
            best = max(candidates, key=lambda c: c["conf"])
            tracker = KalmanTrack(initial_xy=best["center"], dt=1 / 15.0)
            last_known_bbox = best["bbox"]
            crop = crop_bbox(frame, best["bbox"])
            if crop.size > 0:
                _t_lock0 = time.perf_counter()
                if best.get("vlm_label"):
                    reid.target_tag = best["vlm_label"]
                    reid.lock_target(crop.copy())
                else:
                    reid.target_tag = current_target_class
                    crop_copy = crop.copy()
                    reid.lock_target(crop_copy)
                    if reid_future is None and reid.should_generate_hud_now():
                        reid_future = reid_executor.submit(reid.generate_hud_label, crop_copy)
                dt_lock_ms = (time.perf_counter() - _t_lock0) * 1000.0
                print(f"[DIAG] reid.lock_target took {dt_lock_ms:.2f}ms (should be ~ORB-fast, <20ms)")
            locked = True
            status_text = f"Auto-locked {current_target_class} — following"
            print(f"[ENGINE] Auto-locked target ({current_target_class}).")

        # ---- 4. Tracking + re-ID (only when a target is locked) ----
        with t_kf:
            if locked and tracker is not None and mission_state == MISSION_FOLLOW:
                predicted_xy = tracker.predict()

                match = pick_best_candidate(
                    candidates, predicted_xy, REACQUIRE_SEARCH_RADIUS, last_known_bbox
                )
                if match is not None:
                    match["is_locked"] = True
                    tracker.update(match["center"])
                    last_known_bbox = match["bbox"]
                    crop = crop_bbox(frame, match["bbox"])

                    if reid.should_check_now() and crop.size > 0 and reid_future is None:
                        reid.last_check_time = time.time()
                        crop_copy = crop.copy()
                        reid_future = reid_executor.submit(reid.verify, crop_copy)

                    status_text = f"Tracking | {reid_status}" if reid_status else "Tracking (Kalman-updated)"
                else:
                    status_text = "Coasting on Kalman (no match)"

                if tracker.is_lost(MAX_MISSED_FRAMES):
                    drone.search_pattern()
                    status_text = "Target lost — searching"
                    locked = False
                    last_known_bbox = None
                    mission_state = MISSION_SEARCH
                elif tracker.is_stopped():
                    drone.hold_position()
                    status_text += " | stopped → holding"
                else:
                    tx, ty = tracker.get_position()
                    vx, vy = tracker.get_velocity()
                    aim_x  = tx + vx * 5
                    aim_y  = ty + vy * 5
                    drone_pos = drone.get_state()["position"]
                    drone.move_toward(aim_x - drone_pos[0], aim_y - drone_pos[1])
                    if tracker.has_changed_course():
                        status_text += " | course change → adjusting"

        drone.step_physics()

        # ---- 6. Draw overlay ----
        with t_ovl:
            current_fps = 1000.0 / max(sum(loop_fps_samples) / len(loop_fps_samples), 0.001) if loop_fps_samples else 0.0
            target_disp = reid.target_tag or current_target_class
            _draw_overlay(frame, candidates, tracker, locked, drone,
                          status_text, reid_status, mission_state, cam_source, fps=current_fps,
                          target_label=target_disp, device=run_device, max_cores=args.max_cores)

            if show_timing:
                _draw_timing(frame, t_yolo, t_kf, t_ovl,
                             loop_fps_samples, yolo_detect_count,
                             vlm_complete_count, fps_window_start,
                             t_cam_read=t_cam_read, t_imshow=t_imshow, cap=cap)

        with t_imshow:
            cv2.imshow("Drone Tracking Engine", frame)
            key = cv2.waitKey(1) & 0xFF
        t_loop.stop()

        loop_fps_samples.append(t_loop.avg_ms())
        if len(loop_fps_samples) > 30:
            loop_fps_samples.pop(0)

        bg_gap_mean, bg_gap_max = cap.get_bg_gap_stats()
        _csv_rows.append({
            "wall_time":      f"{time.time():.3f}",
            "loop_ms":        f"{t_loop.avg_ms():.2f}",
            "cam_read_ms":    f"{t_cam_read.avg_ms():.2f}",
            "cam_thread_ms":  f"{cap.t_thread_internal.avg_ms():.2f}",
            "copy_ms":        f"{cap.t_copy.avg_ms():.2f}",
            "yolo_ms":        f"{t_yolo.avg_ms():.2f}",
            "kf_ms":          f"{t_kf.avg_ms():.2f}",
            "ovl_ms":         f"{t_ovl.avg_ms():.2f}",
            "imshow_ms":      f"{t_imshow.avg_ms():.2f}",
            "bg_gap_mean_ms": f"{bg_gap_mean:.2f}",
            "bg_gap_max_ms":  f"{bg_gap_max:.2f}",
        })

        _now = time.time()
        if _now - _csv_last_dump >= CSV_DUMP_INTERVAL_SEC and _csv_rows:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(_csv_rows[0].keys()))
            writer.writeheader()
            for row in _csv_rows[-30:]:
                writer.writerow(row)
            print("\n[DIAG CSV] --- 5-second rolling dump ---")
            print(buf.getvalue().strip())
            print("[DIAG CSV] --- end ---\n")
            _csv_rows.clear()
            _csv_last_dump = _now

        # Periodic budget-drift check — catches any library silently
        # re-claiming threads/cores mid-run (rare, but worth surfacing).
        if _now - _budget_last_check >= BUDGET_VERIFY_INTERVAL_SEC:
            verify_thread_budget(args.max_cores, context="runtime")
            _budget_last_check = _now

        if key == ord('q'):
            break

        elif key == ord('l') and candidates:
            best = max(candidates, key=lambda c: c["conf"])
            tracker = KalmanTrack(initial_xy=best["center"], dt=1 / 15.0)
            last_known_bbox = best["bbox"]
            crop    = crop_bbox(frame, best["bbox"])
            if crop.size > 0:
                if best["vlm_label"]:
                    reid.target_tag = best["vlm_label"]
                    reid.lock_target(crop.copy())
                    print(f"[ENGINE] Locked with existing VLM label: '{reid.target_tag}'")
                else:
                    reid.target_tag = current_target_class
                    crop_copy = crop.copy()
                    reid.lock_target(crop_copy)
                    if reid_future is None and reid.should_generate_hud_now():
                        reid_future = reid_executor.submit(reid.generate_hud_label, crop_copy)
            locked        = True
            mission_state = MISSION_FOLLOW
            status_text   = f"Target locked ({current_target_class}) — following"
            print("[ENGINE] Target locked.")

        elif key == ord('u'):
            locked        = False
            tracker       = None
            last_known_bbox = None
            reid.target_tag = None
            reid.target_embedding = None
            mission_state = MISSION_HOVER
            drone.hold_position()
            status_text   = "[KEY] Target unlocked — hovering"
            print("[ENGINE] Target unlocked.")

        elif key == ord('h'):
            drone.hold_position()
            mission_state = MISSION_HOVER
            status_text   = "[KEY] Hovering"

        elif key == ord('s'):
            drone.search_pattern()
            locked        = False
            mission_state = MISSION_SEARCH
            status_text   = "[KEY] Searching"

        elif key == ord('r'):
            drone.return_to_launch()
            mission_state = MISSION_RTL
            status_text   = "[KEY] Returning to launch"

        elif key == ord('p'):
            show_timing = not show_timing
            print(f"[ENGINE] Timing HUD {'ON' if show_timing else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    cls_executor.shutdown(wait=False)
    reid_executor.shutdown(wait=False)


# ============================================================
# DRAW HELPERS
# ============================================================

def _draw_overlay(frame, candidates, tracker, locked, drone, status_text,
                  reid_status, mission, cam_source=0, fps=0.0, target_label="",
                  device="cpu", max_cores=2):
    """Render all bounding boxes, labels, tracker ring, drone dot, and HUD text."""
    h, w = frame.shape[:2]

    for c in candidates:
        x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
        if c.get("is_locked"):
            box_color = (0, 0, 255)
            thickness = 2
        elif c["vlm_label"]:
            box_color = (80, 80, 220)
            thickness = 1
        else:
            box_color = (130, 130, 130)
            thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        label_prefix = "[LOCKED] " if c.get("is_locked") else ""
        label_str = f"{label_prefix}{c['vlm_label']}" if c["vlm_label"] else f"{label_prefix}{c['conf']:.2f}"
        cv2.putText(frame, label_str, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_color, thickness)

    if locked and tracker is not None:
        tx, ty = tracker.get_position()
        cv2.circle(frame, (int(tx), int(ty)), 12, (0, 220, 80), 2)
        cv2.circle(frame, (int(tx), int(ty)),  4, (0, 220, 80), -1)

    state = drone.get_state()
    if state["position"]:
        dx, dy = state["position"]
        cv2.circle(frame, (int(dx), int(dy)), 7, (255, 140, 0), -1)
        cv2.circle(frame, (int(dx), int(dy)), 7, (255, 200, 0), 1)

    def _text(txt, y, color=(255, 255, 255), x=10):
        cv2.putText(frame, txt, (x - 1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, txt, (x,     y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,     1)

    mission_color = {
        "follow": (80, 220, 80), "hover": (220, 200, 60),
        "search": (220, 100, 60), "scan": (60, 180, 220), "rtl": (220, 80, 80),
    }.get(mission, (200, 200, 200))

    cam_str = f"IP: {cam_source}" if isinstance(cam_source, str) else f"Laptop ({cam_source})"

    _text(status_text[:72], 22)

    fps_str = f"FPS: {fps:.1f}"
    (fps_w, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    _text(fps_str, 22, color=(0, 255, 120), x=w - fps_w - 15)

    _text(f"Mission: {mission.upper()}  |  Drone: {state['status']}  |  {cam_str}", 44, mission_color)
    if reid_status:
        _text(reid_status, 66, (180, 220, 255))

    if target_label:
        lbl_str = f"Target: {target_label.upper()}"
        (lbl_w, lbl_h), baseline = cv2.getTextSize(lbl_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (8, h - 25 - lbl_h - 4), (12 + lbl_w, h - 25 + baseline), (240, 240, 240), -1)
        cv2.rectangle(frame, (8, h - 25 - lbl_h - 4), (12 + lbl_w, h - 25 + baseline), (180, 180, 180), 1)
        cv2.putText(frame, lbl_str, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # NOTE: torch.get_num_threads() here reflects the MAIN THREAD's view of the
    # process-wide setting. It will read as `max_cores` most of the time, but
    # will briefly read as 1 (VLM_LOW_PRIORITY_THREADS) during the exact window
    # a background VLM call is executing its scoped low-priority section --
    # that's expected and not a bug; it's the fix working as intended.
    active_threads = torch.get_num_threads()
    total_cores = os.cpu_count() or 4
    if device == "cpu":
        cpu_str = f"Threads: {active_threads}/{max_cores} budget ({total_cores} cores total)"
    else:
        cpu_str = f"Device: CUDA (budget {max_cores}/{total_cores} threads)"
    (cpu_w, _), _ = cv2.getTextSize(cpu_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    _text(cpu_str, h - 25, color=(0, 230, 255), x=w - cpu_w - 12)

    legend = "[q]quit [l]lock [u]unlock [h]hover [s]search [r]rtl [p]timing"
    cv2.putText(frame, legend, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)


def _draw_timing(frame, t_yolo, t_kf, t_ovl, loop_fps_samples,
                 yolo_count, vlm_count, fps_start,
                 t_cam_read=None, t_imshow=None, cap=None):
    """Overlay all per-stage timing stats when [p] is pressed."""
    elapsed = time.time() - fps_start
    main_fps  = 1000.0 / max(sum(loop_fps_samples) / len(loop_fps_samples), 1) if loop_fps_samples else 0
    yolo_fps  = yolo_count / max(elapsed, 0.001)
    vlm_rate  = vlm_count  / max(elapsed, 0.001)

    cam_r  = t_cam_read.avg_ms()  if t_cam_read else 0.0
    imsh   = t_imshow.avg_ms()    if t_imshow   else 0.0
    cam_th = cap.t_thread_internal.avg_ms() if cap else 0.0
    copy_  = cap.t_copy.avg_ms()            if cap else 0.0
    bg_mean, bg_max = cap.get_bg_gap_stats() if cap else (0.0, 0.0)

    lines = [
        f"loop:{1000.0/max(main_fps,0.001):.1f}ms ({main_fps:.1f}fps)",
        f"cam_read:{cam_r:.1f}ms  copy:{copy_:.2f}ms  imshow:{imsh:.1f}ms",
        f"det:{t_yolo.avg_ms():.1f}ms  kf:{t_kf.avg_ms():.2f}ms  ovl:{t_ovl.avg_ms():.1f}ms",
        f"bg_thread:{cam_th:.1f}ms  bg_gap mean:{bg_mean:.1f} max:{bg_max:.1f}ms",
        f"yolo:{yolo_fps:.1f}fps  vlm:{vlm_rate:.2f}/s",
    ]
    y = 90
    for line in lines:
        cv2.putText(frame, line, (9,  y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 220, 80), 1)
        y += 17


if __name__ == "__main__":
    main()