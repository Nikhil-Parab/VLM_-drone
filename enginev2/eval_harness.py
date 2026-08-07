"""
eval_harness.py — offline evaluation (Phase 4).

Replay video clips through grounding + SVLM control; score hit-rate and
directional-bin accuracy against optional JSON ground-truth labels.

Usage:
  python eval_harness.py --video clip.mp4 --phrase "red mug"
  python eval_harness.py --video clip.mp4 --labels labels.json --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

# enginev2 on path when run from repo
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from grounding_engine import GroundingEngine, pick_best_grounded_candidate
from svlm_controller import SVLMController, build_telemetry_text, parse_control_tag


def load_labels(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("frames", [])


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


def horizontal_bin_from_center(cx: float, frame_w: int) -> str:
    if cx < frame_w * 0.35:
        return "left"
    if cx > frame_w * 0.65:
        return "right"
    return "center"


def run_eval(
    video_path: str,
    phrase: str,
    labels_path: Optional[str] = None,
    device: str = "cpu",
    max_frames: int = 0,
    use_svlm: bool = False,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    labels = load_labels(labels_path) if labels_path else []
    label_by_frame = {int(e["frame"]): e for e in labels if "frame" in e}

    ground = GroundingEngine(device=device)
    svlm = None
    if use_svlm:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLMModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLMModel
        proc = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        model = AutoVLMModel.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.float32
        ).to(device)
        model.eval()
        svlm = SVLMController(model, proc)

    ground_hits = 0
    ground_total = 0
    dir_correct = 0
    dir_total = 0
    frame_idx = 0
    results_log: List[dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        h, w = frame.shape[:2]
        candidates = ground.ground_to_candidates(frame, phrase, top_k=3)
        gt = label_by_frame.get(frame_idx)

        if gt and gt.get("bbox"):
            ground_total += 1
            best = max(candidates, key=lambda c: c["conf"]) if candidates else None
            if best and bbox_iou(best["bbox"], gt["bbox"]) >= 0.3:
                ground_hits += 1

        if svlm and frame_idx % 15 == 0:
            telem = build_telemetry_text({"status": "eval", "altitude_m": 5.0}, locked=False)
            dec = svlm.decide(frame, phrase, telem)
            if gt and gt.get("bbox"):
                gt_cx = (gt["bbox"][0] + gt["bbox"][2]) / 2
                expected_h = horizontal_bin_from_center(gt_cx, w)
                dir_total += 1
                if dec.horizontal == expected_h:
                    dir_correct += 1
            results_log.append({"frame": frame_idx, "decision": dec.to_dict()})

        if frame_idx % 30 == 0:
            print(f"[EVAL] frame {frame_idx} candidates={len(candidates)}")

    cap.release()

    report = {
        "video": video_path,
        "phrase": phrase,
        "frames_processed": frame_idx,
        "grounding_hit_rate": ground_hits / max(ground_total, 1),
        "grounding_hits": ground_hits,
        "grounding_labeled_frames": ground_total,
        "directional_accuracy": dir_correct / max(dir_total, 1) if use_svlm else None,
        "directional_correct": dir_correct,
        "directional_total": dir_total,
        "sample_decisions": results_log[:10],
    }
    return report


def main():
    ap = argparse.ArgumentParser(description="Offline grounding/SVLM eval harness")
    ap.add_argument("--video", required=True, help="Path to test video clip")
    ap.add_argument("--phrase", default="person", help="Grounding text phrase")
    ap.add_argument("--labels", default=None, help="JSON ground-truth labels")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--svlm", action="store_true", help="Also score SVLM directional bins")
    ap.add_argument("--out", default=None, help="Write JSON report to file")
    args = ap.parse_args()

    print(f"[EVAL] Starting — video={args.video} phrase='{args.phrase}'")
    t0 = time.perf_counter()
    report = run_eval(
        args.video, args.phrase, args.labels,
        device=args.device, max_frames=args.max_frames, use_svlm=args.svlm,
    )
    report["elapsed_sec"] = time.perf_counter() - t0
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[EVAL] Report written to {args.out}")


if __name__ == "__main__":
    main()
