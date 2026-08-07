"""
small_object_classifier.py — SmolVLM description for small/uncertain
YOLO detections (box area < SMALL_OBJECT_AREA_PX or conf < threshold).

Pass shared_model/shared_processor to reuse the SmolVLM instance from
ReIDEngine (skips a second ~987MB load). Pass shared_inference_lock
(shared with ReIDEngine) so classify() and generate_hud_label() never
run model.generate() concurrently - without it, both callers contend
for the same model object and starve the main loop.
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image

# Thread budget is set once by drone_tracking_engine.set_thread_budget() — do not override here.

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLMModel

from transformers import AutoProcessor

# ============================================================
# CONFIGURATION  <-- adjust these to tune when VLM fires
# ============================================================

SMOLVLM_MODEL_ID       = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEBUG = False  # <-- SET True to re-enable per-call VLM timing/frequency logs if lag reappears

SMALL_OBJECT_AREA_PX   = 64 * 64    # px²  -- boxes smaller than this get classified
SMALL_OBJECT_CONF_THRESH = 0.55     # YOLO confidence -- below this also triggers VLM
CLASSIFY_COOLDOWN_SEC  = 3.0        # seconds -- minimum gap per track_id between VLM calls
MIN_CROP_DIM           = 8          # px  -- don't bother on crops this tiny (no signal)

CLASSIFY_PROMPT = (
    "Describe this object in 6 words or fewer. "
    "Include: object type, dominant colour, and any distinctive feature. "
    "Output ONLY the description, no other text."
)  # <-- CHANGE CLASSIFY PROMPT HERE


class SmolObjectClassifier:
    """
    Wraps SmolVLM-256M for on-demand small-object description.

    Parameters
    ----------
    shared_model          : pre-loaded AutoVLMModel from ReIDEngine (or None to load fresh)
    shared_processor      : pre-loaded AutoProcessor from ReIDEngine (or None to load fresh)
    shared_inference_lock : threading.Lock shared with ReIDEngine so classify() and
                            generate_hud_label() cannot call model.generate() concurrently.
                            CRITICAL: without this, both calls use the same model object
                            simultaneously -> CPU thread starvation -> main-loop lag.
    """

    def __init__(
        self,
        shared_model=None,
        shared_processor=None,
        model_id: str = SMOLVLM_MODEL_ID,
        shared_inference_lock=None,          # H1/H2 fix: shared VLM serialisation lock
    ):
        """
        Parameters
        ----------
        model_id              : HuggingFace model ID to load (if not using shared instance)
        shared_model          : pre-loaded AutoVLMModel — pass this to skip a second model load
        shared_processor      : pre-loaded AutoProcessor — must be provided with shared_model
        shared_inference_lock : threading.Lock shared with ReIDEngine to prevent
                                concurrent model.generate() calls from cls_executor and
                                reid_executor threads simultaneously. Without this, both
                                callers run on the same model object at once -> starvation.
        """
        # H1+H2 fix: shared inference lock
        self._inference_lock = shared_inference_lock or threading.Lock()

        if shared_model is not None and shared_processor is not None:
            print("[SmolObjectClassifier] Using shared SmolVLM model instance (no second load).")
            self.model = shared_model
            self.processor = shared_processor
            self.device = next(shared_model.parameters()).device
        else:
            print(f"[SmolObjectClassifier] Loading {model_id} ...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoVLMModel.from_pretrained(
                model_id, torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            print(f"[SmolObjectClassifier] Loaded on {self.device}.")

        self._last_classified = {}
        self._call_count = 0    # only used/incremented when DEBUG=True

    def classify(self, crop_bgr: np.ndarray) -> str:
        """
        Run SmolVLM on a single cropped image region.
        Returns a short (<6 word) natural language description string.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "too small to classify"

        h, w = crop_bgr.shape[:2]
        if min(h, w) < MIN_CROP_DIM:
            return "too small to classify"

        if DEBUG:
            self._call_count += 1
            print(f"[CLASSIFY] call #{self._call_count}")
        _t0 = time.perf_counter()

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CLASSIFY_PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)

        with self._inference_lock:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=12,        # Phase B: was 20; 6-word desc fits in 12 tokens
                    do_sample=False,          # greedy — faster, more consistent labels
                )

        if DEBUG:
            print(f"[CLASSIFY] done  dt={time.perf_counter()-_t0:.2f}s")

        new_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
        decoded = self.processor.decode(new_ids, skip_special_tokens=True)
        # Strip "Assistant:" prefix that some chat templates add
        label = decoded.strip().lower()
        for prefix in ("assistant:", "assistant\n", "assistant "):
            if label.startswith(prefix):
                label = label[len(prefix):].strip()
                break

        return label

    def classify_if_small(
        self,
        crop_bgr: np.ndarray,
        bbox: Tuple[float, float, float, float],
        yolo_conf: float,
        track_id: int = -1,
    ) -> Optional[str]:
        """
        Returns a VLM label only when:
          • the bounding box area is below SMALL_OBJECT_AREA_PX, OR
          • the YOLO confidence is below SMALL_OBJECT_CONF_THRESH,
          AND the per-track cooldown has elapsed.

        Returns None when the detection is large/confident enough that the
        regular YOLO label is sufficient (or the cooldown hasn't elapsed yet).

        Parameters
        ----------
        crop_bgr  : cropped BGR frame around the detection
        bbox      : (x1, y1, x2, y2) from YOLO
        yolo_conf : YOLO class confidence
        track_id  : tracker ID for cooldown bookkeeping (-1 = always run)
        """
        x1, y1, x2, y2 = bbox
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))

        needs_vlm = (area < SMALL_OBJECT_AREA_PX) or (yolo_conf < SMALL_OBJECT_CONF_THRESH)
        if not needs_vlm:
            return None

        # Cooldown check
        now = time.time()
        if track_id != -1:
            last_t = self._last_classified.get(track_id, 0.0)
            if (now - last_t) < CLASSIFY_COOLDOWN_SEC:
                return None   # still within cooldown window
            self._last_classified[track_id] = now

        label = self.classify(crop_bgr)
        print(
            f"[SmolObjectClassifier] track={track_id} area={int(area)}px² "
            f"conf={yolo_conf:.2f} → '{label}'"
        )
        return label

    def reset_cooldown(self, track_id: int):
        """Force a fresh classification on the next call for this track_id."""
        self._last_classified.pop(track_id, None)

    def clear_stale_cooldowns(self, active_track_ids: set):
        """Remove cooldown entries for tracks that no longer exist."""
        stale = [tid for tid in self._last_classified if tid not in active_track_ids]
        for tid in stale:
            del self._last_classified[tid]


# ============================================================
# Quick standalone smoke-test
#   python small_object_classifier.py
# ============================================================

if __name__ == "__main__":
    print("=== SmolObjectClassifier smoke-test ===")
    classifier = SmolObjectClassifier()  # loads its own model copy

    # Synthetic tiny crop (black square) — just proves inference runs
    fake_crop = np.zeros((40, 40, 3), dtype="uint8")
    label = classifier.classify(fake_crop)
    print(f"Synthetic crop label: '{label}'")

    # classify_if_small — small bbox → should fire
    bbox_small = (10, 10, 40, 40)          # 30×30 = 900 px² < 4096 threshold
    result = classifier.classify_if_small(fake_crop, bbox_small, yolo_conf=0.7, track_id=1)
    print(f"Small bbox result: '{result}'")

    # classify_if_small — large bbox, high conf → should return None
    bbox_large = (0, 0, 200, 200)          # 200×200 = 40000 px² > threshold
    result = classifier.classify_if_small(fake_crop, bbox_large, yolo_conf=0.9, track_id=2)
    print(f"Large bbox result: {result}  (expected: None)")

    print("=== Done ===")
