"""
grounding_engine.py — open-vocabulary target grounding (Phase 1).

Replaces YOLO-COCO for arbitrary text-described targets. Uses Grounding DINO
(OWL-ViT optional fallback) to return candidate boxes from a free-text phrase.

Top-K candidates can be sanity-checked by SmolVLM before lock (Phase 2 hook).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
OWL_MODEL_ID = "google/owlvit-base-patch32"
DEFAULT_TOP_K = 3
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.20

SANITY_PROMPT = (
    "Does this crop show the described target? Answer ONLY yes or no.\n"
    "Target: {phrase}\n"
    "Answer:"
)


@dataclass
class GroundingCandidate:
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    conf: float
    phrase: str
    source: str = "grounding_dino"

    def to_detection_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "center": self.center,
            "conf": self.conf,
            "yolo_label": self.phrase,
            "vlm_label": None,
            "grounding_phrase": self.phrase,
            "source": self.source,
        }


def _post_process_grounding(processor, outputs, inputs, target_sizes, box_threshold):
    """Compat wrapper: transformers >=4.55 uses threshold= not box_threshold=."""
    fn = processor.post_process_grounded_object_detection
    try:
        return fn(outputs=outputs, input_ids=inputs["input_ids"],
                  target_sizes=target_sizes, threshold=box_threshold)
    except TypeError:
        return fn(outputs=outputs, input_ids=inputs["input_ids"],
                  target_sizes=target_sizes, box_threshold=box_threshold)


class GroundingEngine:
    """
    Open-vocabulary grounding via Grounding DINO (primary) or OWL-ViT (fallback).
    """

    def __init__(
        self,
        model_id: str = GROUNDING_MODEL_ID,
        device: str = "cpu",
        backend: str = "grounding_dino",
    ):
        self.device = device
        self.backend = backend
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if self.backend == "owlvit":
            self._load_owlvit()
        else:
            self._load_grounding_dino()
        self._loaded = True

    def _load_grounding_dino(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        print(f"[GROUND] Loading {self.model_id} on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        print("[GROUND] Grounding DINO ready.")

    def _load_owlvit(self) -> None:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        print(f"[GROUND] Loading OWL-ViT {OWL_MODEL_ID} on {self.device} ...")
        self.processor = OwlViTProcessor.from_pretrained(OWL_MODEL_ID)
        self.model = OwlViTForObjectDetection.from_pretrained(OWL_MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        self.backend = "owlvit"
        print("[GROUND] OWL-ViT ready.")

    def ground(
        self,
        frame_bgr: np.ndarray,
        phrase: str,
        top_k: int = DEFAULT_TOP_K,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
    ) -> List[GroundingCandidate]:
        """Return up to top_k candidate boxes for phrase in frame."""
        if not phrase or not phrase.strip():
            return []
        self.load()
        if self.backend == "owlvit":
            return self._ground_owlvit(frame_bgr, phrase, top_k, box_threshold)
        return self._ground_dino(frame_bgr, phrase, top_k, box_threshold)

    def _ground_dino(
        self,
        frame_bgr: np.ndarray,
        phrase: str,
        top_k: int,
        box_threshold: float,
    ) -> List[GroundingCandidate]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        text = phrase.lower().strip()
        if not text.endswith("."):
            text = text + "."

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        h, w = frame_bgr.shape[:2]

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = _post_process_grounding(
            self.processor, outputs, inputs,
            target_sizes=[(h, w)], box_threshold=box_threshold,
        )
        if not results or not results[0]["boxes"].numel():
            return []

        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"]

        ranked = sorted(zip(boxes, scores, labels), key=lambda x: float(x[1]), reverse=True)
        out: List[GroundingCandidate] = []
        for box, score, _lbl in ranked[:top_k]:
            x1, y1, x2, y2 = [float(v) for v in box]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            out.append(GroundingCandidate(
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                conf=float(score),
                phrase=phrase,
                source="grounding_dino",
            ))
        return out

    def _ground_owlvit(
        self,
        frame_bgr: np.ndarray,
        phrase: str,
        top_k: int,
        box_threshold: float,
    ) -> List[GroundingCandidate]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        texts = [[phrase.lower().strip()]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(frame_bgr.shape[0], frame_bgr.shape[1])])
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=box_threshold
        )
        if not results or not results[0]["boxes"].numel():
            return []

        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        ranked = sorted(zip(boxes, scores), key=lambda x: float(x[1]), reverse=True)
        out: List[GroundingCandidate] = []
        for box, score in ranked[:top_k]:
            x1, y1, x2, y2 = [float(v) for v in box]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            out.append(GroundingCandidate(
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                conf=float(score),
                phrase=phrase,
                source="owlvit",
            ))
        return out

    def ground_to_candidates(self, frame_bgr, phrase, top_k=DEFAULT_TOP_K, **kw) -> list:
        """Engine-compatible list[dict] for drop-in beside YOLO detect paths."""
        return [c.to_detection_dict() for c in self.ground(frame_bgr, phrase, top_k=top_k, **kw)]

    def sanity_check_crop(
        self,
        crop_bgr: np.ndarray,
        phrase: str,
        shared_model,
        shared_processor,
        inference_lock=None,
    ) -> bool:
        """
        SmolVLM yes/no sanity check: does this crop match the phrase?
        Used before auto-lock on a grounding candidate.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        prompt = SANITY_PROMPT.format(phrase=phrase)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = shared_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = shared_processor(text=chat, images=[image], return_tensors="pt")
        device = next(shared_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        lock = inference_lock
        ctx = lock if lock is not None else _NullContext()
        with ctx:
            with torch.no_grad():
                gen = shared_model.generate(**inputs, max_new_tokens=4, do_sample=False)
        new_ids = gen[0][inputs["input_ids"].shape[1]:]
        text = shared_processor.decode(new_ids, skip_special_tokens=True).strip().lower()
        return text.startswith("yes") or text == "y"


class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass


def pick_best_grounded_candidate(
    candidates: list,
    frame_bgr: np.ndarray,
    phrase: str,
    grounding_engine: GroundingEngine,
    shared_model=None,
    shared_processor=None,
    inference_lock=None,
    crop_fn=None,
) -> Optional[dict]:
    """
    Run SmolVLM sanity check on top-3 grounding boxes; return first that passes.
    Falls back to highest-confidence box if no model provided.
    """
    if not candidates:
        return None
    if shared_model is None or shared_processor is None:
        return max(candidates, key=lambda c: c["conf"])

    if crop_fn is None:
        def crop_fn(f, bbox, pad=10):
            h, w = f.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            return f[y1:y2, x1:x2]

    for c in sorted(candidates, key=lambda x: x["conf"], reverse=True):
        crop = crop_fn(frame_bgr, c["bbox"])
        if grounding_engine.sanity_check_crop(
            crop, phrase, shared_model, shared_processor, inference_lock
        ):
            print(f"[GROUND] Sanity OK for conf={c['conf']:.2f} phrase='{phrase}'")
            return c
    print(f"[GROUND] No sanity pass — using top conf={candidates[0]['conf']:.2f}")
    return max(candidates, key=lambda c: c["conf"])
