"""
svlm_controller.py — SVLM semantic control layer (Phase 3).

SmolVLM outputs a constrained categorical decision (pipe-delimited tag),
not free text or raw motor floats. Parsed into ControlDecision for the
execution layer.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

DECISION_INTERVAL_SEC = 1.5
DECISION_PROMPT = (
    "You are a drone vision controller. Look at the image and decide movement.\n"
    "Reply in EXACTLY this format and nothing else:\n"
    "<CTRL>visible=yes|no | h=left|center|right | v=up|center|down | "
    "d=too_close|good_distance|far|very_far | u=hold|gentle|moderate|aggressive | "
    "c=low|medium|high</CTRL>\n"
    "Target description: {target_desc}\n"
    "Telemetry: {telemetry}\n"
    "Examples:\n"
    "<CTRL>visible=yes | h=left | v=center | d=good_distance | u=moderate | c=high</CTRL>\n"
    "<CTRL>visible=no | h=center | v=center | d=far | u=hold | c=low</CTRL>\n"
)

HORIZONTAL = ("left", "center", "right")
VERTICAL = ("up", "center", "down")
DEPTH = ("too_close", "good_distance", "far", "very_far")
URGENCY = ("hold", "gentle", "moderate", "aggressive")
CONFIDENCE = ("low", "medium", "high")


@dataclass
class ControlDecision:
    target_visible: bool = False
    horizontal: str = "center"
    vertical: str = "center"
    depth: str = "good_distance"
    urgency: str = "hold"
    confidence: str = "low"
    raw_text: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "target_visible": self.target_visible,
            "horizontal": self.horizontal,
            "vertical": self.vertical,
            "depth": self.depth,
            "urgency": self.urgency,
            "confidence": self.confidence,
        }


def _norm_choice(val: str, choices: tuple, default: str) -> str:
    v = (val or "").strip().lower().replace("-", "_")
    if v in choices:
        return v
    for c in choices:
        if c in v or v in c:
            return c
    return default


def parse_control_tag(raw_text: str) -> ControlDecision:
    """Parse <CTRL>...</CTRL> or fall back to safe hold defaults."""
    match = re.search(r"<ctrl>(.*?)</ctrl>", raw_text, re.IGNORECASE | re.DOTALL)
    body = match.group(1) if match else raw_text
    fields = {}
    for part in body.split("|"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            fields[k.strip().lower()] = v.strip().lower()

    visible_raw = fields.get("visible", "no")
    visible = visible_raw in ("yes", "true", "1", "y")

    return ControlDecision(
        target_visible=visible,
        horizontal=_norm_choice(fields.get("h", fields.get("horizontal", "")), HORIZONTAL, "center"),
        vertical=_norm_choice(fields.get("v", fields.get("vertical", "")), VERTICAL, "center"),
        depth=_norm_choice(fields.get("d", fields.get("depth", "")), DEPTH, "good_distance"),
        urgency=_norm_choice(fields.get("u", fields.get("urgency", "")), URGENCY, "hold"),
        confidence=_norm_choice(fields.get("c", fields.get("confidence", "")), CONFIDENCE, "low"),
        raw_text=raw_text,
        timestamp=time.time(),
    )


def build_telemetry_text(
    drone_state: dict,
    target_center: Optional[tuple] = None,
    frame_size: tuple = (640, 480),
    locked: bool = False,
) -> str:
    """Plain-text telemetry summary for the SVLM prompt."""
    status = drone_state.get("status", "idle")
    alt = drone_state.get("altitude_m", 0.0)
    parts = [f"moving {status}", f"altitude {alt:.1f}m"]
    if locked and target_center:
        fw, fh = frame_size
        tx, ty = target_center
        h_pos = "left-of-center" if tx < fw * 0.4 else ("right-of-center" if tx > fw * 0.6 else "center")
        v_pos = "above-center" if ty < fh * 0.4 else ("below-center" if ty > fh * 0.6 else "center")
        parts.append(f"target {h_pos}, {v_pos} in frame")
    else:
        parts.append("target not locked")
    return ", ".join(parts)


class SVLMController:
    """Async-friendly SmolVLM decision producer."""

    def __init__(self, shared_model, shared_processor, inference_lock=None):
        self.model = shared_model
        self.processor = shared_processor
        self.device = next(shared_model.parameters()).device
        self._inference_lock = inference_lock
        self.last_decision = ControlDecision()
        self.last_decision_time = 0.0

    def should_decide_now(self) -> bool:
        return (time.time() - self.last_decision_time) >= DECISION_INTERVAL_SEC

    def decide(
        self,
        frame_bgr: np.ndarray,
        target_desc: str,
        telemetry: str,
        crop_bgr: Optional[np.ndarray] = None,
    ) -> ControlDecision:
        """Run SmolVLM once; return parsed ControlDecision."""
        img = crop_bgr if crop_bgr is not None and crop_bgr.size > 0 else frame_bgr
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        prompt = DECISION_PROMPT.format(
            target_desc=target_desc or "unknown target",
            telemetry=telemetry,
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat, images=[image], return_tensors="pt").to(self.device)

        lock = self._inference_lock
        if lock:
            lock.acquire()
        try:
            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
        finally:
            if lock:
                lock.release()

        new_ids = gen[0][inputs["input_ids"].shape[1]:]
        decoded = self.processor.decode(new_ids, skip_special_tokens=True)
        decision = parse_control_tag(decoded)
        self.last_decision = decision
        self.last_decision_time = time.time()
        print(f"[SVLM-CTRL] {decision.to_dict()}")
        return decision
