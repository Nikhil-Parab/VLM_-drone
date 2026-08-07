"""
execution_layer.py — deterministic control execution (Phase 5).

Maps SVLM categorical decisions → velocity commands via lookup table.
Includes safety guardrails: velocity caps, staleness watchdog, manual override.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from svlm_controller import ControlDecision

# Hard caps — independent of Pixhawk/MAVSDK limits
MAX_SPEED = 4.0
MAX_YAW_RATE_DEG = 45.0
DECISION_STALE_SEC = 4.0

# urgency → base speed fraction
URGENCY_SCALE = {
    "hold": 0.0,
    "gentle": 0.35,
    "moderate": 0.65,
    "aggressive": 1.0,
}

# horizontal / vertical → unit direction components (image-space)
H_DIR = {"left": (-1.0, 0.0), "center": (0.0, 0.0), "right": (1.0, 0.0)}
V_DIR = {"up": (0.0, -1.0), "center": (0.0, 0.0), "down": (0.0, 1.0)}

# depth modifier on forward component (sim: push toward target when far)
DEPTH_SCALE = {
    "too_close": 0.2,
    "good_distance": 0.8,
    "far": 1.0,
    "very_far": 1.0,
}


class ExecutionLayer:
    """
    Converts ControlDecision → drone.move_toward / hold_position every frame.
    SVLM refreshes decision asynchronously; this layer runs synchronously.
    """

    def __init__(self, max_speed: float = MAX_SPEED, stale_sec: float = DECISION_STALE_SEC):
        self.max_speed = max_speed
        self.stale_sec = stale_sec
        self.active_decision: Optional[ControlDecision] = None
        self.manual_override = False
        self.manual_override_until = 0.0

    def set_decision(self, decision: ControlDecision) -> None:
        self.active_decision = decision

    def trigger_manual_override(self, duration_sec: float = 30.0) -> None:
        """Hover/RTL keyboard or voice commands call this — blocks SVLM follow."""
        self.manual_override = True
        self.manual_override_until = time.time() + duration_sec

    def clear_manual_override(self) -> None:
        self.manual_override = False
        self.manual_override_until = 0.0

    def _is_stale(self) -> bool:
        if self.active_decision is None:
            return True
        return (time.time() - self.active_decision.timestamp) > self.stale_sec

    def decision_to_velocity(self, decision: ControlDecision) -> tuple:
        """Map categorical bins → (dx, dy) pixel-space velocity intent."""
        if not decision.target_visible or decision.urgency == "hold":
            return 0.0, 0.0
        if decision.confidence == "low" and decision.urgency == "aggressive":
            # Downgrade unsafe aggressive moves when confidence is low
            urgency = "gentle"
        else:
            urgency = decision.urgency

        hx, hy = H_DIR.get(decision.horizontal, (0.0, 0.0))
        vx, vy = V_DIR.get(decision.vertical, (0.0, 0.0))
        dx = hx + vx * 0.5
        dy = hy + vy * 0.5
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0
        dx, dy = dx / dist, dy / dist

        scale = URGENCY_SCALE.get(urgency, 0.0) * DEPTH_SCALE.get(decision.depth, 0.8)
        dx *= scale * self.max_speed
        dy *= scale * self.max_speed

        # Hard cap
        spd = math.hypot(dx, dy)
        if spd > self.max_speed:
            dx = dx / spd * self.max_speed
            dy = dy / spd * self.max_speed
        return dx, dy

    def apply(self, drone, frame_size=(640, 480)) -> str:
        """
        Call once per frame. Returns status string for HUD.
        """
        if self.manual_override:
            if time.time() > self.manual_override_until:
                self.manual_override = False
            else:
                drone.hold_position()
                return "manual override — holding"

        if self._is_stale() or self.active_decision is None:
            drone.hold_position()
            return "SVLM decision stale — holding"

        dec = self.active_decision
        if not dec.target_visible:
            drone.hold_position()
            return "target not visible — holding"

        dx, dy = self.decision_to_velocity(dec)
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            drone.hold_position()
            return f"SVLM hold ({dec.confidence} conf)"

        drone.move_toward(dx, dy)
        return f"SVLM {dec.horizontal}/{dec.urgency} ({dec.confidence})"
