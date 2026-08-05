"""
kalman_tracker.py
==================
A minimal, dependency-free (numpy only) constant-velocity Kalman filter for
2D position tracking, plus stop/course-change detection derived from the
filter's own velocity state.

State vector: [x, y, vx, vy]
Measurement:  [x, y]  (detector gives us position only; velocity is inferred)

Why constant-velocity and not something fancier: it's the standard choice
for short-horizon visual tracking (this is what SORT/DeepSORT use under the
hood too) and it's cheap enough to run every frame on a Pi.
"""

import math
import numpy as np


class KalmanTrack:
    def __init__(self, initial_xy, dt=1 / 15.0):
        """
        initial_xy: (x, y) pixel position of the first detection
        dt: expected seconds between frames (used to build the motion model)
        """
        self.dt = dt

        # State: [x, y, vx, vy]
        self.x = np.array([initial_xy[0], initial_xy[1], 0.0, 0.0], dtype=float)

        # State transition (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # We only measure position, not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Process noise - how much we trust the constant-velocity assumption.
        # Higher = filter adapts faster to real direction changes but is
        # noisier; lower = smoother but slower to react.
        q = 2.0  # <-- CHANGE THIS to tune responsiveness to course changes
        self.Q = np.eye(4) * q

        # Measurement noise - how much we trust the detector's box center.
        r = 4.0  # <-- CHANGE THIS to tune trust in raw detections
        self.R = np.eye(2) * r

        # State covariance - starts uncertain, shrinks as we get measurements
        self.P = np.eye(4) * 50.0

        # Bookkeeping for stop/course-change detection.
        #
        # IMPORTANT DESIGN NOTE: the Kalman-filtered velocity (self.x[2:4])
        # is smoothed through the state covariance and reacts to real
        # changes over roughly a second or more - fine for dead-reckoning
        # prediction during a missed detection, but too laggy to notice
        # "the target just stopped" quickly. So stop/course-change decisions
        # use a separate, fast finite-difference velocity computed directly
        # from recent raw positions instead of the Kalman state.
        self.position_history = [initial_xy]   # recent (x, y) positions
        self.position_history_len = 6           # <-- CHANGE window for stop/course-change speed
        self.heading_history = []                # recent instantaneous headings (radians)
        self.speed_history = []                  # recent instantaneous speeds (px/sec)
        self.history_len = 8                     # <-- CHANGE window size for course-change sensitivity
        self.frames_since_update = 0
        self.age = 0

    # ------------------------------------------------------------------
    # Core Kalman steps
    # ------------------------------------------------------------------

    def predict(self):
        """Advance the state one time step using the motion model only."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.frames_since_update += 1
        self.age += 1
        # NOTE: deliberately NOT recording motion here - stop/course-change
        # detection should reflect real measurements, not accumulate fake
        # "motion" samples while we're just coasting on prediction during
        # a missed detection.
        return self.get_position()

    def update(self, measured_xy):
        """Correct the state using a new detection (measured x, y)."""
        z = np.array(measured_xy, dtype=float)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.frames_since_update = 0
        self._record_motion(tuple(z))

    def get_position(self):
        return float(self.x[0]), float(self.x[1])

    def get_velocity(self):
        return float(self.x[2]), float(self.x[3])

    # ------------------------------------------------------------------
    # Stop / course-change detection - this is the "direction change" logic
    # ------------------------------------------------------------------

    def _record_motion(self, measured_xy):
        """
        Compute instantaneous speed/heading directly from the last two REAL
        measured positions (finite difference), not the Kalman-smoothed
        state. This is what makes stop/course-change detection react
        within a frame or two instead of lagging seconds behind, since it
        isn't filtered through the state covariance at all.
        """
        prev_xy = self.position_history[-1]
        dx = measured_xy[0] - prev_xy[0]
        dy = measured_xy[1] - prev_xy[1]
        speed = math.hypot(dx, dy) / self.dt   # pixels/second
        heading = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-3 else None

        self.position_history.append(measured_xy)
        self.position_history = self.position_history[-self.position_history_len:]

        self.speed_history.append(speed)
        if heading is not None:
            self.heading_history.append(heading)

        self.speed_history = self.speed_history[-self.history_len:]
        self.heading_history = self.heading_history[-self.history_len:]

    def is_stopped(self, speed_threshold_px_per_sec=25.0):
        """
        True if the target's recent speed has dropped near zero.
        Units are pixels/second (not pixels/frame) - this scales with your
        actual camera frame rate automatically since speed is computed as
        distance/dt. <-- CHANGE speed_threshold_px_per_sec to tune sensitivity
        """
        if len(self.speed_history) < 2:
            return False
        recent_avg_speed = sum(self.speed_history[-2:]) / 2
        return recent_avg_speed < speed_threshold_px_per_sec

    def has_changed_course(self, angle_threshold_deg=35.0):
        """
        True if the target's heading has swung by more than
        angle_threshold_deg between the start and end of the recent window.
        <-- CHANGE angle_threshold_deg to tune course-change sensitivity
        """
        if len(self.heading_history) < 4:
            return False

        old_heading = self.heading_history[0]
        new_heading = self.heading_history[-1]

        # Angular difference, wrapped to [-pi, pi]
        diff = math.atan2(math.sin(new_heading - old_heading), math.cos(new_heading - old_heading))
        return abs(math.degrees(diff)) > angle_threshold_deg

    def is_lost(self, max_missed_frames=15):
        """True if we haven't had a real detection in too long."""
        return self.frames_since_update > max_missed_frames
