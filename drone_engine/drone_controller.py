"""
drone_controller.py
=====================
Abstract drone controller interface, with two implementations:

1. Sim2DDroneController — a small pygame 2D simulator. Runs standalone,
   no external installs beyond pygame. Good for testing the tracking
   engine's DECISION logic (move/hold/search/scan commands) end-to-end today.

2. MAVSDKDroneController — a REFERENCE STUB for wiring into a real PX4 SITL
   simulation (e.g. Gazebo+PX4) via MAVSDK-Python. Structurally correct but
   untested without an actual SITL environment — see class docstring.

Phase 5 additions (voice command integration):
  • scan_waypoints(waypoints)  — cycle through a list of (x, y) positions
    (sim) or NED waypoints (MAVSDK stub); used by 'scan' / 'survey' commands.
  • return_to_launch()         — RTL action; sim recentres, MAVSDK calls RTL.

pip install: pygame  (for Sim2DDroneController)
pip install: mavsdk  (only if/when you use MAVSDKDroneController)
"""

import math
import threading
import time


class DroneController:
    """Abstract interface every backend implements."""

    def move_toward(self, dx, dy):
        """Command the drone to move in the direction of vector (dx, dy)."""
        raise NotImplementedError

    def hold_position(self):
        """Command the drone to hover / stay in place."""
        raise NotImplementedError

    def search_pattern(self):
        """Command the drone to execute a search maneuver (target lost)."""
        raise NotImplementedError

    def scan_waypoints(self, waypoints):
        """Cycle through a list of (x, y) waypoints for a scan / survey mission."""
        raise NotImplementedError

    def return_to_launch(self):
        """Command the drone to return to its launch / home position."""
        raise NotImplementedError

    def get_state(self):
        """Return a dict describing current position/heading, for display."""
        raise NotImplementedError


# ============================================================
# BACKEND 1: pygame 2D simulator (runnable now, no drone needed)
# ============================================================

class Sim2DDroneController(DroneController):
    """
    A minimal top-down 2D drone simulator. The "drone" is a dot that moves
    toward commanded directions with simple acceleration/drag, so movement
    looks physically plausible without a full flight-dynamics model.

    Run this file directly (python drone_controller.py) to see a demo with
    a synthetic moving target, independent of any camera/vision code - this
    lets you validate the Kalman + decision logic in isolation.
    """

    def __init__(self, world_size=(800, 600), max_speed=4.0):
        self.world_w, self.world_h = world_size
        self.pos = [self.world_w / 2, self.world_h / 2]
        self.vel = [0.0, 0.0]
        self.max_speed = max_speed          # <-- CHANGE drone max speed here
        self.accel = 0.5                    # <-- CHANGE responsiveness here
        self.drag = 0.90                    # <-- CHANGE momentum/drag here
        self.status_text = "idle"
        self.lock = threading.Lock()
        self._scan_waypoints = []           # list of (x,y) for scan/survey missions
        self._scan_index = 0

    def move_toward(self, dx, dy):
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return self.hold_position()
        ux, uy = dx / dist, dy / dist
        with self.lock:
            self.vel[0] += ux * self.accel
            self.vel[1] += uy * self.accel
            self.status_text = "following target"

    def hold_position(self):
        with self.lock:
            self.vel[0] *= 0.5
            self.vel[1] *= 0.5
            self.status_text = "holding position"

    def search_pattern(self):
        # Simple rotating search sweep when target is lost
        t = time.time()
        with self.lock:
            self.vel[0] = math.cos(t * 2) * self.max_speed * 0.5
            self.vel[1] = math.sin(t * 2) * self.max_speed * 0.5
            self.status_text = "searching for target"

    def scan_waypoints(self, waypoints):
        """
        Begin cycling through a list of (x, y) world-space waypoints.
        The drone moves toward each waypoint in order, looping back to
        the start once the last is reached.

        Parameters
        ----------
        waypoints : list of (x, y) tuples in sim pixel coords
        """
        if not waypoints:
            return
        with self.lock:
            self._scan_waypoints = list(waypoints)
            self._scan_index = 0
            self.status_text = "scanning waypoints"

    def _advance_scan(self):
        """Internal: move toward the current scan waypoint; advance when close."""
        if not self._scan_waypoints:
            return
        wp = self._scan_waypoints[self._scan_index]
        dx = wp[0] - self.pos[0]
        dy = wp[1] - self.pos[1]
        if math.hypot(dx, dy) < 8.0:   # arrived — move to next waypoint
            self._scan_index = (self._scan_index + 1) % len(self._scan_waypoints)
        self.vel[0] += (dx / max(math.hypot(dx, dy), 1e-3)) * self.accel
        self.vel[1] += (dy / max(math.hypot(dx, dy), 1e-3)) * self.accel

    def return_to_launch(self):
        """Sim: fly back toward the centre of the world (launch point)."""
        cx, cy = self.world_w / 2, self.world_h / 2
        with self.lock:
            dx = cx - self.pos[0]
            dy = cy - self.pos[1]
            self.vel[0] += (dx / max(math.hypot(dx, dy), 1e-3)) * self.accel * 2
            self.vel[1] += (dy / max(math.hypot(dx, dy), 1e-3)) * self.accel * 2
            self.status_text = "returning to launch"

    def step_physics(self):
        """Call once per simulation tick to advance the drone's position."""
        with self.lock:
            if self.status_text == "scanning waypoints":
                self._advance_scan()
            speed = math.hypot(*self.vel)
            if speed > self.max_speed:
                scale = self.max_speed / speed
                self.vel[0] *= scale
                self.vel[1] *= scale
            self.vel[0] *= self.drag
            self.vel[1] *= self.drag
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.pos[0] = max(0, min(self.world_w, self.pos[0]))
            self.pos[1] = max(0, min(self.world_h, self.pos[1]))

    def get_state(self):
        with self.lock:
            return {"position": tuple(self.pos), "velocity": tuple(self.vel), "status": self.status_text}


def run_sim2d_demo():
    """
    Standalone demo: a synthetic target moves around (occasionally stopping
    or changing course), the Kalman tracker + decision logic tracks it, and
    the pygame simulator drone follows - all without needing a camera.
    This validates the CONTROL/DECISION half of the pipeline independently
    of the VISION half.
    """
    import random
    import pygame
    from kalman_tracker import KalmanTrack

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Drone Tracking Sim - synthetic target demo (no camera)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    drone = Sim2DDroneController(world_size=(800, 600))

    # Synthetic target: moves in straight segments, randomly changes
    # course or stops every few seconds - this is what the Kalman filter
    # and decision layer are being tested against.
    target_pos = [200.0, 300.0]
    target_vel = [3.0, 0.0]
    next_behavior_change = time.time() + 2.0

    tracker = KalmanTrack(initial_xy=tuple(target_pos), dt=1 / 30.0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ---- Update synthetic target motion ----
        now = time.time()
        if now > next_behavior_change:
            behavior = random.choice(["straight", "turn", "stop"])
            if behavior == "turn":
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(2, 4)
                target_vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            elif behavior == "stop":
                target_vel = [0.0, 0.0]
            next_behavior_change = now + random.uniform(2.0, 4.0)

        target_pos[0] = max(20, min(780, target_pos[0] + target_vel[0]))
        target_pos[1] = max(20, min(580, target_pos[1] + target_vel[1]))

        # ---- Kalman tracker sees the target's position (simulating a detector) ----
        tracker.predict()
        tracker.update(tuple(target_pos))

        # ---- Decision layer ----
        if tracker.is_stopped():
            drone.hold_position()
        elif tracker.is_lost():
            drone.search_pattern()
        else:
            tx, ty = tracker.get_position()
            dx, dy = tracker.get_velocity()
            # Lead the target slightly using its velocity - chase ahead, not behind
            aim_x, aim_y = tx + dx * 5, ty + dy * 5
            drone_pos = drone.get_state()["position"]
            drone.move_toward(aim_x - drone_pos[0], aim_y - drone_pos[1])

        drone.step_physics()

        # ---- Draw ----
        screen.fill((30, 30, 35))
        pygame.draw.circle(screen, (220, 80, 80), (int(target_pos[0]), int(target_pos[1])), 10)
        state = drone.get_state()
        pygame.draw.circle(screen, (80, 200, 120), (int(state["position"][0]), int(state["position"][1])), 8)

        course_changed = tracker.has_changed_course()
        info_lines = [
            f"Drone status: {state['status']}",
            f"Target stopped: {tracker.is_stopped()}",
            f"Course changed: {course_changed}",
        ]
        for i, line in enumerate(info_lines):
            color = (255, 210, 90) if "True" in line else (200, 200, 200)
            screen.blit(font.render(line, True, color), (12, 12 + i * 24))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ============================================================
# BACKEND 2: MAVSDK / PX4 SITL reference stub (NOT fully tested here)
# ============================================================

class MAVSDKDroneController(DroneController):
    """
    Reference stub for real simulated (or real) flight via MAVSDK-Python
    talking to PX4 SITL (e.g. running under Gazebo).

    IMPORTANT: MAVSDK-Python is async (asyncio-based). This class exposes
    the same sync-looking methods as the other controller for interface
    consistency, but internally it needs an asyncio event loop running.
    The cleanest way to bridge this with the rest of the (synchronous)
    engine is to run the asyncio loop in a background thread and post
    commands to it - sketched below. Treat this as a working structure to
    build from, not a drop-in tested implementation - you'll need an
    actual PX4 SITL instance (see the Gazebo+PX4 setup mentioned earlier)
    to validate it.

    pip install mavsdk
    """

    def __init__(self, system_address="udpin://0.0.0.0:14540"):
        self.system_address = system_address  # <-- CHANGE to match your SITL connection
        self._loop = None
        self._thread = None
        self._drone = None
        self._ready = threading.Event()
        self._start_background_loop()

    def _start_background_loop(self):
        import asyncio

        def runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._connect())
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    async def _connect(self):
        from mavsdk import System
        self._drone = System()
        await self._drone.connect(system_address=self.system_address)
        print("[MAVSDK] Waiting for drone connection...")
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                print("[MAVSDK] Connected.")
                break
        # Standard PX4 offboard-mode setup - arm, then start offboard control
        await self._drone.action.arm()
        from mavsdk.offboard import VelocityNedYaw
        await self._drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        await self._drone.offboard.start()

    def _run_coro(self, coro):
        import asyncio
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def move_toward(self, dx, dy, speed=2.0):
        """
        Translate a 2D image-space direction into a NED velocity command.
        NOTE: dx/dy here are pixel-space directions from the vision layer -
        in a real deployment you'd convert these into a proper world-frame
        heading (e.g. via the drone's current yaw + a gimbal/camera model),
        not feed pixel deltas directly into NED velocities. This is
        simplified on purpose to show the MAVSDK call shape.
        """
        from mavsdk.offboard import VelocityNedYaw
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return self.hold_position()
        vn = (dy / dist) * speed   # north component
        ve = (dx / dist) * speed   # east component
        self._run_coro(self._drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, 0, 0)))

    def hold_position(self):
        from mavsdk.offboard import VelocityNedYaw
        self._run_coro(self._drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0)))

    def search_pattern(self):
        from mavsdk.offboard import VelocityNedYaw
        # Simple yaw-rotation search — replace with a real sweep pattern as needed
        self._run_coro(self._drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 30)))

    def scan_waypoints(self, waypoints):
        """
        Fly through a list of NED waypoints for a scan / survey mission.
        waypoints: list of (north_m, east_m) offsets from home, at current altitude.
        NOTE: This is a structural stub — wire up proper MissionItem sequences
        via self._drone.mission for a production implementation.
        """
        from mavsdk.offboard import VelocityNedYaw
        # For now, command velocity toward the first waypoint as a placeholder.
        # Replace with self._drone.mission.upload_mission() for full waypoint nav.
        if waypoints:
            n, e = waypoints[0]
            self._run_coro(
                self._drone.offboard.set_velocity_ned(VelocityNedYaw(n * 0.5, e * 0.5, 0, 0))
            )

    def return_to_launch(self):
        """Command PX4 to return to launch position."""
        self._run_coro(self._drone.action.return_to_launch())

    def get_state(self):
        # A real implementation would read self._drone.telemetry.position()
        # via another background coroutine that caches the latest value.
        return {"position": None, "velocity": None, "status": "mavsdk backend (see telemetry stream)"}


if __name__ == "__main__":
    run_sim2d_demo()
