import time
import math
import threading
import logging
import sys
import os

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

from config import FLIGHT_CONFIG

logger = logging.getLogger(__name__)



MAV_MODE_GUIDED = "GUIDED"
MAV_MODE_RTL = "RTL"
MAV_MODE_LAND = "LAND"
MAV_MODE_LOITER = "LOITER"
MAV_MODE_AUTO = "AUTO"

class MavlinkController:
    def __init__(self, config=None):
        self.config = config or FLIGHT_CONFIG
        self.connection_string = self.config.get("connection_string", "/dev/ttyAMA0")
        self.baud_rate = self.config.get("baud_rate", 57600)
        self.sim_mode = self.config.get("sim_mode", True)

        self.master = None
        self.connected = False
        self.simulated = False


        self.state = {
            "armed": False,
            "mode": "DISCONNECTED",
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "heading": 0.0,
            "battery_pct": 100,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "manual_override": False
        }




        self.pid_kp = self.config.get("pid_kp", 0.005)
        self.pid_ki = self.config.get("pid_ki", 0.0001)
        self.pid_kd = self.config.get("pid_kd", 0.001)
        self.max_vel = self.config.get("max_velocity_xy", 2.5)
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_pid_time = time.time()

        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.connect()

    def connect(self):
        try:
            from pymavlink import mavutil
            logger.info(f"Attempting MAVLink serial connection to Pixhawk on {self.connection_string} @ {self.baud_rate}...")
            self.master = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud_rate,
                autoreconnect=True,
                source_system=255,
                source_component=190
            )
            hb = self.master.wait_heartbeat(timeout=2.0)
            if hb:
                self.connected = True
                self.simulated = False
                self.state["mode"] = "INITIALIZED"
                logger.info(f"Connected to Pixhawk! (System {self.master.target_system}, Component {self.master.target_component})")
                return True
        except Exception as e:
            logger.warning(f"Could not connect to Pixhawk hardware ({e}).")

        if self.sim_mode:
            logger.info("Initializing High-Fidelity Pixhawk MAVLink Hardware Simulator...")
            self.connected = True
            self.simulated = True
            self.state["mode"] = "SIM_STANDBY"
            self.state["lat"] = 37.774929
            self.state["lon"] = -122.419416
            self.state["alt"] = 0.0
            return True
        return False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self.thread.start()
        logger.info("Mavlink controller background telemetry loop started.")

    def _telemetry_loop(self):
        while self.running:
            if self.simulated:
                self._update_simulated_telemetry()
            else:
                self._read_mavlink_messages()
            time.sleep(0.05)

    def _update_simulated_telemetry(self):
        with self.lock:
            if self.state["armed"] and self.state["mode"] == MAV_MODE_GUIDED:
                dt = 0.05
                deg_per_meter = 1.0 / 111000.0
                self.state["lat"] += self.state["vx"] * dt * deg_per_meter
                self.state["lon"] += self.state["vy"] * dt * deg_per_meter
                self.state["alt"] = max(0.0, self.state["alt"] + self.state["vz"] * dt)

    def _read_mavlink_messages(self):
        if not self.master:
            return
        try:
            msg = self.master.recv_match(blocking=False)
            if not msg:
                return

            msg_type = msg.get_type()
            with self.lock:
                if msg_type == "HEARTBEAT":
                    from pymavlink import mavutil
                    self.state["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)




                    mode_str = mavutil.mode_string_v10(msg)
                    if self.state["mode"] == MAV_MODE_GUIDED and mode_str not in [MAV_MODE_GUIDED, "GUIDED"]:
                        logger.warning(f"MANUAL RC OVERRIDE DETECTED! Mode changed from GCS to {mode_str}")
                        self.state["manual_override"] = True
                    self.state["mode"] = mode_str

                elif msg_type == "GLOBAL_POSITION_INT":
                    self.state["lat"] = msg.lat / 1e7
                    self.state["lon"] = msg.lon / 1e7
                    self.state["alt"] = msg.relative_alt / 1000.0
                    self.state["heading"] = msg.hdg / 100.0

                elif msg_type == "RC_CHANNELS":
                    rc_ch5 = getattr(msg, "chan5_raw", 1500)
                    if rc_ch5 > 1700:
                        if not self.state["manual_override"]:
                            logger.warning("R12 Receiver RC Switch engaged MANUAL OVERRIDE.")
                            self.state["manual_override"] = True
                    else:
                        self.state["manual_override"] = False

                elif msg_type == "SYS_STATUS":
                    self.state["battery_pct"] = msg.battery_remaining
        except Exception as e:
            logger.error(f"Error reading MAVLink packet: {e}")

    def set_mode(self, mode_name):
        logger.info(f"Requesting flight mode: {mode_name}")
        with self.lock:
            self.state["mode"] = mode_name
            self.state["manual_override"] = False

        if self.simulated or not self.master:
            return True

        try:
            mode_id = self.master.mode_mapping().get(mode_name)
            if mode_id is None:
                logger.error(f"Unknown mode name {mode_name}")
                return False
            self.master.set_mode(mode_id)
            return True
        except Exception as e:
            logger.error(f"Set mode failed: {e}")
            return False

    def arm(self):
        logger.info("Arming Pixhawk motors...")
        with self.lock:
            self.state["armed"] = True

        if self.simulated or not self.master:
            return True

        try:
            self.master.arducopter_arm()
            self.master.motors_armed_wait()
            return True
        except Exception as e:
            logger.error(f"Arm command failed: {e}")
            return False

    def disarm(self):
        logger.info("Disarming Pixhawk motors...")
        with self.lock:
            self.state["armed"] = False

        if self.simulated or not self.master:
            return True

        try:
            self.master.arducopter_disarm()
            return True
        except Exception as e:
            logger.error(f"Disarm command failed: {e}")
            return False

    def takeoff(self, altitude=10.0):
        logger.info(f"Executing GUIDED Takeoff to {altitude}m...")
        self.set_mode(MAV_MODE_GUIDED)
        self.arm()

        with self.lock:
            self.state["vz"] = 1.0

        if self.simulated or not self.master:
            def simulate_climb():
                while self.state["alt"] < altitude and self.state["armed"]:
                    time.sleep(0.2)
                    self.state["alt"] = min(altitude, self.state["alt"] + 0.5)
                self.state["vz"] = 0.0
                logger.info(f"Target altitude {altitude}m reached (Simulated).")
            threading.Thread(target=simulate_climb, daemon=True).start()
            return True

        try:
            self.master.user_takeoff(altitude=altitude)
            return True
        except Exception as e:
            logger.error(f"Takeoff command failed: {e}")
            return False




    def send_velocity_target(self, vx, vy, vz, yaw_rate=0.0):



        if self.state["manual_override"]:
            logger.info("Velocity command suppressed: MANUAL OVERRIDE IS ACTIVE.")
            return

        with self.lock:
            self.state["vx"] = vx
            self.state["vy"] = vy
            self.state["vz"] = vz

        if self.simulated or not self.master:
            return

        try:
            from pymavlink import mavutil
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            )
            self.master.mav.set_position_target_local_ned_send(
                0,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0, 0, 0,
                vx, vy, vz,
                0, 0, 0,
                0, yaw_rate
            )
        except Exception as e:
            logger.error(f"Error sending velocity vector: {e}")






    def track_target_offset(self, err_x, err_y):

        now = time.time()
        dt = max(0.01, now - self.last_pid_time)
        self.last_pid_time = now

        self.integral_x += err_x * dt
        self.integral_y += err_y * dt

        deriv_x = (err_x - self.prev_err_x) / dt
        deriv_y = (err_y - self.prev_err_y) / dt

        self.prev_err_x = err_x
        self.prev_err_y = err_y

        target_vx = -(self.pid_kp * err_y + self.pid_ki * self.integral_y + self.pid_kd * deriv_y) * 100.0
        target_vy = (self.pid_kp * err_x + self.pid_ki * self.integral_x + self.pid_kd * deriv_x) * 100.0

        target_vx = max(-self.max_vel, min(self.max_vel, target_vx))
        target_vy = max(-self.max_vel, min(self.max_vel, target_vy))

        self.send_velocity_target(target_vx, target_vy, 0.0)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Mavlink controller stopped.")