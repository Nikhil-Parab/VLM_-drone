import os
import sys


os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.append(user_site)

for sys_dist in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
    if sys_dist not in sys.path and os.path.exists(sys_dist):
        sys.path.append(sys_dist)

import time
import threading
import logging
import cv2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HexacopterEngine")

from config import CAMERA_CONFIG, VISION_CONFIG, FLIGHT_CONFIG, GCS_CONFIG, SMOLVLM_CONFIG
from vision import CameraStream, ObjectDetector, CentroidTracker, EnvironmentScanner
from flight import MavlinkController
from gcs import CommandParser, VoiceEngine, TelemetryStreamer
from gcs_web import run_web_server
from utils import get_system_health
from nikhilmodel import SmolVLMEngine

class MissionEngine:
    def __init__(self):
        logger.info("Initializing Hexacopter Companion Computer Systems...")


        self.camera = CameraStream(CAMERA_CONFIG)
        self.detector = ObjectDetector(VISION_CONFIG)
        self.tracker = CentroidTracker(max_disappeared=15, max_distance=80.0)
        self.scanner = EnvironmentScanner(VISION_CONFIG)


        self.drone = MavlinkController(FLIGHT_CONFIG)


        # --- SmolVLM Natural Language Command Engine -------------------------
        self.vlm_engine = None
        if SMOLVLM_CONFIG.get("enabled", False):
            model_dir = SMOLVLM_CONFIG["model_dir"]
            if os.path.isdir(model_dir):
                self.vlm_engine = SmolVLMEngine(
                    model_dir=model_dir,
                    device=SMOLVLM_CONFIG.get("device", "cpu"),
                    max_new_tokens=SMOLVLM_CONFIG.get("max_new_tokens", 128),
                )
                logger.info("SmolVLM NL Command Engine registered (lazy-load).")
            else:
                logger.warning(f"SmolVLM model dir not found: {model_dir}. VLM disabled.")

        self.command_parser = CommandParser(
            wake_word=GCS_CONFIG.get("wake_word", "jarvis"),
            vlm_engine=self.vlm_engine,
        )
        self.streamer = TelemetryStreamer(host=GCS_CONFIG["host"], port=GCS_CONFIG["video_port"])
        self.voice_engine = VoiceEngine(wake_word=GCS_CONFIG.get("wake_word", "jarvis"), command_callback=self.on_voice_command)


        self.mode = "STANDBY"
        self.search_target = None
        self.search_color = None
        self.search_text = None
        self.active_track_id = None


        self.latest_detections = []
        self.latest_scan_result = {"water_coverage_percent": 0.0, "overlay": None, "terrain_summary": "unknown"}
        self.inference_latency_ms = 0.0
        self.vision_lock = threading.Lock()

        self.running = False
        self.loop_thread = None
        self.async_vision_thread = None

    def start(self):
        logger.info("Starting Companion Computer Service Threads...")
        self.camera.start()
        self.drone.start()
        self.streamer.start()
        self.voice_engine.start()

        self.running = True


        self.async_vision_thread = threading.Thread(target=self._async_vision_worker, daemon=True)
        self.async_vision_thread.start()


        self.loop_thread = threading.Thread(target=self._main_processing_loop, daemon=True)
        self.loop_thread.start()


        logger.info(f"Starting GCS Web Dashboard on http://{GCS_CONFIG['host']}:{GCS_CONFIG['web_port']}...")
        run_web_server(mission_engine=self, host=GCS_CONFIG["host"], port=GCS_CONFIG["web_port"])

    def _async_vision_worker(self):
        logger.info(f"Asynchronous Vision Worker Thread operational using [{self.detector.backend}].")
        target_fps = VISION_CONFIG.get("max_inference_fps", 12.0)
        target_delay = 1.0 / target_fps

        while self.running:
            start_t = time.time()
            frame = self.camera.read()

            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()


            detections = self.detector.detect(
                frame,
                target_filter=self.search_target,
                color_filter=self.search_color,
                text_filter=self.search_text
            )


            scan_res = {"water_coverage_percent": 0.0, "overlay": None, "terrain_summary": "standby"}
            if self.mode == "SCAN_GEO":
                scan_res = self.scanner.scan_terrain(frame)

            latency = (time.time() - t0) * 1000.0

            with self.vision_lock:
                self.latest_detections = detections
                self.latest_scan_result = scan_res
                self.inference_latency_ms = latency

            elapsed = time.time() - start_t
            sleep_t = target_delay - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _draw_hud_corners(self, frame, x, y, bw, bh, color, length=16, thickness=3):


        cv2.line(frame, (x, y), (x + length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + length), color, thickness)

        cv2.line(frame, (x + bw, y), (x + bw - length, y), color, thickness)
        cv2.line(frame, (x + bw, y), (x + bw, y + length), color, thickness)

        cv2.line(frame, (x, y + bh), (x + length, y + bh), color, thickness)
        cv2.line(frame, (x, y + bh), (x, y + bh - length), color, thickness)

        cv2.line(frame, (x + bw, y + bh), (x + bw - length, y + bh), color, thickness)
        cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - length), color, thickness)

    def _main_processing_loop(self):
        logger.info("Main 30 FPS video streaming & flight control loop operational.")

        while self.running:
            start_t = time.time()
            frame = self.camera.read()

            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            processed_frame = frame.copy()


            with self.vision_lock:
                detections = list(self.latest_detections)
                scan_result = dict(self.latest_scan_result)
                inf_latency = self.inference_latency_ms


            if self.mode == "SCAN_GEO" and scan_result.get("overlay") is not None:
                processed_frame = scan_result["overlay"]


            tracked_objects = self.tracker.update(detections)





            for obj in tracked_objects:
                x, y, bw, bh = obj.box
                cx, cy = obj.center


                is_explicit_locked = getattr(obj, "is_target", False) or (self.active_track_id is not None and self.active_track_id == obj.track_id)

                if is_explicit_locked:

                    color = (0, 0, 255)
                    cv2.rectangle(processed_frame, (x, y), (x + bw, y + bh), color, 2)
                    self._draw_hud_corners(processed_frame, x, y, bw, bh, color, length=16, thickness=3)


                    cv2.circle(processed_frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.drawMarker(processed_frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)

                    label_str = f"LOCKED #{obj.track_id} {obj.label} ({obj.confidence:.2f})"
                    (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(processed_frame, (x, max(0, y - th - 8)), (x + tw + 6, max(th + 8, y)), (0, 0, 255), -1)
                    cv2.putText(processed_frame, label_str, (x + 3, max(th + 3, y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                else:

                    color = (0, 255, 0)
                    cv2.rectangle(processed_frame, (x, y), (x + bw, y + bh), color, 1)
                    label_str = f"#{obj.track_id} {obj.label} ({obj.confidence:.2f})"
                    cv2.putText(processed_frame, label_str, (x, max(15, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


            err_x, err_y, dist_px, active_obj = self.tracker.calculate_tracking_error(
                self.active_track_id, frame_width=w, frame_height=h
            )


            if active_obj and (getattr(active_obj, "is_target", False) or self.active_track_id is not None):
                cv2.line(processed_frame, (w // 2, h // 2), active_obj.center, (0, 255, 255), 2)
                cv2.circle(processed_frame, (w // 2, h // 2), 4, (0, 240, 255), -1)

                if self.mode in ["SEARCH", "TRACK"] and self.drone.state["armed"]:
                    self.drone.track_target_offset(err_x, err_y)


            health = get_system_health()


            self.streamer.update_frame(processed_frame)

            target_display = f"{self.search_color or ''} {self.search_target or 'ANY'}".strip()
            if self.search_text:
                target_display += f" [text: {self.search_text}]"

            speed_xy = (self.drone.state["vx"]**2 + self.drone.state["vy"]**2)**0.5
            self.streamer.update_telemetry({

                "mode": self.drone.state["mode"],
                "armed": self.drone.state["armed"],
                "lat": self.drone.state["lat"],
                "lon": self.drone.state["lon"],
                "alt": self.drone.state["alt"],
                "heading": self.drone.state["heading"],
                "speed": round(speed_xy, 1),
                "battery": self.drone.state["battery_pct"],
                "manual_override": self.drone.state["manual_override"],
                "active_target": active_obj.label if active_obj else "NONE",
                "search_target": target_display,
                "water_coverage": scan_result.get("water_coverage_percent", 0.0),


                "cpu_temp": health["cpu_temp"],
                "cpu_usage": health["cpu_usage"],
                "ram_used_mb": health["ram_used_mb"],
                "ram_total_gb": health["ram_total_gb"],
                "ram_pct": health["ram_pct"],
                "health_status": health["status"],
                "health_color": health["status_color"],
                "inference_latency": round(inf_latency, 1),
                "detector_backend": self.detector.backend
            })


            elapsed = time.time() - start_t
            sleep_t = (1.0 / 30.0) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def dispatch_command_text(self, text):
        cmd = self.command_parser.parse_command(text)
        return self.execute_mission_command(cmd)

    def on_voice_command(self, cmd):
        self.execute_mission_command(cmd)

    def dispatch_voice_phrase(self, phrase):
        cmd = self.voice_engine.process_voice_phrase(phrase)
        return cmd

    def execute_mission_command(self, cmd):
        action = cmd.get("action")
        params = cmd.get("params", {})
        logger.info(f"Executing Mission Directives: Action={action}, Params={params}")

        if action == "ARM":
            self.drone.arm()
        elif action == "DISARM":
            self.drone.disarm()
            self.mode = "STANDBY"
        elif action == "TAKEOFF":
            alt = params.get("altitude", 10.0)
            self.drone.takeoff(altitude=alt)
        elif action == "LAND":
            self.drone.set_mode("LAND")
        elif action == "RTL":
            self.drone.set_mode("RTL")
        elif action == "HOLD":
            self.drone.set_mode("LOITER")
            self.mode = "STANDBY"
        elif action in ["SEARCH", "TRACK"]:
            target_raw = params.get("target", None)
            color_raw = params.get("color", None)
            text_raw = params.get("text_query", None)
            target_id = params.get("target_id", None)

            self.search_target = target_raw.lower() if target_raw else None
            self.search_color = color_raw.lower() if color_raw else None
            self.search_text = text_raw.lower() if text_raw else None
            self.active_track_id = int(target_id) if str(target_id).isdigit() else None

            self.mode = action
            logger.info(f"Mission Mode [{action}]: target=[{self.search_target}], color=[{self.search_color}], text_query=[{self.search_text}], track_id=[{self.active_track_id}]")
        elif action == "SCAN_GEO":
            self.mode = "SCAN_GEO"
            logger.info("Mission Mode: SCAN_GEO Terrain & Water Body Scan Engaged.")
        elif action == "FLY_TO":
            lat = params.get("lat")
            lon = params.get("lon")
            alt = params.get("alt", 10.0)
            self.drone.send_velocity_target(1.0, 1.0, 0.0)

        return {"action": action, "params": params, "drone_mode": self.drone.state["mode"]}

    def stop(self):
        self.running = False
        self.camera.stop()
        self.drone.stop()
        self.streamer.stop()
        self.voice_engine.stop()
        logger.info("Companion computer shutdown complete.")

if __name__ == "__main__":
    engine = MissionEngine()
    try:
        engine.start()
    except KeyboardInterrupt:
        engine.stop()