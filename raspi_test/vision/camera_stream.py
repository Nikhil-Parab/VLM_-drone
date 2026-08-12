

import time
import threading
import logging
import cv2
import numpy as np
from config import CAMERA_CONFIG

logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, config=None):
        self.config = config or CAMERA_CONFIG
        self.width = self.config.get("width", 640)
        self.height = self.config.get("height", 480)
        self.fps = self.config.get("fps", 30)
        self.use_picamera2 = self.config.get("use_picamera2", True)

        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

        self.cap = None
        self.picam2 = None
        self.mode = "simulated"

        self._init_camera()

    def _init_camera(self):
        if self.use_picamera2:
            Picamera2 = None
            try:
                from picamera2 import Picamera2
            except ImportError:
                sys_dist_paths = [
                    "/usr/lib/python3/dist-packages",
                    "/usr/lib/python3.11/dist-packages",
                    "/usr/lib/python3.12/dist-packages",
                    "/usr/lib/python3.13/dist-packages",
                    "/usr/local/lib/python3/dist-packages",
                ]
                for p in sys_dist_paths:
                    if p not in sys.path and os.path.exists(p):
                        sys.path.append(p)
                try:
                    from picamera2 import Picamera2
                except ImportError:
                    Picamera2 = None

            if Picamera2 is not None:
                try:
                    logger.info("Initializing Picamera2 driver for Raspberry Pi Camera (IMX708)...")
                    self.picam2 = Picamera2()
                    try:
                        config = self.picam2.create_video_configuration(
                            main={"size": (self.width, self.height), "format": "RGB888"}
                        )
                    except Exception:
                        config = self.picam2.create_preview_configuration(
                            main={"size": (self.width, self.height), "format": "RGB888"}
                        )
                    self.picam2.configure(config)
                    self.picam2.start()
                    time.sleep(0.3)
                    test_arr = self.picam2.capture_array()
                    if test_arr is not None and test_arr.size > 0:
                        self.mode = "picamera2"
                        logger.info("Picamera2 started successfully in RGB888 true-color mode.")
                        return
                except Exception as e:
                    logger.warning(f"Picamera2 configuration/start failed: {e}. Falling back to OpenCV/V4L2 hardware search.")
            else:
                logger.warning("Picamera2 package not found in Python environment or system dist-packages.")

        # Candidate V4L2 video devices to check
        configured_dev = self.config.get("v4l2_device", 0)
        candidate_devs = [configured_dev]
        for d in [0, 1, 2, 4, 10, 11, -1]:
            if d not in candidate_devs:
                candidate_devs.append(d)

        for dev_idx in candidate_devs:
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(dev_idx, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, self.fps)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        # Warmup retry loop (allow camera hardware buffer time to warm up)
                        for _ in range(15):
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                self.cap = cap
                                self.mode = "v4l2"
                                logger.info(f"Physical camera initialized on /dev/video{dev_idx} (backend={backend})")
                                return
                            time.sleep(0.05)
                        cap.release()
                except Exception as e:
                    logger.debug(f"VideoCapture attempt failed on /dev/video{dev_idx} (backend={backend}): {e}")

        logger.warning("No active hardware camera (Picamera2 or OpenCV /dev/video*) detected. Falling back to Synthetic Simulated Camera Feed.")
        self.mode = "simulated"
        self._sim_t = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera stream thread started in [{self.mode}] mode.")

    def _update_loop(self):
        target_delay = 1.0 / self.fps
        while self.running:
            start_time = time.time()
            frame = self._capture_frame()
            if frame is not None:
                with self.lock:
                    self.frame = frame

            elapsed = time.time() - start_time
            sleep_time = target_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _capture_frame(self):
        if self.mode == "picamera2":
            try:
                array = self.picam2.capture_array()
                if array is not None:
                    if len(array.shape) == 3:
                        if array.shape[2] == 4:
                            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
                        elif array.shape[2] == 3:
                            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                    return array
            except Exception as e:
                logger.error(f"Picamera2 capture error: {e}")

        elif self.mode == "v4l2":
            if self.cap and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        return frame
                    # Retry once on transient dropped frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        return frame
                except Exception as e:
                    logger.error(f"V4L2 read error: {e}")
            
            # Return last valid frame if available instead of falling back to synthetic frame
            with self.lock:
                if self.frame is not None:
                    return self.frame

        return self._generate_simulated_frame()

    def _generate_simulated_frame(self):
        self._sim_t += 0.05
        frame = np.full((self.height, self.width, 3), (45, 90, 45), dtype=np.uint8)

        cv2.ellipse(frame, (int(self.width * 0.75), int(self.height * 0.35)),
                    (110, 70), 30, 0, 360, (180, 110, 30), -1)
        cv2.putText(frame, "LAKE (WATER BODY)", (int(self.width * 0.65), int(self.height * 0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        px = int(self.width * 0.3 + 120 * np.sin(self._sim_t * 0.8))
        py = int(self.height * 0.5 + 60 * np.cos(self._sim_t * 0.8))
        cv2.circle(frame, (px, py), 12, (200, 200, 200), -1)
        cv2.circle(frame, (px, py - 4), 6, (120, 150, 220), -1)
        cv2.putText(frame, "Person", (px - 20, py - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        rx = int(self.width * 0.5 + 40 * np.cos(self._sim_t * 0.5))
        ry = int(self.height * 0.7 + 30 * np.sin(self._sim_t * 0.5))
        cv2.rectangle(frame, (rx - 8, ry - 14), (rx + 8, ry + 14), (20, 20, 220), -1)
        cv2.rectangle(frame, (rx - 4, ry - 18), (rx + 4, ry - 14), (200, 200, 200), -1)
        cv2.putText(frame, "Red Bottle", (rx - 25, ry - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cx, cy = self.width // 2, self.height // 2
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

        return frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        logger.info("Camera stream stopped.")