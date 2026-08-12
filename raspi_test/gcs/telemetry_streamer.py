import time
import json
import threading
import logging
import cv2
from config import GCS_CONFIG, QUALITY_PROFILES

logger = logging.getLogger(__name__)

_current_jpeg_bytes = None
_current_telemetry_data = {}
_active_quality_mode = GCS_CONFIG.get("default_quality_mode", "balanced")
_stream_lock = threading.Lock()

def get_current_frame_jpeg():
    with _stream_lock:
        return _current_jpeg_bytes

def get_current_telemetry():
    with _stream_lock:
        return dict(_current_telemetry_data)

def set_quality_mode(mode_name):
    global _active_quality_mode
    mode_name = mode_name.lower().strip()
    if mode_name in QUALITY_PROFILES:
        with _stream_lock:
            _active_quality_mode = mode_name
        logger.info(f"Dynamic Video Quality switched to [{mode_name.upper()}]: {QUALITY_PROFILES[mode_name]['label']}")
        return QUALITY_PROFILES[mode_name]
    return None

def get_active_quality_mode():
    with _stream_lock:
        return _active_quality_mode

class TelemetryStreamer:
    def __init__(self, host=None, port=None):
        self.host = host or GCS_CONFIG.get("host", "0.0.0.0")
        self.port = port or GCS_CONFIG.get("video_port", 8080)
        self.running = False

    def start(self):
        self.running = True
        logger.info("Video & Telemetry Engine initialized with Dynamic Quality Support.")

    def update_frame(self, frame):
        global _current_jpeg_bytes
        if frame is None:
            return

        try:
            with _stream_lock:
                q_mode = _active_quality_mode

            profile = QUALITY_PROFILES.get(q_mode, QUALITY_PROFILES["balanced"])
            target_w = profile["width"]
            target_h = profile["height"]
            target_q = profile["quality"]

            h, w = frame.shape[:2]
            if w != target_w or h != target_h:
                frame_out = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST if q_mode == "low" else cv2.INTER_LINEAR)
            else:
                frame_out = frame

            ret, jpeg = cv2.imencode(".jpg", frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), target_q])
            if ret:
                jpeg_data = jpeg.tobytes()
                with _stream_lock:
                    _current_jpeg_bytes = jpeg_data
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")

    def update_telemetry(self, telemetry_dict):
        global _current_telemetry_data
        with _stream_lock:
            telemetry_dict["quality_mode"] = _active_quality_mode.upper()
            _current_telemetry_data = telemetry_dict

    def stop(self):
        self.running = False
        logger.info("Telemetry streamer stopped.")