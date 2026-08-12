import os
import sys
import time
import logging

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

from flask import Flask, render_template, request, jsonify, Response
from config import GCS_CONFIG, BASE_DIR
from gcs.telemetry_streamer import get_current_frame_jpeg, get_current_telemetry, set_quality_mode, get_active_quality_mode

logger = logging.getLogger(__name__)

def generate_mjpeg_stream():

    last_sent = None
    while True:
        jpeg_bytes = get_current_frame_jpeg()
        if jpeg_bytes and jpeg_bytes != last_sent:
            last_sent = jpeg_bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg_bytes)).encode('utf-8') + b'\r\n\r\n' +
                   jpeg_bytes + b'\r\n')
        time.sleep(0.005)

def create_app(mission_engine=None):
    template_folder = os.path.join(BASE_DIR, "gcs_web", "templates")
    app = Flask(__name__, template_folder=template_folder)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(generate_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/telemetry")
    def telemetry():
        return jsonify(get_current_telemetry())

    @app.route("/api/stream_quality", methods=["POST"])
    def api_stream_quality():
        data = request.get_json() or {}
        q_mode = data.get("quality", "balanced")
        res = set_quality_mode(q_mode)
        if res:
            return jsonify({"status": "ok", "message": f"Quality updated to {q_mode.upper()}", "profile": res})
        return jsonify({"status": "error", "message": f"Invalid quality mode: {q_mode}"}), 400

    @app.route("/api/command", methods=["POST"])
    def api_command():
        data = request.get_json() or {}
        text = data.get("text", "")
        if not text:
            return jsonify({"status": "error", "message": "Empty command"}), 400

        logger.info(f"GCS Web Received Command: '{text}'")
        if mission_engine:
            res = mission_engine.dispatch_command_text(text)
            return jsonify({"status": "ok", "message": f"Executed {res.get('action')}", "details": res})
        return jsonify({"status": "ok", "message": "Command logged (Engine Standby)"})

    @app.route("/api/voice_command", methods=["POST"])
    def api_voice_command():
        data = request.get_json() or {}
        phrase = data.get("phrase", "")
        if not phrase:
            return jsonify({"status": "error", "message": "Empty voice phrase"}), 400

        logger.info(f"GCS Web Received Voice Command Phrase: '{phrase}'")
        if mission_engine:
            res = mission_engine.dispatch_voice_phrase(phrase)
            return jsonify({"status": "ok", "message": f"Voice Action: {res.get('action')}", "details": res})
        return jsonify({"status": "ok", "message": "Voice phrase logged (Engine Standby)"})

    return app

def run_web_server(mission_engine=None, host=None, port=None):
    app = create_app(mission_engine)
    h = host or GCS_CONFIG.get("host", "0.0.0.0")
    p = port or GCS_CONFIG.get("web_port", 5000)
    app.run(host=h, port=p, debug=False, use_reloader=False)