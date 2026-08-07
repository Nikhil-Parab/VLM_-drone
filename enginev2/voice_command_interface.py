"""
voice_command_interface.py
===========================
Natural-language voice / text command interface for the drone engine.

Two input modes:
  1. Microphone  (default) — uses SpeechRecognition + Google Web Speech API.
                             For offline Pi deployment, swap to Vosk (see VOICE_BACKEND below).
  2. Text stdin  (--text "command string") — useful for testing without a mic.

Pipeline:
  Mic / stdin text
       ↓
  CommandParser  [rule-based: regex + keyword matching + COCO class mapping]
       ↓
  UDP socket → host:port 5000  (drone_tracking_engine.py listens here)
       ↓
  pyttsx3 TTS  (reads back the parsed command so the pilot hears confirmation)

NOTE on VLM for command parsing:
  SmolVLM-256M is too small to reliably produce structured JSON from free-form
  text — it tends to echo instructions back rather than parse them. Rule-based
  parsing is faster, 100% offline, deterministic, and perfectly adequate for
  the finite set of drone commands. SmolVLM stays in the pipeline for vision
  tasks (re-ID, small-object classification) where it actually excels.

Run standalone:
    # Text mode (no mic needed):
    python voice_command_interface.py --text "follow the mobile phone"
    python voice_command_interface.py --text "scan the area"
    python voice_command_interface.py --text "return home"

    # Mic mode (continuous loop):
    python voice_command_interface.py

    # Mic mode, targeting a remote Pi:
    python voice_command_interface.py --host 192.168.1.42
"""

import argparse
import json
import re
import socket
import sys
import threading
import time
from typing import Optional

# ============================================================
# CONFIGURATION  <-- change these to match your deployment
# ============================================================

COMMAND_UDP_HOST   = "127.0.0.1"   # change to Pi IP for remote control
COMMAND_UDP_PORT   = 5000          # must match drone_tracking_engine.py
TTS_ENABLED        = True          # set False if no speaker
VOICE_TIMEOUT_SEC  = 5             # seconds to wait for speech input
VOICE_BACKEND      = "google"      # "google" (online) | "vosk" (offline)
VOSK_MODEL_PATH    = "vosk-model-small-en-us-0.15"  # only for VOICE_BACKEND="vosk"

# Valid mission actions
VALID_ACTIONS = {
    "follow", "track", "hover", "scan", "survey",
    "land", "rtl", "stop", "search", "return",
}

# Phase H: safety-critical actions that require verbal/text confirmation before executing
CONFIRM_BEFORE_EXECUTE = {"rtl", "land"}

# Phrase → canonical action  (longest match wins)
ACTION_SYNONYMS = {
    "return home":  "rtl",
    "go back":      "rtl",
    "come back":    "rtl",
    "get back":     "rtl",
    "don't move":   "hover",
    "do not move":  "hover",
    "stay still":   "hover",
    "keep still":   "hover",
    "hold position":"hover",
    "look for":     "search",
    "search for":   "search",
    "find the":     "search",
    "find a":       "search",
    "orbit":        "scan",
    "circle":       "scan",
    "survey":       "survey",
    "chase":        "follow",
    "pursue":       "follow",
    "track":        "track",
    "follow":       "follow",
    "hover":        "hover",
    "stay":         "hover",
    "hold":         "hover",
    "stop":         "stop",
    "land":         "land",
    "scan":         "scan",
    "search":       "search",
    "return":       "rtl",
    "rtl":          "rtl",
}

# Natural language → YOLO COCO class name
# Add any object you want the drone to track here.
COCO_CLASS_MAP = {
    # People
    "person":       "person",
    "human":        "person",
    "man":          "person",
    "woman":        "person",
    "people":       "person",
    "pedestrian":   "person",
    # Electronics
    "mobile phone": "cell phone",
    "mobile":       "cell phone",
    "phone":        "cell phone",
    "smartphone":   "cell phone",
    "cell phone":   "cell phone",
    "cellphone":    "cell phone",
    "laptop":       "laptop",
    "computer":     "laptop",
    "notebook":     "laptop",
    "keyboard":     "keyboard",
    "mouse":        "mouse",
    "remote":       "remote",
    "tv":           "tv",
    "television":   "tv",
    "monitor":      "tv",
    # Vehicles
    "car":          "car",
    "automobile":   "car",
    "vehicle":      "car",
    "truck":        "truck",
    "bus":          "bus",
    "bicycle":      "bicycle",
    "bike":         "bicycle",
    "cycle":        "bicycle",
    "motorcycle":   "motorcycle",
    "motorbike":    "motorcycle",
    "scooter":      "motorcycle",
    "airplane":     "airplane",
    "plane":        "airplane",
    "boat":         "boat",
    # Animals
    "dog":          "dog",
    "cat":          "cat",
    "bird":         "bird",
    "horse":        "horse",
    "cow":          "cow",
    # Objects
    "backpack":     "backpack",
    "bag":          "backpack",
    "rucksack":     "backpack",
    "umbrella":     "umbrella",
    "handbag":      "handbag",
    "suitcase":     "suitcase",
    "bottle":       "bottle",
    "cup":          "cup",
    "chair":        "chair",
    "book":         "book",
    "clock":        "clock",
    "vase":         "vase",
    "scissors":     "scissors",
    "ball":         "sports ball",
    "sports ball":  "sports ball",
    "frisbee":      "frisbee",
    "kite":         "kite",
}

# Multi-slot direction tokens (Phase 6 — compound commands)
DIRECTION_TOKENS = {
    "left": "left", "right": "right", "up": "up", "down": "down",
    "forward": "forward", "backward": "backward", "back": "backward",
    "north": "forward", "south": "backward", "east": "right", "west": "left",
}

SPEED_TOKENS = {
    "slowly": "slow", "slow": "slow", "gentle": "slow", "gently": "slow",
    "quickly": "fast", "fast": "fast", "quick": "fast", "aggressive": "fast",
    "medium": "medium", "moderate": "medium", "normal": "medium",
}

ALTITUDE_PATTERNS = [
    r"(?:to|at|reach|climb to|fly at)\s+(\d+(?:\.\d+)?)\s*(?:m|meters|metres|meter)",
    r"height\s+(\d+(?:\.\d+)?)",
    r"altitude\s+(\d+(?:\.\d+)?)",
]
# Tried in order; first match wins
TARGET_PATTERNS = [
    r"(?:follow|track|chase|pursue|find|locate|watch)\s+(?:the\s+|a\s+|an\s+)?(.+)",
    r"(?:search|look)\s+for\s+(?:the\s+|a\s+|an\s+)?(.+)",
    r"(?:scan|survey|orbit|circle)\s+(?:the\s+|a\s+)?(.+)",
]


# ============================================================
# Helpers
# ============================================================

def _extract_directions(text: str) -> list:
    """Parse compound direction tokens from one utterance."""
    text = text.lower()
    found = []
    for token, canonical in sorted(DIRECTION_TOKENS.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(token)}\b", text) and canonical not in found:
            found.append(canonical)
    return found


def _extract_speed(text: str) -> str:
    text = text.lower()
    for token, canonical in sorted(SPEED_TOKENS.items(), key=lambda x: -len(x[0])):
        if token in text:
            return canonical
    return "medium"


def _extract_altitude_m(text: str) -> Optional[float]:
    text = text.lower()
    for pat in ALTITUDE_PATTERNS:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _extract_target(text: str) -> Optional[str]:
    """
    Pull the target object out of a natural-language command, e.g.:
      "follow the mobile phone" → "mobile phone"
      "track a red car"         → "red car"
    Returns None if no target is found.
    """
    text = text.lower().strip()
    for pattern in TARGET_PATTERNS:
        m = re.search(pattern, text)
        if m:
            raw = m.group(1).strip().rstrip(". !?,;")
            # strip filler words at the end
            raw = re.sub(r"\s+(please|now|immediately|quickly|fast)$", "", raw)
            return raw if raw else None
    return None


def _normalise_coco(target: Optional[str]) -> Optional[str]:
    """
    Map a natural-language object description to the closest YOLO COCO class.
    Tries longest-key match first so "mobile phone" beats "phone".
    Falls back to Levenshtein fuzzy match (Phase H) to handle STT typos
    like \"selfone\" → \"cell phone\".
    Returns the original target string if no mapping found.
    """
    if not target:
        return None
    t = target.lower().strip()
    # Exact / substring match (longest key wins)
    for key in sorted(COCO_CLASS_MAP, key=len, reverse=True):
        if key in t:
            return COCO_CLASS_MAP[key]
    # Phase H: fuzzy fallback — find nearest COCO key by edit distance
    best_key, best_dist = _fuzzy_coco_match(t)
    if best_key is not None:
        print(f"[CMD] Fuzzy COCO match: '{t}' -> '{best_key}' (dist={best_dist})")
        return COCO_CLASS_MAP[best_key]
    return target   # pass through as-is; let YOLO decide


def _fuzzy_coco_match(target: str, threshold: int = 4) -> tuple:
    """
    Phase H: Pure-Python Levenshtein edit-distance fuzzy match against
    COCO_CLASS_MAP keys. Returns (best_key, distance) or (None, inf).
    threshold controls max allowed edit distance.
    """
    def _lev(a: str, b: str) -> int:
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                                prev[j] + (0 if ca == cb else 1)))
            prev = curr
        return prev[-1]

    best_key, best_dist = None, float("inf")
    # Compare against each word in multi-word keys too
    for key in COCO_CLASS_MAP:
        # Try the whole key and also individual words
        candidates_to_check = [key] + key.split()
        for candidate in candidates_to_check:
            d = _lev(target, candidate)
            if d < best_dist:
                best_dist, best_key = d, key
    if best_dist <= threshold:
        return best_key, best_dist
    return None, float("inf")


def _say(text: str):
    """Non-blocking TTS via pyttsx3 (fire-and-forget thread)."""
    if not TTS_ENABLED:
        return
    try:
        import pyttsx3

        def _run():
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()

        threading.Thread(target=_run, daemon=True).start()
    except Exception as exc:
        print(f"[TTS] Warning: {exc}")


# ============================================================
# Rule-based command parser  (no model — fast, deterministic, offline)
# ============================================================

class CommandParser:
    """
    Parses free-form voice / text commands into structured dicts using
    regex + keyword matching + COCO class normalisation.

    Why not SmolVLM here?
    SmolVLM-256M is too small to reliably produce structured JSON —
    it echoes instructions back rather than parsing them. Rule-based
    parsing is instant, fully offline, and handles all realistic drone
    commands without a model. SmolVLM stays in the pipeline for vision
    tasks (re-ID, small-object classification) where it actually excels.
    """

    def parse(self, user_text: str) -> dict:
        """
        Parse a natural-language command into:
          {"action": str, "target": str|None, "coco_class": str|None, "params": {}}

        "coco_class" is the mapped YOLO class name (e.g. "cell phone") so
        drone_tracking_engine.py can immediately switch the detection target.
        """
        text = user_text.lower().strip()

        # 1. Action: longest synonym match first
        action = "hover"   # safe default
        for phrase, canonical in sorted(ACTION_SYNONYMS.items(), key=lambda x: -len(x[0])):
            if phrase in text:
                action = canonical
                break

        # 2. Target object extraction
        raw_target = _extract_target(text)
        coco_class = _normalise_coco(raw_target)
        directions = _extract_directions(text)
        speed = _extract_speed(text)
        target_altitude_m = _extract_altitude_m(text)

        # Open-vocabulary grounding phrase: full target text for Grounding DINO
        grounding_phrase = raw_target or coco_class

        cmd = {
            "action":     action,
            "target":     raw_target,
            "coco_class": coco_class,
            "grounding_phrase": grounding_phrase,
            "directions": directions,
            "speed":      speed,
            "target_altitude_m": target_altitude_m,
            "params":     {},
        }
        return cmd


# ============================================================
# UDP sender
# ============================================================

class CommandSender:
    """Sends parsed command dicts to the drone engine via UDP."""

    def __init__(self, host: str = COMMAND_UDP_HOST, port: int = COMMAND_UDP_PORT):
        self.addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: dict):
        payload = json.dumps(command).encode("utf-8")
        self._sock.sendto(payload, self.addr)
        print(f"[CommandSender] → {self.addr}  payload={payload.decode()}")

    def close(self):
        self._sock.close()


# ============================================================
# Voice listener
# ============================================================

def listen_once(timeout_sec: int = VOICE_TIMEOUT_SEC) -> Optional[str]:
    """
    Block until speech is detected and transcribed, or timeout expires.
    Returns transcribed string, or None on silence/error.
    """
    if VOICE_BACKEND == "vosk":
        return _listen_vosk(timeout_sec)
    return _listen_google(timeout_sec)


def _listen_google(timeout_sec: int) -> Optional[str]:
    try:
        import speech_recognition as sr
    except ImportError:
        print("[Voice] SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio")
        return None

    r   = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.3)
        print("[Voice] Listening ... (Google backend)")
        try:
            audio = r.listen(source, timeout=timeout_sec, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            return None

    try:
        text = r.recognize_google(audio)
        print(f"[Voice] Heard: '{text}'")
        return text
    except sr.UnknownValueError:
        print("[Voice] Could not understand audio.")
        return None
    except sr.RequestError as exc:
        print(f"[Voice] Google API error: {exc}")
        return None


def _listen_vosk(timeout_sec: int) -> Optional[str]:
    """Offline speech recognition via Vosk — requires 'vosk' package + model download."""
    try:
        import json as _json
        import queue as _queue

        import sounddevice as sd
        from vosk import KaldiRecognizer, Model

        model = Model(VOSK_MODEL_PATH)
        rec   = KaldiRecognizer(model, 16000)
        q: _queue.Queue = _queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(bytes(indata))

        deadline = time.time() + timeout_sec
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                                channels=1, callback=callback):
            print("[Voice] Listening ... (Vosk offline backend)")
            while time.time() < deadline:
                data = q.get(timeout=1)
                if rec.AcceptWaveform(data):
                    result = _json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"[Voice] Heard: '{text}'")
                        return text
    except Exception as exc:
        print(f"[Voice] Vosk error: {exc}")
    return None


# ============================================================
# Main loop
# ============================================================

def run(host: str, port: int, text_command: Optional[str] = None):
    parser = CommandParser()
    sender = CommandSender(host=host, port=port)

    def _process(raw_text: str):
        print(f"\n[CMD] Raw input: '{raw_text}'")
        cmd = parser.parse(raw_text)
        print(f"[CMD] Parsed  : {json.dumps(cmd, indent=2)}")

        # Phase H: safety gate for RTL / land — require explicit confirmation
        if cmd["action"] in CONFIRM_BEFORE_EXECUTE:
            confirm_prompt = f"Confirm: {cmd['action']}? Say 'yes' or type 'yes' to execute."
            print(f"[CMD] ⚠ Safety confirm required: {confirm_prompt}")
            _say(confirm_prompt)
            if text_command:
                # Text mode: prompt inline
                try:
                    confirm_input = input("Confirm (yes/no): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm_input = ""
            else:
                # Mic mode: listen for a short confirmation utterance
                confirm_input = (listen_once(timeout_sec=6) or "").lower()
            if confirm_input not in ("yes", "y", "confirm", "affirmative"):
                print(f"[CMD] ✗ Action '{cmd['action']}' cancelled (no confirmation).")
                _say(f"{cmd['action']} cancelled.")
                return None
            print(f"[CMD] ✓ Confirmed. Executing '{cmd['action']}'.")

        sender.send(cmd)
        tts_msg = f"Command received: {cmd['action']}"
        if cmd.get("target"):
            tts_msg += f", target {cmd['target']}"
        _say(tts_msg)
        return cmd

    if text_command:
        # One-shot text mode
        _process(text_command)
        time.sleep(1)   # give TTS thread time to finish
        sender.close()
        return

    # Continuous mic mode
    print("=" * 55)
    print("Drone Voice Command Interface")
    print(f"  Sending commands to {host}:{port} via UDP")
    print(f"  Voice backend : {VOICE_BACKEND}")
    print("  Press Ctrl+C to quit.")
    print("=" * 55)
    _say("Drone voice interface ready.")

    try:
        while True:
            transcript = listen_once()
            if transcript:
                _process(transcript)
            else:
                print("[Voice] No input detected, listening again ...")
    except KeyboardInterrupt:
        print("\n[CMD] Shutting down.")
    finally:
        sender.close()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Drone voice/text command interface")
    ap.add_argument("--text", type=str, default=None,
                    help="Send a single text command instead of using the microphone")
    ap.add_argument("--host", type=str, default=COMMAND_UDP_HOST,
                    help=f"Drone engine UDP host (default: {COMMAND_UDP_HOST})")
    ap.add_argument("--port", type=int, default=COMMAND_UDP_PORT,
                    help=f"Drone engine UDP port (default: {COMMAND_UDP_PORT})")
    args = ap.parse_args()

    run(host=args.host, port=args.port, text_command=args.text)
