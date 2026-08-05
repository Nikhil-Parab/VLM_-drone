# Full Project & Chat Context Summary

## Overview & Architecture
This project is an AI-powered autonomous drone tracking engine running on a Raspberry Pi / laptop with Pixhawk flight controller support.

It uses a multi-stage vision and language pipeline:
1. **Detection:** YOLOv8n for real-time target detection (21+ FPS).
2. **Tracking:** Constant-velocity Kalman Filter (`KalmanTrack`) predicting position and velocity across missed frames.
3. **Small-Object VLM Classifier:** `SmolObjectClassifier` running SmolVLM-256M-Instruct on low-confidence or small crops (<64x64 px) to generate fine-grained semantic descriptions.
4. **Re-Identification:** `ReIDEngine` generating compact PSTG-style `<REID>` appearance tags for periodic verification (catching identity switches or re-acquiring lost targets).
5. **Command & Voice Interface:** `voice_command_interface.py` using a rule-based parser (regex + keyword matching + COCO class mapping) listening on UDP port 5000.
6. **State Machine & Control:** `drone_tracking_engine.py` with states (`FOLLOW`, `HOVER`, `SEARCH`, `SCAN`, `RTL`) driving `Sim2DDroneController` / `MAVSDKDroneController`.
7. **Threaded Camera Stream:** `ThreadedCameraStream` draining network/MJPEG buffers in a background thread to enable zero-lag streaming from IP Webcams (e.g. Android IP Webcam app) and USB webcams.

---

## Detailed Chat Trajectory & Problem Resolution

### Phase 1–4 Base Engine
- Built Kalman filter, YOLOv8n integration, SmolVLM re-ID engine, and Pygame sim controller.

### Phase 5 & 6 Enhancements
- Added `small_object_classifier.py`: Enriches tiny/uncertain YOLO crops with SmolVLM descriptions.
- Added `voice_command_interface.py`: Parses text or mic commands, maps natural language object names to COCO classes (e.g., "mobile phone" -> "cell phone"), and sends JSON packets over UDP.
- Shared VLM instance: Single loaded `SmolVLM-256M-Instruct` instance shared between ReID and Classifier to respect the ~1.5 GB RAM budget.

### Bug Fixes & Optimization Passes

1. **VLM Command Parser Removal:**
   - *Issue:* SmolVLM-256M was too small to reliably generate raw JSON for natural language text commands, hallucinating prompt headers.
   - *Fix:* Replaced text VLM in `voice_command_interface.py` with a fast, deterministic, rule-based parser with COCO class normalisation.

2. **Dynamic YOLO Target Classing:**
   - *Issue:* Tracking engine was hardcoded to `TARGET_CLASS_NAME = "person"`.
   - *Fix:* Updated `drone_tracking_engine.py` to parse `coco_class` from UDP packets (e.g. `"cell phone"` for "follow the mobile phone") and dynamically update detector target class.

3. **Background Thread Offloading for VLM (Camera Freezes):**
   - *Issue:* Synchronous VLM inference on CPU took ~10s, blocking `cap.read()` and `cv2.imshow()`, causing camera feed freezes.
   - *Fix:* Offloaded VLM classification and Re-ID verification in `drone_tracking_engine.py` to an asynchronous `ThreadPoolExecutor`. Added a 4-second spatial cache for VLM labels.

4. **Prompt Decoding Fix:**
   - *Issue:* Model output leaked prompt prefixes like `user:`.
   - *Fix:* Added token slicing (`generated_ids[0][inputs["input_ids"].shape[1]:]`) before decoding in `small_object_classifier.py` and `reid_engine.py`.

5. **Locked Target Visuals & Keyboard Control:**
   - *Feature:* Locked targets now render with a **bright red** bounding box (`BGR: (0, 0, 255)`), 2px thickness, and `[LOCKED]` label.
   - *Feature:* Added keyboard key **`u`** to unlock the target and return to hover.

6. **IP Webcam Integration & Zero-Lag Stream:**
   - *Feature:* Added interactive camera selection menu on startup (Option 1: Laptop, Option 2: IP Webcam). Auto-formats URLs like `http://192.168.0.253:8080/video`.
   - *Feature:* Added `ThreadedCameraStream` to continuously drain OpenCV MJPEG buffer queues in a background thread, eliminating 5-10s stream lag and auto-resizing 1080p/4K phone streams to 640x480.

7. **Auto-Locking & Non-Blocking ReID:**
   - *Issue:* Sending "follow the mobile phone" did not automatically lock onto the detected object, requiring manual `l` key press. Pressing `l` ran synchronous `reid.lock_target`, causing a 10s main-thread freeze.
   - *Fix:* Added auto-locking when entering `MISSION_FOLLOW` with detected candidates. Offloaded `reid.lock_target` tag generation to background thread pool.

---

## Workspace File Structure

- `c:\codes\drone_vlm\drone_engine\drone_tracking_engine.py` — Main integrated tracking loop & UI
- `c:\codes\drone_vlm\drone_engine\voice_command_interface.py` — Natural language & voice UDP sender
- `c:\codes\drone_vlm\drone_engine\small_object_classifier.py` — SmolVLM small-object descriptor
- `c:\codes\drone_vlm\drone_engine\reid_engine.py` — SmolVLM appearance token & re-ID verifier
- `c:\codes\drone_vlm\drone_engine\kalman_tracker.py` — Constant-velocity Kalman Filter
- `c:\codes\drone_vlm\drone_engine\drone_controller.py` — 2D Pygame Sim & MAVSDK Drone Controller interface
- `c:\codes\drone_vlm\drone_engine\_smoke_test.py` — Logic & synonym test script
- `c:\codes\drone_vlm\drone_engine\requirements.txt` — Dependencies list
- `c:\codes\drone_vlm\anticontext\context.md` — Complete conversation and project context log

---

## How to Run

1. **Start the Tracking Engine:**
   ```powershell
   c:\codes\drone_vlm\.venv\Scripts\python.exe drone_tracking_engine.py
   ```
   - Select Option `1` for Laptop camera or `2` for IP Webcam (e.g. `http://192.168.0.253:8080/video`).

2. **Send Voice / Text Commands:**
   ```powershell
   c:\codes\drone_vlm\.venv\Scripts\python.exe voice_command_interface.py --text "follow the mobile phone"
   c:\codes\drone_vlm\.venv\Scripts\python.exe voice_command_interface.py --text "hover"
   c:\codes\drone_vlm\.venv\Scripts\python.exe voice_command_interface.py --text "return home"
   ```

3. **Keyboard Controls:**
   - `[q]` — Quit
   - `[l]` — Lock onto highest confidence candidate
   - `[u]` — Unlock target
   - `[h]` — Hover
   - `[s]` — Search pattern
   - `[r]` — Return to Launch (RTL)
