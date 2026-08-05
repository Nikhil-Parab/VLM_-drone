# Autonomous AI Drone System Summary

## Overview
An autonomous drone pipeline for real-time target detection, tracking, recognition, localization, and mission execution.

## Camera Input
- Live camera: **30 FPS**
- Bounded ring buffer: **2–4 frames**
- Low-latency processing

## Perception Pipeline
1. **Global Detection**
   - YOLO-based detector (YOLO-XP)
   - Global Attention Module (GAM)
   - Detection: ~5–10 FPS
2. **ROI Extraction**
   - Extract small Regions of Interest (ROIs)
3. **Recognition**
   - LVLM (Large Vision Language Model)
   - Person/Object Re-ID
   - Semantic Guided Interaction (SGI)
4. **Multi-Object Tracking**
   - Tracker at ~30 FPS
   - Motion prediction and data association
5. **Temporal Evidence Fusion**
   - Target states:
     - Candidate
     - Verified
     - Confirmed
     - Tracking
     - Lost
     - Reacquiring
     - Dropped

## Processing Flow
```text
Camera (30 FPS)
      ↓
Ring Buffer
      ↓
YOLO Detection
      ↓
ROI Extraction
      ↓
LVLM + ReID
      ↓
Multi-Object Tracker
      ↓
Temporal Evidence Fusion
      ↓
Target Manager
      ↓
GPS Localization
      ↓
Mission Planner
      ↓
Pixhawk Flight Controller
```

## Vision Stack
- OpenCV
- NumPy

## Networking
- Port 5000 – Wi-Fi (UDP)
- Port 5002 – Telemetry (UDP)
- Port 5001 – Live Video (HTTP)

## Flight Controller
- PyMAVLink communication
- Pixhawk integration

## System Monitoring
- CPU usage
- RAM (<1.5 GB target)
- Thread watchdog
- Raspberry Pi SoC temperature
- Uses `psutil` and Linux `sysfs`

## Localization
Convert:
- 2D Bounding Boxes
- Pixhawk telemetry

Into:
- Roll
- Pitch
- Yaw
- Latitude
- Longitude
- Estimated 3D target GPS coordinates

## Mission Manager
Supports:
- Follow
- Track
- Hover
- Scan
- Survey

Safety:
- Maximum velocity
- Altitude limits
- Geofence
- Guidance safety

## AI Components
### Detection
- YOLO-XP
- P2 Feature Maps
- Global Attention Module (GAM)

### Recognition
- LVLM
- Person Re-ID
- Semantic Guided Interaction (SGI)

### Tracking
- Multi-object tracking
- Temporal evidence accumulation

## Hardware / Power Notes
- Raspberry Pi: 5V × 5A = 25 W
- Estimated total power: ~37.5 W
- Hover current: ~30 A
- Maximum thrust current: ~120 A
- XT60 connector: ~60 A rating

## Hyperspectral Camera Pipeline
```text
Hyperspectral Camera
        ↓
RGB Image
        ↓
Reshape (M × N × 3)
        ↓
Linearize Camera Response
        ↓
Apply Calibration Matrix
        ↓
Reshape Output
```

## Goal
Create an edge-AI autonomous drone capable of:
- Real-time object detection
- Robust target tracking
- Semantic recognition
- GPS localization
- Autonomous navigation
- Safe mission execution
