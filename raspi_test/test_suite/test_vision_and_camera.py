

import os
import sys
import time
import cv2
import numpy as np
import logging


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CAMERA_CONFIG, VISION_CONFIG
from vision import CameraStream, ObjectDetector, CentroidTracker, EnvironmentScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestVision")

def run_vision_test(duration_seconds=10, save_snapshot=True):
    logger.info("=" * 60)
    logger.info("STARTING HEXACOPTER VISION & CAMERA TEST PROGRAM (RPi 5)")
    logger.info("=" * 60)


    logger.info("1. Initializing Camera Stream...")
    cam = CameraStream(CAMERA_CONFIG)
    cam.start()
    time.sleep(1.0)

    logger.info("2. Initializing Object Detector Engine...")
    detector = ObjectDetector(VISION_CONFIG)
    logger.info(f"Detector Active Backend: [{detector.backend}]")

    logger.info("3. Initializing Centroid Multi-Object Tracker...")
    tracker = CentroidTracker(max_disappeared=15, max_distance=80.0)

    logger.info("4. Initializing Environment Geography Scanner...")
    scanner = EnvironmentScanner(VISION_CONFIG)

    start_time = time.time()
    frame_count = 0
    total_det_time = 0.0

    logger.info(f"Running inspection stream for {duration_seconds} seconds...")

    try:
        while time.time() - start_time < duration_seconds:
            loop_start = time.time()
            frame = cam.read()

            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]


            scan_res = scanner.scan_terrain(frame)
            processed_frame = scan_res["overlay"]


            t0 = time.time()
            detections = detector.detect(frame)
            det_latency = (time.time() - t0) * 1000.0
            total_det_time += det_latency


            tracked_objs = tracker.update(detections)


            for obj in tracked_objs:
                bx, by, bw, bh = obj.box
                cx, cy = obj.center
                cv2.rectangle(processed_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"#{obj.track_id} {obj.label} ({obj.confidence:.2f})",
                            (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.circle(processed_frame, (cx, cy), 4, (0, 0, 255), -1)


            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0.0


            cv2.putText(processed_frame, f"RPi 5 Vision HUD - FPS: {current_fps:.1f} | Latency: {det_latency:.1f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(processed_frame, f"Backend: {detector.backend} | Detections: {len(detections)} | Tracks: {len(tracked_objs)}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


            if save_snapshot and frame_count == 15:
                snapshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_inspection_frame.jpg")
                cv2.imwrite(snapshot_path, processed_frame)
                logger.info(f"SAVED INSPECTION SNAPSHOT FRAME to: {snapshot_path}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Test manually stopped.")
    finally:
        cam.stop()

    avg_fps = frame_count / (time.time() - start_time)
    avg_latency = total_det_time / max(1, frame_count)

    logger.info("=" * 60)
    logger.info("TEST PROGRAM RESULTS SUMMARY:")
    logger.info(f"- Total Frames Processed: {frame_count}")
    logger.info(f"- Average Execution FPS:  {avg_fps:.2f} FPS")
    logger.info(f"- Average Inference Time: {avg_latency:.2f} ms")
    logger.info(f"- Camera Driver Mode:     {cam.mode}")
    logger.info(f"- Detector Engine Mode:   {detector.backend}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_vision_test(duration_seconds=5)