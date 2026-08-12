

import cv2
import numpy as np
import logging
from config import VISION_CONFIG

logger = logging.getLogger(__name__)

class EnvironmentScanner:
    def __init__(self, config=None):
        self.config = config or VISION_CONFIG
        self.water_hsv_lower = np.array(self.config.get("water_body_hsv_lower", [90, 40, 40]))
        self.water_hsv_upper = np.array(self.config.get("water_body_hsv_upper", [135, 255, 255]))

    def scan_terrain(self, frame):

        if frame is None:
            return {"water_bodies": [], "water_coverage_percent": 0.0, "terrain_summary": "unknown", "overlay": frame}

        h, w = frame.shape[:2]
        total_pixels = h * w
        overlay = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        water_mask = cv2.inRange(hsv, self.water_hsv_lower, self.water_hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

        water_pixels = cv2.countNonZero(water_mask)
        water_pct = (water_pixels / total_pixels) * 100.0

        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        water_bodies = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, bw, bh = cv2.boundingRect(cnt)

                    water_bodies.append({
                        "contour": cnt,
                        "area": area,
                        "center": (cx, cy),
                        "box": (x, y, bw, bh)
                    })


                    cv2.drawContours(overlay, [cnt], -1, (255, 200, 0), 2)
                    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 255, 0), 1)
                    cv2.putText(overlay, f"Water Body ({int(area)}px)", (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)


        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pixels = cv2.countNonZero(green_mask)
        green_pct = (green_pixels / total_pixels) * 100.0


        if water_pct > 30.0:
            dominant = "Water Body / Lake Region"
        elif green_pct > 40.0:
            dominant = "Dense Vegetation / Forest"
        elif green_pct > 15.0:
            dominant = "Mixed Rural Terrain"
        else:
            dominant = "Open Ground / Urban Area"


        cv2.putText(overlay, f"Geography Scan: {dominant}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(overlay, f"Water Coverage: {water_pct:.1f}%", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return {
            "water_bodies": water_bodies,
            "water_coverage_percent": round(water_pct, 2),
            "vegetation_percent": round(green_pct, 2),
            "terrain_summary": dominant,
            "overlay": overlay
        }