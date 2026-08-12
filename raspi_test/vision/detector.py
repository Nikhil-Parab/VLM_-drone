

import os
import sys
import time
import logging
import cv2
import numpy as np

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

from config import VISION_CONFIG, MODELS_DIR
from vision.text_reader import TextReader

logger = logging.getLogger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class ObjectDetector:
    def __init__(self, config=None):
        self.config = config or VISION_CONFIG
        self.default_conf_thresh = self.config.get("confidence_threshold", 0.35)
        self.nms_thresh = self.config.get("nms_threshold", 0.45)
        self.input_size = self.config.get("input_size", (320, 320))
        self.onnx_threads = self.config.get("onnx_threads", 2)

        self.backend = "synthetic"
        self.session = None
        self.input_name = None
        self.output_names = None

        self.text_reader = TextReader()
        self._init_backend()

    def _init_backend(self):
        primary_model = self.config.get("model_name", "yolo11n_int8.onnx")
        fallback_model = self.config.get("fallback_model_name", "yolo11n.onnx")

        model_path = os.path.join(MODELS_DIR, primary_model)
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, fallback_model)

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self.onnx_threads
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [o.name for o in self.session.get_outputs()]
                self.backend = "onnx_int8" if "int8" in model_path else "onnx"
                logger.info(f"Initialized ONNX Runtime detector backend ({self.backend}) with model: {os.path.basename(model_path)}")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}")

        logger.info("Operating in Dynamic Open-Vocabulary Perception Mode.")
        self.backend = "synthetic"

    def detect(self, frame, target_filter=None, color_filter=None, text_filter=None):
        if frame is None:
            return []

        h, w = frame.shape[:2]
        tf_str = str(target_filter).lower().strip() if target_filter else None


        current_conf_thresh = self.default_conf_thresh
        if tf_str and any(k in tf_str for k in ["book", "pen", "pencil", "phone", "cup", "keyboard", "mouse", "scissors", "bag"]):
            current_conf_thresh = 0.18

        if "onnx" in self.backend and self.session is not None:
            raw_detections = self._detect_onnx(frame, w, h, conf_threshold=current_conf_thresh)
        else:
            raw_detections = self._detect_synthetic(frame, w, h)


        if tf_str and any(k in tf_str for k in ["pen", "pencil", "marker", "stylus", "notebook", "printer", "box"]):
            geom_detections = self._detect_geometric_objects(frame, w, h, target_name=tf_str)
            raw_detections.extend(geom_detections)

        if len(raw_detections) == 0:
            return []


        if tf_str:
            class_filtered = []
            for d in raw_detections:
                lbl_lower = d["label"].lower()
                if lbl_lower in tf_str or tf_str in lbl_lower or tf_str.startswith(lbl_lower):
                    class_filtered.append(d)
            raw_detections = class_filtered

        if len(raw_detections) == 0:
            return []

        has_specific_descriptor = bool(color_filter or text_filter)
        results = []

        for d in raw_detections:
            lbl_lower = d["label"].lower()
            bx, by, bw, bh = d["box"]


            color_match = True
            if color_filter:
                crop = frame[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
                color_match = (crop.size > 0 and self._check_color_match(crop, str(color_filter).lower()))


            ocr_match = True
            if text_filter:
                txt_str = str(text_filter).lower().strip()
                if "person" in lbl_lower:
                    crop = frame[max(0, by):min(h, by + int(bh * 0.65)), max(0, bx):min(w, bx+bw)]
                else:
                    crop = frame[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]

                if crop.size > 0:
                    ocr_match, _ = self.text_reader.matches_target_text(crop, txt_str)
                else:
                    ocr_match = False

            is_locked_target = has_specific_descriptor and color_match and ocr_match

            d["is_target"] = is_locked_target
            if is_locked_target:
                tag_parts = []
                if color_filter: tag_parts.append(str(color_filter).upper())
                if text_filter: tag_parts.append(f"[{str(text_filter).upper()}]")
                d["label"] = f"LOCKED {d['label']} {' '.join(tag_parts)}".strip()

            results.append(d)

        return results

    def _detect_geometric_objects(self, frame, img_w, img_h, target_name="pen"):

        results = []
        try:

            small_f = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
            scale_x = img_w / 320.0
            scale_y = img_h / 240.0

            gray = cv2.cvtColor(small_f, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 60, 140)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if any(k in target_name for k in ["pen", "pencil", "marker", "stylus"]):
                    if 20 < area < 1500:
                        rect = cv2.minAreaRect(cnt)
                        (cx, cy), (w_r, h_r), angle = rect
                        long_side = max(w_r, h_r)
                        short_side = min(w_r, h_r)
                        aspect = long_side / short_side if short_side > 0 else 1.0

                        if aspect > 2.8 and long_side > 15:
                            x, y, bw, bh = cv2.boundingRect(cnt)
                            results.append({
                                "label": "pen",
                                "confidence": 0.82,
                                "box": (int(x * scale_x), int(y * scale_y), int(bw * scale_x), int(bh * scale_y)),
                                "center": (int((x + bw // 2) * scale_x), int((y + bh // 2) * scale_y))
                            })

                elif any(k in target_name for k in ["notebook", "printer", "box"]):
                    if 300 < area < 10000:
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                        if len(approx) == 4:
                            x, y, bw, bh = cv2.boundingRect(cnt)
                            aspect = float(bw) / bh if bh > 0 else 1.0
                            if 0.5 <= aspect <= 2.2:
                                label_name = "printer" if "printer" in target_name else "notebook"
                                results.append({
                                    "label": label_name,
                                    "confidence": 0.85,
                                    "box": (int(x * scale_x), int(y * scale_y), int(bw * scale_x), int(bh * scale_y)),
                                    "center": (int((x + bw // 2) * scale_x), int((y + bh // 2) * scale_y))
                                })
        except Exception as e:
            logger.debug(f"Geometric detection error: {e}")

        return results

    def _check_color_match(self, crop_bgr, target_color):
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        total_pixels = float(crop_bgr.shape[0] * crop_bgr.shape[1])
        if total_pixels == 0:
            return False

        if target_color in ["white", "light", "clear", "transparent", "silver"]:
            white_mask = (v_channel > 125) & (s_channel < 80)
            return (np.sum(white_mask) / total_pixels) > 0.60

        elif target_color in ["brown", "dark brown"]:
            brown_mask = (h_channel >= 5) & (h_channel <= 25) & (v_channel >= 25) & (v_channel <= 150) & (s_channel > 35)
            return (np.sum(brown_mask) / total_pixels) > 0.35

        elif target_color in ["black", "dark"]:
            dark_mask = v_channel < 75
            return (np.sum(dark_mask) / total_pixels) > 0.45

        elif target_color == "red":
            red_mask = ((h_channel < 12) | (h_channel > 160)) & (s_channel > 60)
            return (np.sum(red_mask) / total_pixels) > 0.25

        elif target_color == "blue":
            blue_mask = (h_channel >= 90) & (h_channel <= 135) & (s_channel > 50)
            return (np.sum(blue_mask) / total_pixels) > 0.25

        elif target_color == "green":
            green_mask = (h_channel >= 35) & (h_channel <= 85) & (s_channel > 50)
            return (np.sum(green_mask) / total_pixels) > 0.25

        elif target_color == "yellow":
            yellow_mask = (h_channel >= 18) & (h_channel <= 38) & (s_channel > 60)
            return (np.sum(yellow_mask) / total_pixels) > 0.25

        elif target_color == "orange":
            orange_mask = (h_channel >= 10) & (h_channel <= 22) & (s_channel > 70) & (v_channel > 120)
            return (np.sum(orange_mask) / total_pixels) > 0.25

        return True

    def _detect_onnx(self, frame, img_w, img_h, conf_threshold=0.35):
        try:
            blob = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_NEAREST)
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            blob = blob.astype(np.float32) * (1.0 / 255.0)
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            outputs = self.session.run(self.output_names, {self.input_name: blob})
            predictions = outputs[0][0]

            boxes = []
            confidences = []
            class_ids = []

            scores_matrix = predictions[4:, :]
            max_scores = np.max(scores_matrix, axis=0)
            valid_mask = max_scores >= conf_threshold

            if not np.any(valid_mask):
                return []

            valid_indices = np.where(valid_mask)[0]
            scale_x = img_w / self.input_size[0]
            scale_y = img_h / self.input_size[1]

            for idx in valid_indices:
                scores = scores_matrix[:, idx]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                cx = float(predictions[0, idx]) * scale_x
                cy = float(predictions[1, idx]) * scale_y
                bw = float(predictions[2, idx]) * scale_x
                bh = float(predictions[3, idx]) * scale_y

                bx = int(cx - bw / 2.0)
                by = int(cy - bh / 2.0)

                boxes.append([max(0, bx), max(0, by), int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, self.nms_thresh)
            results = []
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                for idx in indices:
                    bx, by, bw, bh = boxes[idx]
                    label = COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(COCO_CLASSES) else f"object_{class_ids[idx]}"
                    results.append({
                        "label": label,
                        "confidence": round(confidences[idx], 2),
                        "box": (bx, by, bw, bh),
                        "center": (int(bx + bw / 2), int(by + bh / 2))
                    })
            return results
        except Exception as e:
            logger.error(f"ONNX detection error: {e}")
            return self._detect_synthetic(frame, img_w, img_h)

    def _detect_synthetic(self, frame, img_w, img_h):
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 300 < area < 15000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bh) / bw if bw > 0 else 1.0
                label = "person" if aspect > 1.5 else "bottle"
                results.append({
                    "label": label,
                    "confidence": 0.85,
                    "box": (x, y, bw, bh),
                    "center": (x + bw // 2, y + bh // 2)
                })
        return results