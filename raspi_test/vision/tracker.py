

import math
import numpy as np

class TrackedObject:
    def __init__(self, track_id, label, confidence, box, center, is_target=False):
        self.track_id = track_id
        self.label = label
        self.confidence = confidence
        self.box = box
        self.center = center
        self.is_target = is_target
        self.disappeared = 0

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80.0):
        self.next_object_id = 1
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    del self.objects[obj_id]
            return list(self.objects.values())

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for (i, d) in enumerate(detections):
            input_centroids[i] = d["center"]

        if len(self.objects) == 0:
            for i in range(len(detections)):
                d = detections[i]
                obj = TrackedObject(
                    self.next_object_id, d["label"], d["confidence"], d["box"], d["center"], d.get("is_target", False)
                )
                self.objects[self.next_object_id] = obj
                self.next_object_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj.center for obj in self.objects.values()]


            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                d = detections[col]
                self.objects[obj_id].center = d["center"]
                self.objects[obj_id].box = d["box"]
                self.objects[obj_id].label = d["label"]
                self.objects[obj_id].confidence = d["confidence"]
                self.objects[obj_id].is_target = d.get("is_target", False)
                self.objects[obj_id].disappeared = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                obj_id = object_ids[row]
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    del self.objects[obj_id]

            for col in unused_cols:
                d = detections[col]
                obj = TrackedObject(
                    self.next_object_id, d["label"], d["confidence"], d["box"], d["center"], d.get("is_target", False)
                )
                self.objects[self.next_object_id] = obj
                self.next_object_id += 1

        return list(self.objects.values())

    def calculate_tracking_error(self, active_track_id=None, frame_width=640, frame_height=480):
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        active_obj = None
        if active_track_id and active_track_id in self.objects:
            active_obj = self.objects[active_track_id]
        else:

            target_objs = [obj for obj in self.objects.values() if getattr(obj, "is_target", False)]
            if len(target_objs) > 0:
                active_obj = target_objs[0]
            elif len(self.objects) > 0:
                active_obj = list(self.objects.values())[0]

        if active_obj is None:
            return 0, 0, 0.0, None

        cx, cy = active_obj.center
        err_x = cx - frame_center_x
        err_y = cy - frame_center_y
        dist = math.sqrt(err_x**2 + err_y**2)

        return err_x, err_y, dist, active_obj