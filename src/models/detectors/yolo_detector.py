from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from models.bbox import BoundingBox
from models.detectors.detector import AbstractDetector, DetectionResult
from utils.paths import Paths

class YOLODetector(AbstractDetector):
    def __init__(self, weights_filename, conf=None, iou=None, device=None, classes=None):
        model_weights_dir = Paths.model_weights_dir()
        full_weights_path = model_weights_dir / weights_filename
        self.detector = YOLO(full_weights_path)

        # Only forward what the caller set, so ultralytics' own defaults apply otherwise.
        self._predict_kwargs = {
            name: value
            for name, value in (("conf", conf), ("iou", iou), ("device", device), ("classes", classes))
            if value is not None
        }

    def detect(self, image) -> List[DetectionResult]:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"detect() expects an RGB numpy frame (H, W, 3), got {type(image).__name__}. "
                "Read files with cv2.imread + cv2.cvtColor(..., cv2.COLOR_BGR2RGB) first."
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"detect() expects a 3-channel (H, W, 3) frame, got shape {image.shape}")

        # This repo passes RGB frames around; ultralytics takes numpy input as BGR and
        # flips it back internally, so convert here rather than at every call site.
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        results = self.detector.predict(bgr_image, verbose=False, **self._predict_kwargs)
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        # .cpu().numpy() on the Boxes wrapper is a safe no-op when the data is already
        # a numpy array, so this works whether inference ran on CUDA or CPU.
        boxes = result.boxes.cpu().numpy()
        class_ids = boxes.cls.astype(int).tolist()

        detections = []
        for (x1, y1, x2, y2), confidence, class_id in zip(boxes.xyxy, boxes.conf, class_ids):
            # Ultralytics' .xywh is centre-based; BoundingBox is top-left, so go via .xyxy.
            bbox = BoundingBox(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
            )
            detections.append(DetectionResult(
                bbox=bbox,
                confidence=float(confidence),
                class_name=result.names[class_id],
            ))

        return detections
