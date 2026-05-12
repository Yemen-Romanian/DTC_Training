import cv2
from models.trackers.tracker import (
    SingleObjectTrackerBase,
    SingleObjectTrackResult,
    BoundingBox
)
from utils.paths import Paths

class TrackerViT(SingleObjectTrackerBase):
    def __init__(self):
        model_dir = Paths.model_weights_dir()
        params = cv2.TrackerVit_Params()
        params.net = str(model_dir / "object_tracking_vittrack_2023sep.onnx")
        self.tracker = cv2.TrackerVit_create(params)

    def initialize(self, image, bbox):
        bbox = list(map(int, bbox))
        return self.tracker.init(image, bbox)
    
    def track(self, image) -> SingleObjectTrackResult:
        success, bbox = self.tracker.update(image)
        if not success:
            return SingleObjectTrackResult(BoundingBox(0, 0, 0, 0), 0.0)
        x, y, w, h = map(int, bbox)
        score = self.tracker.getTrackingScore()
        return SingleObjectTrackResult(BoundingBox(x, y, w, h), score)
    
    def to_device(self, device: str):
        pass