import cv2
from models.trackers.tracker import (
    SingleObjectTrackerBase,
    SingleObjectTrackResult,
    BoundingBox
)
from utils.paths import Paths


class TrackerNano(SingleObjectTrackerBase):
    def __init__(self, device):
        model_dir = Paths.model_weights_dir()
        params = cv2.TrackerNano_Params()
        params.backbone = str(model_dir / "nanotrack_backbone_sim.onnx")
        params.neckhead = str(model_dir / "nanotrack_head_sim.onnx")
        params.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        params.target = cv2.dnn.DNN_TARGET_CPU if device == 'cpu' else cv2.dnn.DNN_TARGET_OPENCL
        self.tracker = cv2.TrackerNano_create(params)

    def initialize(self, image, bbox):
        bbox = list(map(int, bbox))
        return self.tracker.init(image, bbox)
    
    def track(self, image) -> SingleObjectTrackResult:
        success, bbox = self.tracker.update(image)
        if not success:
            return SingleObjectTrackResult(BoundingBox(0, 0, 0, 0), 0.0)
        x, y, w, h = map(int, bbox)
        return SingleObjectTrackResult(BoundingBox(x, y, w, h), 1.0)
    
    def to_device(self, device: str):
        # This is handled by Opencv at the initialization of the tracker
        pass