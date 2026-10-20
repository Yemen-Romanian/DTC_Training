from pathlib import Path

import numpy as np
import onnxruntime as ort

from models.trackers.siamban import TrackerSiamBAN
from utils.paths import Paths


class TrackerSiamBANOnnx(TrackerSiamBAN):
    """TrackerSiamBAN running on ONNX Runtime instead of PyTorch.

    Expects the backbone and head exported separately by utils/onnx/export_to_onnx.py:
    the backbone maps 'image' -> 'features' (dynamic spatial size, so one graph serves both
    the 127 exemplar and the 255 search crop), the head maps ('z_feat', 'x_feat') -> ('cls', 'reg').
    Preprocessing (ImageNet normalization) is baked into the backbone graph.
    """

    def __init__(self, backbone_path: str | Path = None, head_path: str | Path = None,
                 providers: list[str] = None):
        # Skips TrackerSiamBAN.__init__ on purpose: there is no torch model to move or eval.
        backbone_path = backbone_path or Paths.model_weights_dir() / "siamban_backbone.onnx"
        head_path = head_path or Paths.model_weights_dir() / "siamban_head.onnx"
        providers = providers or ['CPUExecutionProvider']
        self.device = 'cpu'

        self.backbone_session = ort.InferenceSession(str(backbone_path), providers=providers)
        self.head_session = ort.InferenceSession(str(head_path), providers=providers)

        self.window = self._cosine_window()

    def to_device(self, device: str):
        pass

    @staticmethod
    def _to_input(crop: np.ndarray) -> np.ndarray:
        return crop.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0

    def _embed_exemplar(self, z_crop: np.ndarray) -> np.ndarray:
        return self.backbone_session.run(['features'], {'image': self._to_input(z_crop)})[0]

    def _score_search(self, x_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_feat = self.backbone_session.run(['features'], {'image': self._to_input(x_crop)})[0]
        cls, reg = self.head_session.run(['cls', 'reg'], {'z_feat': self.exemplar_features, 'x_feat': x_feat})
        return cls, reg
