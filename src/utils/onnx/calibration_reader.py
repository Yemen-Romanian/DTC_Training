import random

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, quantize_static

from datasets.siamban_dataset import SiamBANDataset
from datasets.mixed_dataset import MixedDataset


def sample_calibration_pairs(mixed_dataset: MixedDataset, num_pairs: int = 200, seed: int = 0):
    """Draw (exemplar, search) crop pairs, preprocessed exactly as TrackerSiamBANOnnx feeds them.

    Each crop comes out as float32 [1, 3, H, W] in [0, 1]: 127x127 for the exemplar, 255x255 for
    the search crop. Sampled once and shared by both readers, so the head is calibrated on the
    features of the very crops the backbone was calibrated on.
    """
    dataset = SiamBANDataset(mixed_dataset, augmentation=False)
    rng = random.Random(seed)
    # The dataset also draws the search frame with `random`, so seed it for a repeatable set.
    random.seed(seed)
    indices = rng.sample(range(len(dataset)), min(num_pairs, len(dataset)))

    pairs = []
    for i in indices:
        z, x, _, _ = dataset[i]   # uint8 CHW tensors
        pairs.append((_to_input(z.numpy()), _to_input(x.numpy())))
    return pairs


def _to_input(crop_chw: np.ndarray) -> np.ndarray:
    return crop_chw.astype(np.float32)[np.newaxis] / 255.0


class SiamBANBackboneCalibrationReader(CalibrationDataReader):
    """Feeds the backbone both crop sizes through its single 'image' input.

    The calibrator records each tensor's value range over all samples, so mixing 127x127
    and 255x255 inputs is fine: the scales it picks then cover both of the tracker's uses of
    the backbone.
    """

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]]):
        self.samples = [crop for pair in pairs for crop in pair]   # z0, x0, z1, x1, ...
        self.rewind()

    def get_next(self) -> dict | None:
        crop = next(self._iter, None)
        return None if crop is None else {'image': crop}

    def rewind(self):
        self._iter = iter(self.samples)


class SiamBANHeadCalibrationReader(CalibrationDataReader):
    """Feeds the head (z_feat, x_feat) pairs computed by the FP32 backbone.

    Use the FP32 backbone, not a quantized one: the head's scales should describe the
    features it is meant to receive, not ones already distorted by backbone quantization.
    """

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]], fp32_backbone_path):
        backbone = ort.InferenceSession(str(fp32_backbone_path), providers=['CPUExecutionProvider'])
        extract = lambda crop: backbone.run(['features'], {'image': crop})[0]
        self.samples = [{'z_feat': extract(z), 'x_feat': extract(x)} for z, x in pairs]
        self.rewind()

    def get_next(self) -> dict | None:
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self.samples)
