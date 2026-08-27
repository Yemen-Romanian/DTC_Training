import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from models.abstract_trainable import AbstractTrainable
from models.losses import BANLoss
from models.trackers.tracker import SingleObjectTrackerBase, SingleObjectTrackResult, BoundingBox
from models.trackers.feature_extractors import (
    AlexNetFeatureExtractor,
    MobileNetV3FeatureExtractor,
    MobileNetV3MultiLayerFeatureExtractor,
)
from datasets.mixed_dataset import MixedDataset
from datasets.siamban_dataset import SiamBANDataset
from datasets.utils.tracking_augmentation_utils import get_subwindow, mean_channels
from utils.config import Config

logger = logging.getLogger(__name__)

# Feature map sizes are not fixed here: they depend on the backbone. MobileNetV3 emits 16x16 for
# the 127px exemplar and 32x32 for the 255px search crop, AlexNet 6x6 and 22x22. What matters is
# that the valid-padded correlation of the two yields the 17x17 response the training targets
# expect (32-16+1 = 22-6+1 = 17), which every supported backbone satisfies.


class DepthwiseCorr(nn.Module):
    """
    Depth-wise cross-correlation: each channel of the template is convolved
    independently over the corresponding channel of the search region.
    """

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, C, Hz, Wz = z.shape
        x_flat = x.view(1, B * C, x.shape[2], x.shape[3])
        z_flat = z.view(B * C, 1, Hz, Wz)
        out = F.conv2d(x_flat, z_flat, groups=B * C)
        return out.view(B, C, out.shape[2], out.shape[3])


def _head_branch(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 1),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1),
    )


class BANHead(nn.Module):
    """
    Box Adaptive Network head that performs classification and
    bbox regression for a single set of feature z and x.
    This unit is taken from https://arxiv.org/abs/2003.06761

    For each spatial location on the response map the head predicts:
      - cls: 2-channel foreground/background logits
      - reg: 4-channel distances to the target box edges (dl, dt, dr, db)
    """

    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()

        self.cls_z_proj = nn.Conv2d(in_channels, hidden_channels, 1)
        self.cls_x_proj = nn.Conv2d(in_channels, hidden_channels, 1)
        self.reg_z_proj = nn.Conv2d(in_channels, hidden_channels, 1)
        self.reg_x_proj = nn.Conv2d(in_channels, hidden_channels, 1)

        self.corr = DepthwiseCorr()

        self.cls_head = _head_branch(hidden_channels, 2)
        self.reg_head = _head_branch(hidden_channels, 4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        cls_feat = self.corr(self.cls_z_proj(z), self.cls_x_proj(x))
        reg_feat = self.corr(self.reg_z_proj(z), self.reg_x_proj(x))

        cls = self.cls_head(cls_feat)
        reg = F.relu(self.reg_head(reg_feat))

        return cls, reg


class SiamBANNet(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, hidden_channels: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = BANHead(in_channels, hidden_channels)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        return self.head(self.extract_features(z), self.extract_features(x))

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    @classmethod
    def from_config(cls, model_config: dict) -> 'SiamBANNet':
        backbone_config = model_config.get('backbone', {})
        backbone_type = backbone_config.get('type', 'AlexNet')
        freeze_backbone = backbone_config.get('freeze', False)
        pretrained = backbone_config.get('pretrained', True)
        hidden_channels = model_config.get('params', {}).get('hidden_channels', 256)

        logger.info(f"Creating SiamBAN model with backbone: {backbone_type}")
        logger.info(f"Backbone weights freeze: {freeze_backbone}")

        if backbone_type == 'AlexNet':
            backbone = AlexNetFeatureExtractor()
        elif backbone_type == 'MobileNetV3':
            backbone = MobileNetV3FeatureExtractor(
                freeze_weights=freeze_backbone,
                pretrained=pretrained
            )
        elif backbone_type == 'MobileNetV3Multi':
            taps = tuple(backbone_config.get('taps', (4, 9)))
            backbone = MobileNetV3MultiLayerFeatureExtractor(
                taps=taps,
                out_channels=backbone_config.get('out_channels', 48),
                freeze_weights=freeze_backbone,
                pretrained=pretrained
            )
            logger.info(f"Backbone taps: {taps} -> {backbone.out_channels} fused channels")
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        # Read from the backbone rather than a lookup table, so the head can never drift out of
        # sync with the width the backbone actually emits.
        return cls(backbone, backbone.out_channels, hidden_channels)


class TrainableSiamBAN(AbstractTrainable):
    def __init__(self, model_config: dict):
        self.net = SiamBANNet.from_config(model_config)
        self.loss_fn = BANLoss(cls_weight=1.0, reg_weight=1.0)

    @staticmethod
    def _prepare_batch(batch, device):
        """Move a batch to the device and normalize the uint8 crops to [0, 1] there.

        The dataset hands over uint8 crops so the worker->main transfer stays small;
        the /255 is a trivial GPU op.
        """
        z, x, cls_target, reg_target = [t.to(device, non_blocking=True) for t in batch]
        return z.float().div_(255.0), x.float().div_(255.0), cls_target, reg_target

    def train_step(self, batch, device) -> torch.Tensor:
        z, x, cls_target, reg_target = self._prepare_batch(batch, device)
        cls_pred, reg_pred = self.net(z, x)
        return self.loss_fn(cls_pred, reg_pred, cls_target, reg_target)

    def val_step(self, batch, device) -> torch.Tensor:
        z, x, cls_target, reg_target = self._prepare_batch(batch, device)
        cls_pred, reg_pred = self.net(z, x)
        return self.loss_fn(cls_pred, reg_pred, cls_target, reg_target)

    def build_datasets(self, config: Config) -> tuple[Dataset, Dataset, Dataset | None]:
        train_ds = SiamBANDataset(MixedDataset(config.get_train_paths()), augmentation=True)
        val_ds = SiamBANDataset(MixedDataset(config.get_val_paths()), augmentation=False)
        test_paths = config.get_test_paths()
        test_ds = SiamBANDataset(MixedDataset(test_paths), augmentation=False) if test_paths else None
        return train_ds, val_ds, test_ds

    def get_module(self) -> nn.Module:
        return self.net


class TrackerSiamBAN(SingleObjectTrackerBase):
    EXEMPLAR_SIZE = 127
    SEARCH_SIZE = 255
    RESPONSE_SIZE = 17
    STRIDE = 8

    # Weight of the cosine window in the blended score, in [0, 1].
    WINDOW_INFLUENCE = 0.30

    # Rate at which the target size is smoothed towards the newly predicted one.
    SIZE_LR = 0.1

    # Strength of the scale/aspect-ratio penalty; 0.0 disables it.
    PENALTY_K = 0.0

    def __init__(self, model: SiamBANNet, device: str):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Normalized by max, not sum. The score this is blended into is a softmax
        # probability peaking near 1.0, so the window has to peak at 1.0 too. Dividing
        # by the sum (as TrackerSiamFC does) drops the peak to 1/64, which made the
        # window term worth at most 0.0028 against a score term weighted 0.824 - the
        # spatial prior was ~250x weaker than WINDOW_INFLUENCE implies, i.e. inert.
        # The sum form is correct in TrackerSiamFC only because it normalizes its
        # response map to sum=1 first, putting both on the same scale.
        hann = np.hanning(self.RESPONSE_SIZE)
        self.window = np.outer(hann, hann)
        self.window /= self.window.max()

    def initialize(self, image: np.ndarray, bbox):
        # bbox: [x, y, w, h]
        self.target_sz = np.array([bbox[3], bbox[2]], dtype=float)   # [h, w]
        self.pos = np.array([bbox[1] + bbox[3] / 2, bbox[0] + bbox[2] / 2])  # [cy, cx]

        self._update_scales()

        avg_chans = mean_channels(image)
        z_crop = get_subwindow(image, self.pos, self.EXEMPLAR_SIZE, round(self.s_z), avg_chans)
        z_tensor = torch.from_numpy(z_crop).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            self.exemplar_features = self.model.extract_features(z_tensor)

    def track(self, image: np.ndarray) -> SingleObjectTrackResult:
        avg_chans = mean_channels(image)
        x_crop = get_subwindow(image, self.pos, self.SEARCH_SIZE, round(self.s_x), avg_chans)
        x_tensor = torch.from_numpy(x_crop).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            x_feat = self.model.extract_features(x_tensor)
            cls, reg = self.model.head(self.exemplar_features, x_feat)

        # Foreground score map: softmax over the 2-class dimension, take foreground score
        score = torch.softmax(cls[0], dim=0)[1].cpu().numpy()  # [17, 17]
        confidence = self._calculate_confidence(cls.cpu().numpy())

        reg_map = reg[0].cpu().numpy()  # [4, 17, 17] — (dl, dt, dr, db) per cell

        # Two priors are applied to the classification score before the argmax, in this
        # order: the scale/ratio penalty (per-cell, depends on what box that cell
        # predicts) and then the cosine window (positional). Applying the penalty first
        # matches the reference implementation - the window is a blend, so folding it in
        # earlier would let it dilute the penalty rather than act on the penalized map.
        score = score * self._size_penalty(reg_map)
        score = (1 - self.WINDOW_INFLUENCE) * score + self.WINDOW_INFLUENCE * self.window

        r_max, c_max = np.unravel_index(score.argmax(), score.shape)

        # Decode (dl, dt, dr, db) at the best location
        dl, dt, dr, db = reg_map[:, r_max, c_max]

        image_center = self.SEARCH_SIZE // 2   # 127
        px = image_center + (c_max - self.RESPONSE_SIZE // 2) * self.STRIDE
        py = image_center + (r_max - self.RESPONSE_SIZE // 2) * self.STRIDE

        cx_crop = np.clip(px + (dr - dl) / 2, 0.0, self.SEARCH_SIZE)
        cy_crop = np.clip(py + (db - dt) / 2, 0.0, self.SEARCH_SIZE)
        w_crop = np.clip(dl + dr, 1.0, self.SEARCH_SIZE)
        h_crop = np.clip(dt + db, 1.0, self.SEARCH_SIZE)

        # Map to original image space
        scale = self.s_x / self.SEARCH_SIZE
        cx_img = self.pos[1] + (cx_crop - image_center) * scale
        cy_img = self.pos[0] + (cy_crop - image_center) * scale
        w_img = w_crop * scale
        h_img = h_crop * scale

        # Cap per-frame size change to ±30% — prevents runaway when regression is noisy
        max_change = 1.3
        w_img = np.clip(w_img, self.target_sz[1] / max_change, self.target_sz[1] * max_change)
        h_img = np.clip(h_img, self.target_sz[0] / max_change, self.target_sz[0] * max_change)

        # Smooth update. The centre is clamped to the frame: once it leaves, get_subwindow
        # returns pure avg_chans padding, so the tracker sees a blank crop and can never
        # recover on its own. Clamping keeps at least part of the search region over real
        # pixels, leaving a chance to reacquire.
        h_img_bound, w_img_bound = image.shape[0], image.shape[1]
        self.pos = np.array([np.clip(cy_img, 0.0, h_img_bound - 1),
                             np.clip(cx_img, 0.0, w_img_bound - 1)])
        self.target_sz = (1 - self.SIZE_LR) * self.target_sz + self.SIZE_LR * np.array([h_img, w_img])
        self.target_sz = np.maximum(self.target_sz, 2.0)
        self._update_scales()

        bbox = BoundingBox(
            x=int(self.pos[1] - self.target_sz[1] / 2),
            y=int(self.pos[0] - self.target_sz[0] / 2),
            width=int(self.target_sz[1]),
            height=int(self.target_sz[0]),
        )
        return SingleObjectTrackResult(bbox=bbox, confidence=confidence)

    def to_device(self, device: str):
        self.device = device
        self.model.to(device)

    def _size_penalty(self, reg_map: np.ndarray) -> np.ndarray:
        """Down-weight cells whose predicted box disagrees in size or aspect with the target.

        The cosine window is a purely positional prior: it says nothing about whether the
        box a cell predicts is plausible. This is the other half of the standard SiamBAN
        selection rule - a cell that would triple the target's area or flip its aspect
        ratio is almost always a distractor or a bad regression, even when its
        classification score is high.

        Returns a [17, 17] multiplier in (0, 1]; identically 1.0 when PENALTY_K is 0.
        """
        if self.PENALTY_K <= 0.0:
            return np.ones_like(reg_map[0])

        # Predicted extents per cell, mapped from crop pixels to image pixels so they are
        # comparable with target_sz. eps guards the ratios: reg is ReLU'd, so a cell can
        # predict exactly 0 width or height.
        eps = 1e-6
        scale = self.s_x / self.SEARCH_SIZE
        w = (reg_map[0] + reg_map[2]) * scale + eps   # dl + dr
        h = (reg_map[1] + reg_map[3]) * scale + eps   # dt + db

        target_h, target_w = self.target_sz[0] + eps, self.target_sz[1] + eps

        def _change(ratio):
            """Symmetric deviation from 1.0, so growing and shrinking are penalized alike."""
            return np.maximum(ratio, 1.0 / ratio)

        def _padded_size(width, height):
            """Context-padded side length — the same convention as _update_scales."""
            pad = 0.5 * (width + height)
            return np.sqrt((width + pad) * (height + pad))

        scale_change = _change(_padded_size(w, h) / _padded_size(target_w, target_h))
        ratio_change = _change((target_w / target_h) / (w / h))

        return np.exp(-(scale_change * ratio_change - 1.0) * self.PENALTY_K)

    def _update_scales(self):
        """Recompute s_z / s_x from the current target size."""
        context = 0.5 * self.target_sz.sum()
        self.s_z = np.sqrt((self.target_sz[0] + context) * (self.target_sz[1] + context))
        self.s_x = self.s_z * (self.SEARCH_SIZE / self.EXEMPLAR_SIZE)

    def _calculate_confidence(self, classification_map: np.ndarray) -> float:
        class_map_diff = classification_map[0, 1] - classification_map[0, 0] # foreground - background scores
        min_value = np.min(class_map_diff)
        apce = (np.max(class_map_diff) - min_value)**2 / np.mean((class_map_diff - min_value)**2)
        return apce
