import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from models.abstract_trainable import AbstractTrainable
from models.losses import BANLoss
from models.trackers.tracker import SingleObjectTrackerBase, SingleObjectTrackResult, BoundingBox
from models.trackers.feature_extractors import AlexNetFeatureExtractor, MobileNetV3FeatureExtractor
from datasets.mixed_dataset import MixedDataset
from datasets.siamban_dataset import SiamBANDataset
from datasets.utils.tracking_augmentation_utils import get_subwindow
from utils.config import Config

logger = logging.getLogger(__name__)

# Feature map sizes after backbone + bilinear interpolation
EXEMPLAR_FEATURE_SIZE = (6, 6)
SEARCH_FEATURE_SIZE = (22, 22)


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


# Backbone output channels — kept here so SiamBANNet.from_config can look them up
# without importing feature_extractors in two places.
_BACKBONE_OUT_CHANNELS = {
    'AlexNet': 128,
    'MobileNetV3': 48,
}


class SiamBANNet(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, hidden_channels: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = BANHead(in_channels, hidden_channels)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        z_feat = self.extract_features(z, EXEMPLAR_FEATURE_SIZE)
        x_feat = self.extract_features(x, SEARCH_FEATURE_SIZE)
        return self.head(z_feat, x_feat)

    def extract_features(self, image: torch.Tensor, output_size=None) -> torch.Tensor:
        feat = self.backbone(image)
        return feat

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
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        in_channels = _BACKBONE_OUT_CHANNELS[backbone_type]
        return cls(backbone, in_channels, hidden_channels)


class TrainableSiamBAN(AbstractTrainable):
    def __init__(self, model_config: dict):
        self.net = SiamBANNet.from_config(model_config)
        self.loss_fn = BANLoss(cls_weight=1.0, reg_weight=1.0)

    def train_step(self, batch, device) -> torch.Tensor:
        z, x, cls_target, reg_target = [t.to(device) for t in batch]
        cls_pred, reg_pred = self.net(z, x)
        return self.loss_fn(cls_pred, reg_pred, cls_target, reg_target)

    def val_step(self, batch, device) -> torch.Tensor:
        z, x, cls_target, reg_target = [t.to(device) for t in batch]
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

    def __init__(self, model: SiamBANNet, device: str):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        hann = np.hanning(self.RESPONSE_SIZE)
        self.window = np.outer(hann, hann)
        self.window /= self.window.sum()

        self.window_influence = 0.176
        self.size_lr = 0.1

    def initialize(self, image: np.ndarray, bbox):
        # bbox: [x, y, w, h]
        self.target_sz = np.array([bbox[3], bbox[2]], dtype=float)   # [h, w]
        self.pos = np.array([bbox[1] + bbox[3] / 2, bbox[0] + bbox[2] / 2])  # [cy, cx]

        self._update_scales()

        avg_chans = image.mean(axis=(0, 1))
        z_crop = get_subwindow(image, self.pos, self.EXEMPLAR_SIZE, round(self.s_z), avg_chans)
        z_tensor = torch.from_numpy(z_crop).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            self.exemplar_features = self.model.extract_features(z_tensor, EXEMPLAR_FEATURE_SIZE)

    def track(self, image: np.ndarray) -> SingleObjectTrackResult:
        avg_chans = image.mean(axis=(0, 1))
        x_crop = get_subwindow(image, self.pos, self.SEARCH_SIZE, round(self.s_x), avg_chans)
        x_tensor = torch.from_numpy(x_crop).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            x_feat = self.model.extract_features(x_tensor, SEARCH_FEATURE_SIZE)
            cls, reg = self.model.head(self.exemplar_features, x_feat)

        # Foreground score map: softmax over the 2-class dimension, take foreground score
        score = torch.softmax(cls[0], dim=0)[1].cpu().numpy()  # [17, 17]
        confidence = self._calculate_confidence(score)

        score = (1 - self.window_influence) * score + self.window_influence * self.window

        r_max, c_max = np.unravel_index(score.argmax(), score.shape)

        # Decode (dl, dt, dr, db) at the best location
        dl, dt, dr, db = reg[0, :, r_max, c_max].cpu().numpy()

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

        # Smooth update
        self.pos = np.array([cy_img, cx_img])
        self.target_sz = (1 - self.size_lr) * self.target_sz + self.size_lr * np.array([h_img, w_img])
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

    def _update_scales(self):
        """Recompute s_z / s_x from the current target size."""
        context = 0.5 * self.target_sz.sum()
        self.s_z = np.sqrt((self.target_sz[0] + context) * (self.target_sz[1] + context))
        self.s_x = self.s_z * (self.SEARCH_SIZE / self.EXEMPLAR_SIZE)

    def _calculate_confidence(self, score_map: np.ndarray) -> float:
        return score_map.max()
