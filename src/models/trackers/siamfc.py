import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import logging

from models.abstract_trainable import AbstractTrainable
from models.losses import BalancedLoss
from models.trackers.tracker import SingleObjectTrackerBase, SingleObjectTrackResult, BoundingBox
from datasets.utils.tracking_augmentation_utils import get_subwindow
from datasets.siamfc_dataset import SiamFCDataset
from models.trackers.feature_extractors import AlexNetFeatureExtractor, MobileNetV3FeatureExtractor
from datasets.mixed_dataset import MixedDataset
from utils.config import Config

SEARCH_FEATURE_SIZE = (22, 22)
EXAMPLAR_FEATURE_SIZE = (6, 6)
logger = logging.getLogger(__name__)


class SiamFCNet(nn.Module):
    def __init__(self, backbone):
        super(SiamFCNet, self).__init__()
        self.backbone = backbone
        self.scale_step = nn.Parameter(torch.ones(1)*0.001)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, z, x):
        """
        z: [B, 3, 127, 127] - examplar image
        x: [B, 3, 255, 255] - search image
        """
        examplar_features = self.extract_features(z, output_size=EXAMPLAR_FEATURE_SIZE)  # [B, C, Hz, Wz]
        search_features = self.extract_features(x, output_size=SEARCH_FEATURE_SIZE)     # [B, C, Hx, Wx]
        score = self.compute_score(examplar_features, search_features)  # [B, 17, 17]
        return score
    
    def extract_features(self, image, output_size=None):
        features = self.backbone(image)
        if output_size is not None:
            features = F.interpolate(features, size=output_size, mode='bilinear', align_corners=False)
        return features
    
    def compute_score(self, examplar_features, search_features):
        b, c, h, w = search_features.size()
        _, _, hz, wz = examplar_features.size()
        
        search_features = search_features.reshape(1, b * c, h, w)
        examplar_features = examplar_features.view(b, c, hz, wz)
        
        score = F.conv2d(search_features, examplar_features, groups=b) # [1, b, 17, 17]
        score = score.view(b, 1, score.size(2), score.size(3))
        score = score * self.scale_step + self.bias
        return score.squeeze(1)
    
    @classmethod
    def from_config(cls, model_config: dict):
        backbone_config = model_config.get('backbone', {})
        backbone_type = backbone_config.get('type', 'AlexNet')
        freeze_backbone = backbone_config.get('freeze', False)
        logger.info(f"Creating SiamFC model with backbone: {backbone_type}")
        logger.info(f"Backbone weights freeze: {freeze_backbone}")

        # Instantiate Backbone
        if backbone_type == 'AlexNet':
            backbone = AlexNetFeatureExtractor()
        elif backbone_type == 'MobileNetV3':
            backbone = MobileNetV3FeatureExtractor(freeze_weights=freeze_backbone)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        return cls(backbone)


class TrackerSiamFC(SingleObjectTrackerBase):
    EXAMPLAR_SIZE = 127
    ROI_SIZE = 255
    ROI_PAD_FACTOR = 2

    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.score_size = 17
        
        hann = np.hanning(self.score_size)
        self.window = np.outer(hann, hann)
        self.window /= self.window.sum()
        self.window_influence = 0.176

        self.scale_step = 1.0375
        self.scale_penalty = 0.9745
        self.scale_lr = 0.59
        self.scale_num = 3

        self.scales = [self.scale_step ** i for i in range(-(self.scale_num // 2), self.scale_num // 2 + 1)]

    def initialize(self, image, bbox):
        self.pos = np.array([bbox[1] + bbox[3]/2, bbox[0] + bbox[2]/2])
        self.target_sz = np.array([bbox[3], bbox[2]])
        
        context = 0.5 * self.target_sz.sum()
        self.s_z = np.sqrt((self.target_sz[0] + context) * (self.target_sz[1] + context))
        self.s_x = self.s_z * (self.ROI_SIZE / self.EXAMPLAR_SIZE)
        z_crop = get_subwindow(image, self.pos, self.EXAMPLAR_SIZE, self.s_z, image.mean(axis=(0, 1)))
        z_tensor = torch.from_numpy(z_crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        z_tensor = z_tensor.to(self.device)
        
        with torch.no_grad():
            self.examplar_features = self.model.extract_features(z_tensor, output_size=EXAMPLAR_FEATURE_SIZE)

    def track(self, image):
        crops = []
        for s in self.scales:
            cur_s_x = max(1.0, self.s_x * s)
            crop = get_subwindow(image, self.pos, self.ROI_SIZE, cur_s_x, image.mean(axis=(0, 1)))
            crops.append(crop)

        x_batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        x_feat = self.model.extract_features(x_batch, output_size=SEARCH_FEATURE_SIZE)

        with torch.no_grad():
            z_feat = self.examplar_features.repeat(self.scale_num, 1, 1, 1)
            score_maps = self.model.compute_score(z_feat, x_feat).squeeze().cpu().numpy()

        score_maps[0, :, :] *= self.scale_penalty
        score_maps[2, :, :] *= self.scale_penalty

        scale_idx = np.argmax(np.amax(score_maps, axis=(1, 2)))
        best_score_map = score_maps[scale_idx]
        best_score_map -= best_score_map.min()
        best_score_map /= (best_score_map.sum() + 1e-5)
        best_score_map = (1 - self.window_influence) * best_score_map + self.window_influence * self.window

        chosen_scale = self.scales[scale_idx]
        self.s_x = (1 - self.scale_lr) * self.s_x + self.scale_lr * (self.s_x * chosen_scale)
        self.s_x = max(1.0, self.s_x)  # Ensure s_x does not become too small

        self.target_sz = (1 - self.scale_lr) * self.target_sz + self.scale_lr * (self.target_sz * chosen_scale)
        self.target_size = np.maximum(self.target_sz, 2.0)  # Ensure target size does not become too small

        r_max, c_max = np.unravel_index(best_score_map.argmax(), best_score_map.shape)
        disp_score = np.array([r_max - 8, c_max - 8])

        network_stride = self.model.backbone.stride
        disp_real = disp_score * network_stride * ((self.s_x * chosen_scale) / self.ROI_SIZE)

        self.pos += disp_real

        bbox = BoundingBox(
            x=int(self.pos[1] - self.target_sz[1]/2),
            y=int(self.pos[0] - self.target_sz[0]/2),
            width=int(self.target_sz[1]),
            height=int(self.target_sz[0])
        )

        confidence = 1.0  # Placeholder for confidence score

        return SingleObjectTrackResult(bbox=bbox, confidence=confidence)
    
    def to_device(self, device):
        self.device = device
        self.model.to(device)


class TrainableSiamFC(AbstractTrainable):
    def __init__(self, model_config: dict):
        self.net = SiamFCNet.from_config(model_config)
        self.loss_fn = BalancedLoss()

    def train_step(self, batch, device) -> torch.Tensor:
        z, x, gt = [t.to(device) for t in batch]
        pred = self.net(z, x)
        return self.loss_fn(pred, gt)

    def val_step(self, batch, device) -> torch.Tensor:
        z, x, gt = [t.to(device) for t in batch]
        pred = self.net(z, x)
        return self.loss_fn(pred, gt)

    def build_datasets(self, config: Config) -> tuple[Dataset, Dataset, Dataset | None]:
        train_ds = SiamFCDataset(MixedDataset(config.get_train_paths()), apply_augmentation=True)
        val_ds = SiamFCDataset(MixedDataset(config.get_val_paths()), apply_augmentation=False)
        test_paths = config.get_test_paths()
        test_ds = SiamFCDataset(MixedDataset(test_paths), apply_augmentation=False) if test_paths else None
        return train_ds, val_ds, test_ds

    def get_module(self) -> nn.Module:
        return self.net
