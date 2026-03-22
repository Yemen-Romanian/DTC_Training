import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import argparse

from models.trackers.feature_extractors import AlexNetFeatureExtractor
from models.trackers.tracker import SingleObjectTrackerBase, SingleObjectTrackResult, BoundingBox


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
        examplar_features = self.extract_features(z)  # [B, C, Hz, Wz]
        search_features = self.extract_features(x)     # [B, C, Hx, Wx]
        score = self.compute_score(examplar_features, search_features)  # [B, 17, 17]
        return score
    
    def extract_features(self, image):
        return self.backbone(image)
    
    def compute_score(self, examplar_features, search_features):
        b, c, h, w = search_features.size()
        _, _, hz, wz = examplar_features.size()
        
        search_features = search_features.reshape(1, b * c, h, w)
        examplar_features = examplar_features.view(b, c, hz, wz)
        
        score = F.conv2d(search_features, examplar_features, groups=b) # [1, b, 17, 17]
        score = score.view(b, 1, score.size(2), score.size(3))
        score = score * self.scale_step + self.bias
        return score.squeeze(1)


class TrackerSiamFC(SingleObjectTrackerBase):
    EXAMPLAR_SIZE = 127
    ROI_SIZE = 255
    ROI_PAD_FACTOR = 2

    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


    def _get_subwindow(self, image, pos, model_sz, original_sz, avg_chans):
        sz = original_sz
        im_sz = image.shape
        
        c = (sz + 1) / 2
        context_xmin = round(pos[1] - c)
        context_xmax = context_xmin + sz - 1
        context_ymin = round(pos[0] - c)
        context_ymax = context_ymin + sz - 1
        
        left_pad = int(max(0, -context_xmin))
        top_pad = int(max(0, -context_ymin))
        right_pad = int(max(0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0, context_ymax - im_sz[0] + 1))

        context_xmin = int(context_xmin + left_pad)
        context_xmax = int(context_xmax + left_pad)
        context_ymin = int(context_ymin + top_pad)
        context_ymax = int(context_ymax + top_pad)

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.full((im_sz[0] + top_pad + bottom_pad, 
                            im_sz[1] + left_pad + right_pad, 3), 
                            avg_chans, dtype=np.uint8)
            te_im[top_pad:top_pad + im_sz[0], left_pad:left_pad + im_sz[1]] = image
            im_patch = te_im[context_ymin:context_ymax + 1, 
                             context_xmin:context_xmax + 1]
        else:
            im_patch = image[context_ymin:context_ymax + 1, 
                             context_xmin:context_xmax + 1]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
            
        return im_patch

    def initialize(self, image, bbox):
        self.pos = np.array([bbox[1] + bbox[3]/2, bbox[0] + bbox[2]/2])
        self.target_sz = np.array([bbox[3], bbox[2]])
        
        context = 0.5 * self.target_sz.sum()
        self.s_z = np.sqrt((self.target_sz[0] + context) * (self.target_sz[1] + context))
        self.s_x = self.s_z * (self.ROI_SIZE / self.EXAMPLAR_SIZE)
        z_crop = self._get_subwindow(image, self.pos, self.EXAMPLAR_SIZE, self.s_z, image.mean(axis=(0, 1)))
        z_tensor = torch.from_numpy(z_crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        z_tensor = z_tensor.to(self.device)
        
        with torch.no_grad():
            self.examplar_features = self.model.extract_features(z_tensor)

    def track(self, image):
        scales = [self.scale_step ** i for i in range(-(self.scale_num // 2), self.scale_num // 2 + 1)]

        crops = []
        for s in scales:
            cur_s_x = self.s_x * s
            crop = self._get_subwindow(image, self.pos, self.ROI_SIZE, cur_s_x, image.mean(axis=(0, 1)))
            crops.append(crop)

        x_batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        x_feat = self.model.extract_features(x_batch)

        with torch.no_grad():
            z_feat = self.examplar_features.repeat(self.scale_num, 1, 1, 1)
            score_maps = self.model.compute_score(z_feat, x_feat).squeeze().cpu().numpy()

        score_maps[0, :, :] *= self.scale_penalty
        score_maps[2, :, :] *= self.scale_penalty

        scale_idx = np.argmax(np.amax(score_maps, axis=(1, 2)))
        best_score_map = score_maps[scale_idx]

        print("Confidence:", best_score_map.max())
        best_score_map -= best_score_map.min()
        best_score_map /= (best_score_map.sum() + 1e-5)
        best_score_map = (1 - self.window_influence) * best_score_map + self.window_influence * self.window

        chosen_scale = scales[scale_idx]
        self.s_x = (1 - self.scale_lr) * self.s_x + self.scale_lr * (self.s_x * chosen_scale)

        self.target_sz = (1 - self.scale_lr) * self.target_sz + self.scale_lr * (self.target_sz * chosen_scale)

        r_max, c_max = np.unravel_index(best_score_map.argmax(), best_score_map.shape)
        disp_score = np.array([r_max - 8, c_max - 8])

        disp_real = disp_score * 8 * ((self.s_x * chosen_scale) / self.ROI_SIZE)

        self.pos += disp_real

        bbox = BoundingBox(
            x=int(self.pos[1] - self.target_sz[1]/2),
            y=int(self.pos[0] - self.target_sz[0]/2),
            width=int(self.target_sz[1]),
            height=int(self.target_sz[0])
        )

        confidence = 1.0  # Placeholder for confidence score

        return SingleObjectTrackResult(bbox=bbox, confidence=confidence)

