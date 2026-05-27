import torch
import torch.utils.data
import random
import numpy as np
import bisect

from datasets.mixed_dataset import MixedDataset
from datasets.utils.tracking_augmentation_utils import (
    get_subwindow,
    SiamBANAugmentor
)


class SiamBANDataset(torch.utils.data.Dataset):
    """
    Produces (exemplar, search, cls_target, reg_target) tuples for SiamBAN training.
    """

    EXEMPLAR_SIZE = 127
    SEARCH_SIZE = 255
    RESPONSE_SIZE = 17
    STRIDE = 8

    def __init__(
        self,
        dataset: MixedDataset,
        pos_factor: float = 0.5,
        neg_factor: float = 1.0,
        pos_num: int = 16,
        neg_num: int = 48,
        augmentation=False
    ):
        self.augmentation = augmentation
        self.dataset = dataset
        self.pos_factor = pos_factor
        self.neg_factor = neg_factor
        self.pos_num = pos_num
        self.neg_num = neg_num

        self.videos = self.dataset.parse()
        self.min_gap = 2
        self.max_gap = 10
        print(f"Total videos: {len(self.videos)}")

        self.video_slots = [max(0, len(v.gt_rects) - self.min_gap) for v in self.videos]
        self.cumulative_slots = np.cumsum(self.video_slots).tolist()

        if self.augmentation:
            self.augmentor = SiamBANAugmentor()

    def __len__(self) -> int:
        return self.cumulative_slots[-1] if self.cumulative_slots else 0

    def __getitem__(self, index: int):
        video_index = bisect.bisect_right(self.cumulative_slots, index)
        exemplar_index = (
            index if video_index == 0
            else index - self.cumulative_slots[video_index - 1]
        )

        video = self.videos[video_index]
        low = exemplar_index + self.min_gap
        high = min(len(video.gt_rects) - 1, exemplar_index + self.max_gap)
        search_index = random.randint(low, high)

        return self._create_sample(video_index, exemplar_index, search_index)

    def _create_sample(self, video_index: int, exemplar_index: int, search_index: int):
        video = self.videos[video_index]

        gt_z = video.gt_rects[exemplar_index][1]  # [x, y, w, h]
        gt_x = video.gt_rects[search_index][1]    # [x, y, w, h]

        img_z = video.source[video.gt_rects[exemplar_index][0]]
        img_x = video.source[video.gt_rects[search_index][0]]
        avg_chans = np.mean(img_z, axis=(0, 1))

        # Crop scale derived from exemplar context (SiamFC convention)
        wz, hz = gt_z[2], gt_z[3]
        context = 0.5 * (wz + hz)
        s_z = np.sqrt((wz + context) * (hz + context))
        s_x = s_z * (self.SEARCH_SIZE / self.EXEMPLAR_SIZE)

        if self.augmentation:
            scale_jitter, shift_y, shift_x = self.augmentor.sample_crop_jitter()
        else:
            scale_jitter, shift_y, shift_x = 1.0, 0.0, 0.0

        s_x_aug = s_x * scale_jitter
        image_shift = np.array([shift_y, shift_x]) * (s_x_aug / self.SEARCH_SIZE)

        pos_z = [gt_z[1] + gt_z[3] / 2, gt_z[0] + gt_z[2] / 2]  # [cy, cx]
        pos_x = [gt_x[1] + gt_x[3] / 2, gt_x[0] + gt_x[2] / 2]
        pos_x_aug = pos_x + image_shift  # shifted crop center in image space

        z_crop = get_subwindow(img_z, pos_z, self.EXEMPLAR_SIZE, round(s_z), avg_chans)
        x_crop = get_subwindow(img_x, pos_x_aug, self.SEARCH_SIZE, round(s_x_aug), avg_chans)

        # Target dimensions in the 255×255 search crop
        scale = self.SEARCH_SIZE / s_x_aug
        w_crop = gt_x[2] * scale
        h_crop = gt_x[3] * scale
        gt_offset_y = -shift_y
        gt_offset_x = -shift_x

        cls_target, reg_target = self._create_targets(w_crop, h_crop, gt_offset_y, gt_offset_x)

        if self.augmentation:
            z_crop, x_crop, cls_target, reg_target = self.augmentor(z_crop, x_crop, cls_target, reg_target)

        z_tensor = torch.from_numpy(z_crop.copy()).permute(2, 0, 1).float() / 255.0
        x_tensor = torch.from_numpy(x_crop.copy()).permute(2, 0, 1).float() / 255.0
        cls_tensor = torch.from_numpy(cls_target)
        reg_tensor = torch.from_numpy(reg_target)

        return z_tensor, x_tensor, cls_tensor, reg_tensor

    def _create_targets(self, w_crop: float, h_crop: float,
                        gt_offset_y: float = 0.0, gt_offset_x: float = 0.0):
        """
        Build classification and regression targets for one training sample.
        """

        H = W = self.RESPONSE_SIZE
        cy, cx = H // 2, W // 2

        # Displacement of each response-map cell from the crop center [H, W],
        # measured in crop pixels.
        j_grid, i_grid = np.meshgrid(np.arange(W), np.arange(H))
        dy = (i_grid - cy) * self.STRIDE  # [H, W]
        dx = (j_grid - cx) * self.STRIDE  # [H, W]

        rel_dy = dy - gt_offset_y  # [H, W]
        rel_dx = dx - gt_offset_x  # [H, W]

        hw = w_crop / 2
        hh = h_crop / 2

        a_pos = max(1.0, self.pos_factor * hw)
        b_pos = max(1.0, self.pos_factor * hh)
        a_neg = max(1.0, self.neg_factor * hw)
        b_neg = max(1.0, self.neg_factor * hh)

        # Elliptical positive / negative regions centred on the GT location.
        dist_pos = (rel_dx / a_pos) ** 2 + (rel_dy / b_pos) ** 2  # [H, W]
        dist_neg = (rel_dx / a_neg) ** 2 + (rel_dy / b_neg) ** 2  # [H, W]

        cls = np.zeros((H, W), dtype=np.int64)   
        cls[dist_neg <= 1] = -1   # ignore ring
        cls[dist_pos <= 1] =  1   # positive
        dl = (hw + rel_dx).astype(np.float32)
        dt = (hh + rel_dy).astype(np.float32)
        dr = (hw - rel_dx).astype(np.float32)
        db = (hh - rel_dy).astype(np.float32)

        reg = np.stack([dl, dt, dr, db], axis=0)  # [4, H, W]

        return cls, reg

    @staticmethod
    def _subsample(idx, keep_num: int):
        """Randomly keep at most keep_num entries from a np.where result tuple."""
        n = idx[0].shape[0]
        if n <= keep_num:
            return idx
        sel = np.random.choice(n, keep_num, replace=False)
        return tuple(ax[sel] for ax in idx)
