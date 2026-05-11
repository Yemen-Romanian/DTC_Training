import torch
import torch.utils
import random
import numpy as np
import bisect
import matplotlib.pyplot as plt

from datasets.utils.tracking_augmentation_utils import SiameseAugmentor, get_subwindow, create_label, sample_translation_jitter, sample_scale_jitter


class SiamFCDataset(torch.utils.data.Dataset):
    EXAMPLAR_SIZE = 127
    ROI_SIZE = 255

    def __init__(self, dataset, apply_augmentation=False):
        self.dataset = dataset
        self.apply_augmentation = apply_augmentation
        self.videos = self.dataset.parse()
        self.min_gap = 2
        self.max_gap = 10
        print(f"Total videos: {len(self.videos)}")
        
        self.video_slots = [max(0, len(v.gt_rects) - self.min_gap) for v in self.videos]
    
        self.cumulative_slots = np.cumsum(self.video_slots).tolist()

        self.augmentor = SiameseAugmentor(apply_augmentation=self.apply_augmentation)

    def _create_data_point(self, video_index, examplar_index, search_index):
        video = self.videos[video_index]
    
        gt_z = video.gt_rects[examplar_index][1]
        gt_x = video.gt_rects[search_index][1]

        img_z = video.source[video.gt_rects[examplar_index][0]]
        img_x = video.source[video.gt_rects[search_index][0]]
        avg_chans = np.mean(img_z, axis=(0, 1))

        def get_sz(bbox):
            w, h = bbox[2], bbox[3]
            context = (w + h) * 0.5
            return np.sqrt((w + context) * (h + context))

        s_z = get_sz(gt_z)
        s_x = s_z * (self.ROI_SIZE / self.EXAMPLAR_SIZE)

        if self.apply_augmentation:
            x_jitter, y_jitter = sample_translation_jitter(gt_x)
            scale_jitter = sample_scale_jitter()
            offset_y = (-y_jitter * (float(self.ROI_SIZE) / s_x)) / 8.0
            offset_x = (-x_jitter * (float(self.ROI_SIZE) / s_x)) / 8.0
        else:
            x_jitter, y_jitter = 0, 0
            offset_y, offset_x = 0, 0
            scale_jitter = 1.0

        pos_z = [gt_z[1] + gt_z[3]/2, gt_z[0] + gt_z[2]/2]
        pos_x = [gt_x[1] + gt_x[3]/2 + y_jitter, gt_x[0] + gt_x[2]/2 + x_jitter]
        s_x *= scale_jitter

        z_crop = get_subwindow(img_z, pos_z, self.EXAMPLAR_SIZE, round(s_z), avg_chans)
        x_crop = get_subwindow(img_x, pos_x, self.ROI_SIZE, round(s_x), avg_chans)
        label = create_label(size=17, radius=16, stride=8, offset=(offset_y, offset_x))

        return self.augmentor(z_crop, x_crop, label)
        
    def __len__(self):
        return self.cumulative_slots[-1] if self.cumulative_slots else 0
    
    def __getitem__(self, index):
        video_index = bisect.bisect_right(self.cumulative_slots, index)
    
        if video_index == 0:
            exemplar_frame_index = index
        else:
            exemplar_frame_index = index - self.cumulative_slots[video_index - 1]
        
        video = self.videos[video_index]

        low = exemplar_frame_index + self.min_gap
        high = min(len(video.gt_rects) - 1, exemplar_frame_index + self.max_gap)
        search_frame_index = random.randint(low, high)
        
        examplar_image, search_image, label = self._create_data_point(video_index, exemplar_frame_index, search_frame_index)
        return examplar_image, search_image, label

if __name__ == '__main__':
    dataset = SiamFCDataset(r"path", 
                            transform=ToTensor())

    sample_idx = np.random.randint(0, len(dataset)-1)
    examplar, search, gt = dataset[sample_idx]
    search = search.numpy().transpose(1, 2, 0)
    examplar = examplar.numpy().transpose(1, 2, 0)
    gt = gt.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(examplar)
    axes[0].set_title(f"Exemplar (z) {examplar.shape[:2]}")
    axes[0].axis('off')

    axes[1].imshow(search)
    axes[1].set_title(f"Search Area (x) {search.shape[:2]}")
    axes[1].axis('off')

    im3 = axes[2].imshow(gt, cmap='jet', interpolation='nearest')
    axes[2].set_title(f"Ground Truth {gt.shape}")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()