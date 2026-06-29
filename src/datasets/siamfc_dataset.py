import torch
import torch.utils
import random
import numpy as np
import bisect

from datasets.mixed_dataset import MixedDataset
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
        
        self.valid_gt_rects = [
            [(fi, rect) for fi, rect in v.gt_rects if rect[2] > 0 and rect[3] > 0]
            for v in self.videos
        ]
        self.video_slots = [max(0, len(valid) - self.min_gap) for valid in self.valid_gt_rects]
        self.cumulative_slots = np.cumsum(self.video_slots).tolist()

        self.augmentor = SiameseAugmentor(apply_augmentation=self.apply_augmentation)

    def _create_data_point(self, video_index, examplar_index, search_index):
        video = self.videos[video_index]
        valid_rects = self.valid_gt_rects[video_index]

        gt_z = valid_rects[examplar_index][1]
        gt_x = valid_rects[search_index][1]

        img_z = video.source[valid_rects[examplar_index][0]]
        img_x = video.source[valid_rects[search_index][0]]
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
        label = create_label(size=17, radius=16, stride=8, offset=(offset_y, offset_x), label_type='binary')

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
        high = min(len(self.valid_gt_rects[video_index]) - 1, exemplar_frame_index + self.max_gap)
        search_frame_index = random.randint(low, high)
        
        examplar_image, search_image, label = self._create_data_point(video_index, exemplar_frame_index, search_frame_index)
        return examplar_image, search_image, label
    
def demo(paths, num_samples=3):
    import matplotlib.pyplot as plt
    inner_dataset = MixedDataset(paths)
    dataset = SiamFCDataset(inner_dataset, apply_augmentation=True)

    if len(dataset) > 0:
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        for i in range(num_samples):
            sample_idx = np.random.randint(0, len(dataset))
            examplar, search, gt = dataset[sample_idx]

            search_img = search.numpy().transpose(1, 2, 0)
            examplar_img = examplar.numpy().transpose(1, 2, 0)
            gt_img = gt.numpy()

            axes[i, 0].imshow(examplar_img)
            axes[i, 0].set_title(f"Exemplar (z) {examplar_img.shape[:2]}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(search_img)
            axes[i, 1].set_title(f"Search Area (x) {search_img.shape[:2]}")
            axes[i, 1].axis('off')

            im_gt = axes[i, 2].imshow(gt_img, cmap='jet', interpolation='nearest')
            axes[i, 2].set_title(f"Ground Truth {gt_img.shape}")
            axes[i, 2].axis('off')
            if i == 0:
                plt.colorbar(im_gt, ax=axes[i, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
    else:
        print("Dataset is empty. Check your paths.")

if __name__ == '__main__':
    # To run this example, provide valid paths to your datasets
    paths = {
        "uav123": "C:\\Users\\yevhe\\PhDProjects\\datasets\\UAV123_Test",
    }
    demo(paths, num_samples=5)