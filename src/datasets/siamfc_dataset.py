import torch
import torch.utils
import random
import cv2
import numpy as np
import bisect
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor


class SiamFCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.videos = self.dataset.parse()
        self.transform = transform
        self.min_gap = 2
        self.max_gap = 10
        print(f"Total videos: {len(self.videos)}")
        
        self.video_slots = [max(0, len(v.gt_rects) - self.min_gap) for v in self.videos]
    
        self.cumulative_slots = np.cumsum(self.video_slots).tolist()
        self.label = torch.from_numpy(SiamFCDataset.create_label(size=17, radius=16, stride=8))
        
    def create_label(size, radius, stride):
        tc = (size - 1) / 2
        y, x = np.ogrid[-tc:size-tc, -tc:size-tc]
        
        dist = np.sqrt(x**2 + y**2) * stride
        labels = (dist <= radius).astype(np.float32)
        return labels
    
    def get_subwindow_avg(im, pos, model_sz, original_sz, avg_chans):
        sz = original_sz
        im_sz = im.shape
        c = (sz + 1) / 2
        context_xmin = round(pos[1] - c)
        context_xmax = context_xmin + sz - 1
        context_ymin = round(pos[0] - c)
        context_ymax = context_ymin + sz - 1
        
        left_pad = int(max(0, -context_xmin))
        top_pad = int(max(0, -context_ymin))
        right_pad = int(max(0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0, context_ymax - im_sz[0] + 1))

        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((im_sz[0] + top_pad + bottom_pad,
                              im_sz[1] + left_pad + right_pad, 3), np.uint8)
            te_im[top_pad:top_pad + im_sz[0], left_pad:left_pad + im_sz[1]] = im
            if top_pad: 
                te_im[0:top_pad, :] = avg_chans
            if bottom_pad: 
                te_im[im_sz[0] + top_pad:, :] = avg_chans
            if left_pad: 
                te_im[:, 0:left_pad] = avg_chans
            if right_pad: 
                te_im[:, im_sz[1] + left_pad:] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1)]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1)]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        return im_patch

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
        s_x = s_z * (255 / 127)

        pos_z = [gt_z[1] + gt_z[3]/2, gt_z[0] + gt_z[2]/2]
        pos_x = [gt_x[1] + gt_x[3]/2, gt_x[0] + gt_x[2]/2]

        z_crop = SiamFCDataset.get_subwindow_avg(img_z, pos_z, 127, round(s_z), avg_chans)
        x_crop = SiamFCDataset.get_subwindow_avg(img_x, pos_x, 255, round(s_x), avg_chans)

        if self.transform:
            z_crop = self.transform(z_crop)
            x_crop = self.transform(x_crop)

        return z_crop, x_crop
        
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
        
        examplar_image, search_image = self._create_data_point(video_index, exemplar_frame_index, search_frame_index)
        return examplar_image, search_image, self.label

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