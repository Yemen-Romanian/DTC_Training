import torch
import numpy as np
import random
import cv2
import torchvision.transforms as transforms

def get_subwindow(image, pos, model_sz, original_sz, avg_chans):
    """
    Extracts a square crop from an image centered at 'pos' with 'original_sz',
    pads with 'avg_chans' if the crop goes out of bounds, and resizes to 'model_sz'.
    """

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
        im_patch = cv2.resize(im_patch, (int(model_sz), int(model_sz)))
        
    return im_patch

def create_label(size, radius, stride, offset=(0, 0), label_type='gaussian'):
    """
    Creates a ground truth response map (label) for tracker training.
    """
    assert radius > 0, "Radius must be positive"
    assert stride > 0, "Stride must be positive"
    assert size > 0, "Size must be positive"
    
    tc = (size - 1) / 2
    y_idx, x_idx = np.ogrid[:size, :size]

    dy = y_idx - tc - offset[0]
    dx = x_idx - tc - offset[1]

    if label_type == 'binary':
        dist = np.sqrt(dx**2 + dy**2) * stride
        labels = (dist <= radius).astype(np.float32)
    elif label_type == 'gaussian':
        sigma = radius / stride
        dist_sq = dx**2 + dy**2
        labels = np.exp(-dist_sq / (2 * (sigma**2))).astype(np.float32)
        labels[labels < 1e-3] = 0 # Zero out negligible values
    else:
        raise ValueError(f"Unsupported label type: {label_type}")

    assert labels.shape == (size, size), f"Label shape mismatch: expected ({size}, {size}), got {labels.shape}"
    return labels

def sample_translation_jitter(bbox, factor=0.1):
    """
    Samples random translation jitter based on the bounding box size.
    """
    w, h = bbox[2], bbox[3]
    x_jitter = np.random.uniform(-w * factor, w * factor)
    y_jitter = np.random.uniform(-h * factor, h * factor)
    return x_jitter, y_jitter

def sample_scale_jitter(low=0.95, high=1.05):
    """
    Samples random scale jitter.
    """

    return np.random.uniform(low, high)

class SiameseAugmentor:
    """
    Handles synchronized data augmentation for Siamese tracking networks.
    Ensures that geometric transformations (like flipping) are applied consistently
    across the exemplar, search area, and ground truth label.
    """
    
    def __init__(self, apply_augmentation=True, flip_prob=0.5, color_params=None):
        self.apply_augmentation = apply_augmentation
        self.flip_prob = flip_prob
        
        if color_params is None:
            color_params = {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            }
        
        if self.apply_augmentation:
            self.color_jitter = transforms.ColorJitter(**color_params)

    def __call__(self, z_crop, x_crop, label):
        if self.apply_augmentation:
            if random.random() < self.flip_prob:
                z_crop = np.flip(z_crop, axis=1).copy()
                x_crop = np.flip(x_crop, axis=1).copy()
                label = np.flip(label, axis=1).copy()

        z_tensor = torch.from_numpy(z_crop).permute(2, 0, 1).float() / 255.0
        x_tensor = torch.from_numpy(x_crop).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.from_numpy(label).float()

        if self.apply_augmentation:
            x_tensor = self.color_jitter(x_tensor)

        return z_tensor, x_tensor, label_tensor
