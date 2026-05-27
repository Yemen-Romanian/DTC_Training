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

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Simulate torchvision ColorJitter on a BGR uint8 numpy array.
    """
    img = image.astype(np.float32)

    ops = np.random.permutation(4)   # random order, same as PyTorch

    for op in ops:
        if op == 0:   # brightness
            f = np.random.uniform(max(0.0, 1 - brightness), 1 + brightness)
            img = img * f
        elif op == 1:   # contrast
            f = np.random.uniform(max(0.0, 1 - contrast), 1 + contrast)
            grey_mean = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).mean()
            img = img * f + grey_mean * (1 - f)

        elif op == 2:   # saturation
            f = np.random.uniform(max(0.0, 1 - saturation), 1 + saturation)
            hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * f, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        elif op == 3:   # hue — OpenCV H is 0..180, not 0..360
            delta = np.random.uniform(-hue * 180, hue * 180)
            hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        img = np.clip(img, 0, 255)

    return img.astype(np.uint8)

def random_blur(image):
    """Blur with a random cross-shaped (motion-blur-like) kernel."""
    # Odd sizes 5, 7, 9, … 45
    size = np.random.choice(np.arange(5, 36, 2))
    kernel = np.zeros((size, size), dtype=np.float32)
    c = size // 2

    # wx splits weight between the vertical bar and the horizontal bar
    wx = np.random.random()
    kernel[:, c] += (1.0 / size) * wx          # vertical bar
    kernel[c, :] += (1.0 / size) * (1.0 - wx)  # horizontal bar

    return cv2.filter2D(image, -1, kernel)

def convert_to_3channel_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to 3-channel grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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
    
class SiamBANAugmentor:
    """
    Image-level and geometric augmentation for SiamBAN training.
    """

    def __init__(
        self,
        scale_jitter_range: float = 0.18,
        max_shift: float = 48.0,
        grayscale_prob: float = 0.25,
        color_jitter_prob: float = 0.6,
        blur_prob: float = 0.2,
        flip_prob: float = 0.5,
        color_params: dict | None = None,
    ):
        self.scale_jitter_range = scale_jitter_range
        self.max_shift = max_shift
        self.grayscale_prob = grayscale_prob
        self.color_jitter_prob = color_jitter_prob
        self.blur_prob = blur_prob
        self.flip_prob = flip_prob

        cp = color_params or {}
        self.brightness = cp.get('brightness', 0.2)
        self.contrast = cp.get('contrast',   0.2)
        self.saturation = cp.get('saturation', 0.2)
        self.hue = cp.get('hue', 0.1)

    def sample_crop_jitter(self):
        scale_factor = 1.0 + (np.random.random() * 2 - 1.0) * self.scale_jitter_range
        shift_y = (np.random.random() * 2 - 1.0) * self.max_shift
        shift_x = (np.random.random() * 2 - 1.0) * self.max_shift
        return scale_factor, shift_y, shift_x

    def __call__(
        self,
        z_crop: np.ndarray,
        x_crop: np.ndarray,
        cls_label: np.ndarray,
        regression_label: np.ndarray,
    ):

        if np.random.random() < self.flip_prob:
            z_crop = np.flip(z_crop, axis=1).copy()
            x_crop = np.flip(x_crop, axis=1).copy()
            cls_label = np.flip(cls_label, axis=1).copy()
            regression_label = np.flip(regression_label, axis=2).copy()
            regression_label[0], regression_label[2] = (
                regression_label[2].copy(), regression_label[0].copy()
            )

        if np.random.random() < self.grayscale_prob:
            z_crop = convert_to_3channel_grayscale(z_crop)
            x_crop = convert_to_3channel_grayscale(x_crop)

        if np.random.random() < self.color_jitter_prob:
            z_crop = color_jitter(z_crop, self.brightness, self.contrast,
                                  self.saturation, self.hue)
        if np.random.random() < self.color_jitter_prob:
            x_crop = color_jitter(x_crop, self.brightness, self.contrast,
                                  self.saturation, self.hue)

        if np.random.random() < self.blur_prob:
            x_crop = random_blur(x_crop)

        return z_crop, x_crop, cls_label, regression_label
