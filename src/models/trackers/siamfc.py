import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import argparse

from models.trackers.feature_extractors import AlexNetFeatureExtractor
from models.trackers.tracker import TrackerBase


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
        
        search_features = search_features.view(1, b * c, h, w)
        examplar_features = examplar_features.view(b, c, hz, wz)
        
        score = F.conv2d(search_features, examplar_features, groups=b) # [1, b, 17, 17]
        score = score.view(b, 1, score.size(2), score.size(3))
        score = score * self.scale_step + self.bias
        return score.squeeze(1)


class TrackerSiamFC(TrackerBase):
    EXAMPLAR_SIZE = 127
    ROI_SIZE = 255
    ROI_PAD_FACTOR = 2

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.score_size = 17
        
        hann = np.hanning(self.score_size)
        self.window = np.outer(hann, hann)
        self.window /= self.window.sum()
        self.window_influence = 0.176

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
        
        with torch.no_grad():
            self.examplar_features = self.model.extract_features(z_tensor)

    def track(self, image):
        x_crop = self._get_subwindow(image, self.pos, self.ROI_SIZE, self.s_x, image.mean(axis=(0, 1)))
        x_tensor = torch.from_numpy(x_crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        search_image_features = self.model.extract_features(x_tensor)

        with torch.no_grad():
            score_map = self.model.compute_score(self.examplar_features, search_image_features).squeeze().cpu().numpy()

        score_map -= score_map.min()
        score_map /= (score_map.sum() + 1e-5)
        score_map = (1 - self.window_influence) * score_map + self.window_influence * self.window

        r_max, c_max = np.unravel_index(score_map.argmax(), score_map.shape)
        disp_score = np.array([r_max - 8, c_max - 8])
        disp_real = disp_score * 8 * (self.s_x / self.ROI_SIZE)
        self.pos += disp_real
        
        return [int(self.pos[1] - self.target_sz[1]/2),
                int(self.pos[0] - self.target_sz[0]/2),
                self.target_sz[1], self.target_sz[0]]

    
def demo(model_path, image_folder_path):
    siamfc_model = SiamFCNet(AlexNetFeatureExtractor())
    siamfc_model.load_state_dict(torch.load(model_path))
    tracker = TrackerSiamFC(siamfc_model)

    image_folder = Path(image_folder_path)
    cap = cv2.VideoCapture(str(image_folder / "%06d.jpg"), cv2.CAP_IMAGES)
    ret, frame = cap.read()

    roi = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
    tracker.initialize(frame, roi)

    if not cap.isOpened():
        print("Error: Could not open the image sequence.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # Break the loop when no more frames are returned
            
            bbox = tracker.track(frame)
            x, y, w, h = bbox
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Process the frame (e.g., display it)
            cv2.imshow('Tracking result', frame)
            
            # Wait for 30ms and break if 'q' is pressed
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo SiamFC Tracker")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--image_folder_path', type=str)
    args = parser.parse_args()

    demo(args.model_path, args.image_folder_path)
