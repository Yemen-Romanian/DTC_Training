from pathlib import Path
import pandas as pd
import numpy as np
import logging

from utils.video_source import VideoSource
from datasets.utils.video import Video

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SyntheticDataset:
    """"Reader for synthetic dataset captured using Unreal Engine 5.0.
    The dataset should have the following structure:
    root/
        video1/
            images/
                image1.jpg
                image2.jpg
                ...
            bboxes.csv
        video2/
            images/
                image1.jpg
                image2.jpg
                ...
            bboxes.csv
        ...
    The bboxes.csv should have the following format:
    image_idx, class_id, x, y, w, h, unknown
    """
    
    def __init__(self, root_path, image_extension='.jpg', csv_name='labels.txt'):
        self.root_path = Path(root_path)
        self.image_extension = image_extension
        self.csv_name = csv_name
        self._videos = None

    def parse(self):
        video_list = []
        
        for video_dir in self.root_path.iterdir():
            video_source = VideoSource(video_dir / "images")
            
            if len(video_source) == 0:
                logging.warning(f"No videos found in folder {video_dir.name}.")
                continue

            csv_path = video_dir / self.csv_name
            if not csv_path.exists():
                logging.warning(f"No {self.csv_name} file found in {video_dir.name}")
                continue

            gt_rects = self.parse_ground_truth(csv_path)

            video_list.append(Video(video_dir.name, video_source, gt_rects))
            
        logging.info(f"Video successfully extracted: {len(video_list)}")
        return video_list
    
    @staticmethod
    def parse_ground_truth(csv_path):
        df = pd.read_csv(csv_path, header=None, names=["image_idx", "class_id", "x", "y", "w", "h", "unknown"])
        df = df[(df["w"] > 0) & (df["h"] > 0)]
        gt_rects = df[["x", "y", "w", "h"]].values.astype(np.float32)
        image_indices = df["image_idx"].values.astype(int) - 1
        gt_rects = list(zip(image_indices, gt_rects))
        return gt_rects
    
    def __getitem__(self, i):
        if self._videos is None:
            self._videos = self.parse()
        return self._videos[i]
    
    def __len__(self):
        if self._videos is None:
            self._videos = self.parse()
        return len(self._videos)
