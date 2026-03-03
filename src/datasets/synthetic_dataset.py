from pathlib import Path
import pandas as pd
import numpy as np
import logging

from datasets.utils.video import Video

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SyntheticDataset:
    def __init__(self, root_path, image_extension='.jpg', csv_name='bboxes.csv'):
        self.root_path = Path(root_path)
        self.image_extension = image_extension
        self.csv_name = csv_name
        self._videos = None

    def parse(self):
        video_list = []
        
        for video_dir in self.root_path.iterdir():
            if not video_dir.is_dir():
                continue
            
            frames = sorted(list(video_dir.glob(f"*{self.image_extension}")))
            
            if not frames:
                logging.warning(f"No videos found in folder {video_dir.name}.")
                continue

            csv_path = video_dir / self.csv_name
            if not csv_path.exists():
                logging.warning(f"No {self.csv_name} file found in {video_dir.name}")
                continue

            df = pd.read_csv(csv_path, header=None, names=["image_idx", "class_id", "x", "y", "w", "h", "unknown"])
            gt_rects = df[["x", "y", "w", "h"]].values.astype(np.float32)
            image_indices = df["image_idx"].values.astype(int) - 1
            frames = np.array(frames)[image_indices]

            video_list.append(Video(video_dir.name, frames, gt_rects))
            
        logging.info(f"Video successfully extracted: {len(video_list)}")
        return video_list
    
    def __getitem__(self, i):
        if self._videos is None:
            self._videos = self.parse()
        return self._videos[i]
    
    def __len__(self):
        if self._videos is None:
            self._videos = self.parse()
        return len(self._videos)
