from pathlib import Path
import pandas as pd
import numpy as np
import logging

from utils.video_source import VideoSource
from datasets.utils.video import Video

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class UAV123Dataset:
    """Reader for UAV123 dataset. The dataset should have the following structure:
    root/
        data_seq/
            video1/
                image1.jpg
                image2.jpg
                ...
            video2/
                image1.jpg
                image2.jpg
                ...
        anno/
            video1.txt
            video2.txt
            ...
    The video1.txt is a csv file and should have the following format:
    x, y, w, h """
    
    def __init__(self, root_path, image_extension='.jpg'):
        self.root_path = Path(root_path)
        self.image_extension = image_extension
        self._videos = list()
        self.annotation_folder = self.root_path / "anno"
        self.images_folder = self.root_path / "data_seq"

    def parse(self):
        annotation_files = list(self.annotation_folder.glob("*.txt"))

        for annotation_path in annotation_files:
            video_source_name = annotation_path.stem
            video_source_path = self.images_folder / video_source_name
            video_source = VideoSource(video_source_path)

            if len(video_source) == 0:
                logging.warning(f"No videos found in folder {video_source_name}.")
                continue

            bboxes = pd.read_csv(annotation_path, header=None, names=["x", "y", "w", "h"])
            valid_boxes_idx = bboxes[~bboxes['w'].isna()].index.astype(int).values
            gt_rects = bboxes.loc[valid_boxes_idx].values.astype(int)
            gt_rects = list(zip(valid_boxes_idx, gt_rects))
            video = Video(video_source_name, video_source, gt_rects)
            self._videos.append(video)

        logging.info(f"Video successfully extracted: {len(self._videos)}")
        return self._videos
    
    def __getitem__(self, i):
        if self._videos is None:
            self._videos = self.parse()
        return self._videos[i]
    
    def __len__(self):
        if self._videos is None:
            self._videos = self.parse()
        return len(self._videos)
    

if __name__ == "__main__":
    dataset = UAV123Dataset(r"C:\Users\yevhe\PhDProjects\datasets\UAV123_uav_only\train")
    videos = dataset.parse()