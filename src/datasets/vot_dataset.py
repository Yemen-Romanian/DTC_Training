from pathlib import Path
import pandas as pd
import numpy as np
import logging

from utils.video_source import VideoSource
from datasets.utils.video import Video

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VOTDataset:
    """Reader for VOT dataset. The dataset should have the following structure:
    root/
        video1/
            color/
                image1.jpg
                image2.jpg
                ...
            groundtruth.txt
        video2/
            color/
                image1.jpg
                image2.jpg
                ...
            groundtruth.txt
        ...
    The groundtruth.txt is a csv file and should have the following format:
    x, y, w, h """

    def __init__(self, root_path, image_extension='.jpg', annotation_name='groundtruth.txt', images_folder_name='color'):
        self.root_path = Path(root_path)
        self.image_extension = image_extension
        self.annotation_name = annotation_name
        self.images_folder_name = images_folder_name
        self._videos = list()

    def parse(self):
        video_folders = sorted(folder for folder in self.root_path.iterdir() if folder.is_dir())

        for video_folder in video_folders:
            video_source_name = video_folder.name
            annotation_path = video_folder / self.annotation_name

            if not annotation_path.is_file():
                logging.warning(f"No annotation file found in folder {video_source_name}.")
                continue

            video_source = VideoSource(video_folder / self.images_folder_name)

            if len(video_source) == 0:
                logging.warning(f"No videos found in folder {video_source_name}.")
                continue

            gt_rects = self.parse_ground_truth(annotation_path)
            video = Video(video_source_name, video_source, gt_rects)
            self._videos.append(video)

        logging.info(f"Video successfully extracted: {len(self._videos)}")
        return self._videos

    @staticmethod
    def parse_ground_truth(csv_path):
        bboxes = pd.read_csv(csv_path, header=None, names=["x", "y", "w", "h"])
        bboxes = bboxes.fillna(0)
        gt_rects = bboxes.values.astype(int)
        gt_rects = list(zip(bboxes.index.values, gt_rects))
        return gt_rects

    def __getitem__(self, i):
        if self._videos is None:
            self._videos = self.parse()
        return self._videos[i]

    def __len__(self):
        if self._videos is None:
            self._videos = self.parse()
        return len(self._videos)
