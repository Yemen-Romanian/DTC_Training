from pathlib import Path
import pandas as pd
import logging

from utils.video_source import VideoSource
from datasets.utils.video import Video

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ManualUAVDataset:
    """Reader for datasets annotated manually using CVAT tool.
    The dataset should have the following structure:
    root/
        videos/
            video1.mp4
            video2.mp4
            ...
        annotations/
            video1/
                gt.txt
            video2/
                gt.txt
            ...
    The gt.txt is a csv file and should have the following format (MOT 1.1):
    frame_id, track, x, y, w, h, not_ignored, class_id, visibility"""

    def __init__(self, root_path, video_extension='.mp4'):
        self.root_path = Path(root_path)
        self.video_extension = video_extension
        self._videos = list()
        self.annotation_folder = self.root_path / "annotations"
        self.videos_folder = self.root_path / "videos"

    def parse(self):
        annotation_files = list(self.annotation_folder.rglob("gt.txt"))

        for annotation_path in annotation_files:
            video_source_name = annotation_path.parent.name
            video_source_path = (self.videos_folder / video_source_name).with_suffix(self.video_extension)
            video_source = VideoSource(video_source_path)

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
        bboxes = pd.read_csv(
            csv_path, header=None, 
            names=["frame_id", "track", "x", "y", "w", "h", "not_ignored", "class_id", "visibility"]
            )
        
        valid_boxes_idx = bboxes['frame_id'].values.astype(int) - 1
        gt_rects = bboxes[["x", "y", "w", "h"]].values.astype(int)
        gt_rects = list(zip(valid_boxes_idx, gt_rects))
        return gt_rects
    
    def __getitem__(self, i):
        if self._videos is None:
            self._videos = self.parse()
        return self._videos[i]
    
    def __len__(self):
        if self._videos is None:
            self._videos = self.parse()
        return len(self._videos)
