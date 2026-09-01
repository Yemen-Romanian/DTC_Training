from pathlib import Path
import pandas as pd
import numpy as np
import logging

from utils.video_source import VideoSource
from datasets.utils.video import Video

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#: Annotations sit next to their video and are named after it, e.g. clip.mp4 -> clip_groundtruth.txt.
GROUND_TRUTH_SUFFIX = "_groundtruth.txt"


class ManualUAVDataset:
    """Reader for manually annotated tracking videos.

    The dataset is a single flat folder holding videos and their annotations side by side:
    root/
        video1.mp4
        video1_groundtruth.txt
        video2.mp4
        video2_groundtruth.txt
        ...
    A video with no matching annotation file is skipped with a warning.

    The annotation file is a headerless csv with one row per frame, in order, so the row index
    is the frame index:
    x, y, w, h
    Values are floats today and are read as such, which also covers an int-valued file. An
    all-zero row means the target is absent in that frame.
    """

    def __init__(self, root_path, video_extension='.mp4'):
        self.root_path = Path(root_path)
        self.video_extension = video_extension
        self._videos = None

    def parse(self):
        video_list = []

        # Suffix compared case-insensitively: the same folder holds both .mp4 and .MP4.
        for video_path in sorted(self.root_path.iterdir()):
            if not video_path.is_file() or video_path.suffix.lower() != self.video_extension.lower():
                continue

            annotation_path = self.root_path / f"{video_path.stem}{GROUND_TRUTH_SUFFIX}"
            if not annotation_path.exists():
                logging.warning(f"No {annotation_path.name} found for video {video_path.name}, skipping.")
                continue

            video_source = VideoSource(video_path)
            if len(video_source) == 0:
                logging.warning(f"No frames found in video {video_path.name}.")
                continue

            gt_rects = self.parse_ground_truth(annotation_path)
            # One row per frame is the whole contract of this format, so a mismatch means the
            # annotation and the video have drifted apart - loud here, invisible in the metrics.
            if len(gt_rects) != len(video_source):
                logging.warning(
                    f"{video_path.name}: {len(gt_rects)} annotated rows against "
                    f"{len(video_source)} frames."
                )

            video_list.append(Video(video_path.stem, video_source, gt_rects))

        logging.info(f"Video successfully extracted: {len(video_list)}")
        return video_list

    @staticmethod
    def parse_ground_truth(csv_path):
        bboxes = pd.read_csv(csv_path, header=None, names=["x", "y", "w", "h"])
        # A short or ragged line becomes an absent-object box rather than a NaN that would
        # travel silently into a metric.
        bboxes = bboxes.fillna(0)
        gt_rects = bboxes.values.astype(np.float32)
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
