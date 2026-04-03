from dataclasses import dataclass
from pathlib import Path
import numpy as np

from utils.video_source import VideoSource

@dataclass
class Video:
    label: str
    source: VideoSource
    gt_rects: list
    metadata: list = None