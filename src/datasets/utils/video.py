from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Video:
    label: str
    frames: list
    gt_rects: np.ndarray
    metadata: list = None