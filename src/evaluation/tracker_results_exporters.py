import cv2
import os
import datetime
import pandas as pd
from pathlib import Path

from models.trackers.tracker import SingleObjectTrackResult


class BaseExporter:
    def on_track(self, frame, bbox: SingleObjectTrackResult):
        pass
  
    def close(self):
        pass


class CSVTrackerExporter(BaseExporter):
    FRAME_ID = "frame_id"
    TRACK_ID = "track_id"
    LIFETIME = "lifetime"
    TIME = "time"
    X1 = "x1"
    Y1 = "y1"
    X2 = "x2"
    Y2 = "y2"
    CONF = "conf"
    CLASS_IDS = "class_ids"
    CLASS_NAME = "class_name"
    X = "x"
    Y = "y"
    Z = "z"
    X_TRG = "x_trg"
    Y_TRG = "y_trg"
    Z_TRG = "z_trg"
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"

    def __init__(self, output_path):
        self.output_path = output_path / "tracking_results.csv"
        self.results = []
        self.frame_id = 0
        self.track_id = 0
        self.lifetime = 0

    def on_track(self, frame, result: SingleObjectTrackResult, score_map=None):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]
        self.frame_id += 1
        self.lifetime += 1
        x, y, w, h = result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height
        x1, y1, x2, y2 = x, y, x + w, y + h

        self.results.append({
            self.FRAME_ID: self.frame_id,
            self.TRACK_ID: self.track_id,
            self.LIFETIME: self.lifetime,
            self.TIME: timestamp,
            self.X1: x1,
            self.Y1: y1,
            self.X2: x2,
            self.Y2: y2,
            self.CONF: result.confidence,
            self.CLASS_IDS: None,
            self.CLASS_NAME: None,
            self.X: None,
            self.Y: None,
            self.Z: None,
            self.X_TRG: None,
            self.Y_TRG: None,
            self.Z_TRG: None,
            self.ROLL: None,
            self.PITCH: None,
            self.YAW: None
        })

    def close(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_path, index=False)


class VisualizerTrackerExporter(BaseExporter):
    def __init__(self, output_dir, show=False):
        self.output_dir = Path(output_dir) / "visualization"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_idx = 0
        self.show = show

    def on_track(self, frame, result: SingleObjectTrackResult):
        self.frame_idx += 1
        x, y, w, h = result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height
        vis_frame = frame.copy()
        vis_frame = cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(self.output_dir / f"frame_{self.frame_idx:04d}.jpg", vis_frame)

        if self.show:
            cv2.imshow("Tracking Results", vis_frame)
            cv2.waitKey(1)
