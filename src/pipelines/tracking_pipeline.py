import cv2
import torch
import argparse
from pathlib import Path
import datetime

from evaluation.tracker_results_exporters import CSVTrackerExporter, VisualizerTrackerExporter
from models.trackers.tracker import SingleObjectTrackResult, BoundingBox
from models.trackers.tracker_factory import create_tracker
from utils.video_source import VideoSource

class TrackingPipeline:
    def __init__(self, tracker, exporters):
        self.tracker = tracker
        self.exporters = exporters

    def run(self, video_path):
        video_source = VideoSource(video_path)
        if len(video_source) == 0:
            print("Error: No frames found in the video source.")
            return
        
        initial_frame = next(video_source)

        roi = cv2.selectROI("Select Object to Track", initial_frame, fromCenter=False, showCrosshair=True)
        self.tracker.initialize(initial_frame, roi)
        cv2.destroyWindow("Select Object to Track")

        initial_bbox = BoundingBox(x=int(roi[0]), y=int(roi[1]), width=int(roi[2]), height=int(roi[3]))
        for exporter in self.exporters:
            exporter.on_track(initial_frame, SingleObjectTrackResult(bbox=initial_bbox, confidence=1.0))

        for frame in video_source:
            track_result = self.tracker.track(frame)
            for exporter in self.exporters:
                exporter.on_track(frame, track_result)

        print("Tracking completed. Exporting results...")

        for exporter in self.exporters:
            exporter.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo SiamFC Tracker")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--image_folder_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_path) / f"tracking" / datetime_str
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracker = create_tracker('siamfc', state_dict=args.model_path, device=device)

    exporters = [
        CSVTrackerExporter(output_dir),
        VisualizerTrackerExporter(output_dir, show=True)
    ]

    pipeline = TrackingPipeline(tracker, exporters)
    pipeline.run(Path(args.image_folder_path))
