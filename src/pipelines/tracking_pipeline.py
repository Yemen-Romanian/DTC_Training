import cv2
import torch
import argparse
from pathlib import Path
import datetime

from models.trackers.siamfc import SiamFCNet, TrackerSiamFC
from evaluation.tracker_results_exporters import CSVTrackerExporter, VisualizerTrackerExporter
from models.trackers.siamfc import TrackerSiamFC, SiamFCNet, AlexNetFeatureExtractor

class TrackingPipeline:
    def __init__(self, tracker, exporters):
        self.tracker = tracker
        self.exporters = exporters

    def run(self, video_path):
        cap = cv2.VideoCapture(str(video_path / "%06d.jpg"), cv2.CAP_IMAGES)
        ret, frame = cap.read()
        roi = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
        self.tracker.initialize(frame, roi)
        cv2.destroyWindow("Select Object to Track")

        if not cap.isOpened():
            print("Error: Could not open the image sequence.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            track_result = self.tracker.track(frame)
            for exporter in self.exporters:
                exporter.on_track(frame, track_result)

        cap.release()
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

    siamfc_model = SiamFCNet(AlexNetFeatureExtractor())
    siamfc_model.load_state_dict(torch.load(args.model_path))
    tracker = TrackerSiamFC(siamfc_model)

    exporters = [
        CSVTrackerExporter(output_dir),
        VisualizerTrackerExporter(output_dir, show=True)
    ]

    pipeline = TrackingPipeline(tracker, exporters)
    pipeline.run(Path(args.image_folder_path))
