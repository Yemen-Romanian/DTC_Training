import argparse
import cv2
import numpy as np
import pandas as pd
import torch

from utils.video_source import VideoSource
from datasets.utils.video import Video
from datasets.synthetic_dataset import SyntheticDataset
from datasets.manual_uav_dataset import ManualUAVDataset
from datasets.uav123_dataset import UAV123Dataset
from evaluation.metrics import match_boxes
from models.trackers.siamfc import SiamFCNet, TrackerSiamFC
from models.trackers.feature_extractors import AlexNetFeatureExtractor
from utils.paths import Paths

TRUE_COLOR = (0, 255, 0)  # Green for ground truth
FALSE_COLOR = (0, 0, 255)  # Red for predictions
TEXT_COLOR = (255, 255, 255) # White text

def main():
    parser = argparse.ArgumentParser(description="Tracking Demo")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file or image sequence")
    parser.add_argument('--gt_path', type=str, required=True, help="Path to the ground truth annotations file.")
    parser.add_argument('--data_type', type=str, choices=['synthetic', 'manual', 'uav123'], required=True, help="Type of the dataset to use for demo")
    parser.add_argument('--tracker_results', type=str, required=False, help='Path to CSV file with tracker results to visualize. If not provided, the demo will create a tracker and run it in real time.')
    parser.add_argument('--debug', action='store_true', help='Whether to manually control the video playback (Default to false).')

    args = parser.parse_args()
    video = create_video_from_input_info(args.video_path, args.gt_path, args.data_type)
    debug_mode = args.debug
    
    file_mode = args.tracker_results is not None
    if file_mode:
        tracker_results = pd.read_csv(args.tracker_results)
    else:
        tracker = create_tracker(Paths.model_weights_dir() / "siamfc.pth")

    if len(video.source) == 0:
        print("Error: No frames found in the video source.")
        return
    
    current_gt_index = 0
    cv2.namedWindow("Tracking Demo", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for frame_idx, frame in enumerate(video.source):
        gt_frame = frame.copy()
        results_frame = frame.copy()

        if frame_idx == 0 and not file_mode:
            tracker.initialize(frame, video.gt_rects[0][1])

        if current_gt_index < len(video.gt_rects) and video.gt_rects[current_gt_index][0] == frame_idx:
            gt_rect = video.gt_rects[current_gt_index][1]
            current_gt_index += 1
        else:
            gt_rect = np.zeros((4, ))  # No ground truth for this frame

        gt_frame = draw_bounding_box(gt_frame, gt_rect, color=TRUE_COLOR)
        if file_mode:
            bbox_pred = tracker_results.iloc[[frame_idx + 1]]
        else:
            track_result = tracker.track(frame)
            bbox_pred = pd.DataFrame([{
                'x1': track_result.bbox.x,
                'y1': track_result.bbox.y,
                'x2': track_result.bbox.x + track_result.bbox.width,
                'y2': track_result.bbox.y + track_result.bbox.height
            }])
        bbox_gt = convert_gt_for_evaluation(frame_idx + 1, gt_rect)
        tp, fp, fn, metrics_list = match_boxes(bbox_gt, bbox_pred)

        bbox_to_draw = (bbox_pred['x1'].values[0],
                        bbox_pred['y1'].values[0],
                        bbox_pred['x2'].values[0] - bbox_pred['x1'].values[0],
                        bbox_pred['y2'].values[0] - bbox_pred['y1'].values[0])
        
        results_frame = draw_frame_number(results_frame, frame_idx + 1)
        results_frame = draw_metrics_info(results_frame, tp=tp, fp=fp, fn=fn, metrics_list=metrics_list)

        if tp > 0 and gt_rect.any():
            prediction_color = TRUE_COLOR
        else:
            prediction_color = FALSE_COLOR

        results_frame = draw_bounding_box(results_frame, bbox_to_draw, color=prediction_color)

        delimiter_image = np.zeros((20, frame.shape[1], 3), dtype=np.uint8)  # A black vertical bar as delimiter
        frame_to_display = np.vstack([gt_frame, delimiter_image, results_frame])  # Just display the same frame twice for demo
        cv2.imshow("Tracking Demo", frame_to_display)

        if debug_mode:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print('Exiting demo.')
                exit(0)
            elif key == ord(' '):
                continue
        elif cv2.waitKey(30) & 0xFF == ord('q'):
            break

def create_tracker(model_path):
    siamfc_model = SiamFCNet(AlexNetFeatureExtractor())
    siamfc_model.load_state_dict(torch.load(model_path))
    tracker = TrackerSiamFC(siamfc_model)
    return tracker


def draw_bounding_box(frame, bbox, color, thickness=2):
    """
    bbox -- tuple (x, y, w, h)
    """
    x, y, w, h = bbox
    frame = cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return frame


def draw_frame_number(frame, frame_number, relative_position=(0.01, 0.03)):
    text_scale = 1.0
    text_scale = max(0.5, min(frame.shape[1], frame.shape[0]) / 1000 * text_scale)
    position = (int(frame.shape[1] * relative_position[0]), int(frame.shape[0] * relative_position[1]))
    frame = cv2.putText(frame, f"Frame: {frame_number}", position, cv2.FONT_HERSHEY_SIMPLEX, text_scale, TEXT_COLOR, 2)
    return frame


def draw_metrics_info(frame, tp, fp, fn, metrics_list, relative_position=(0.15, 0.03)):
    text_scale = 1.0
    text_scale = max(0.5, min(frame.shape[1], frame.shape[0]) / 1000 * text_scale)
    position = (int(frame.shape[1] * relative_position[0]), int(frame.shape[0] * relative_position[1]))
    metrics = metrics_list[0] if metrics_list else {}
    metrics_str = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
    metrics_str += f", TP: {tp}, FP: {fp}, FN: {fn}"
    frame = cv2.putText(frame, metrics_str, position, cv2.FONT_HERSHEY_SIMPLEX, text_scale, TEXT_COLOR, 2)
    return frame


def create_video_from_input_info(video_path: str, gt_path: str, data_type: str) -> Video:
    if data_type == 'synthetic':
        gt_rects = SyntheticDataset.parse_ground_truth(gt_path)
    elif data_type == 'manual':
        gt_rects = ManualUAVDataset.parse_ground_truth(gt_path)
    elif data_type == 'uav123':
        gt_rects = UAV123Dataset.parse_ground_truth(gt_path)
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Supported types are: synthetic, manual, uav123.")
    
    video_source = VideoSource(video_path)
    return Video(str(video_path), video_source, gt_rects)


def convert_gt_for_evaluation(frame_id, gt_rect, class_id=0, track_id=0):
    """
    gt_rect is in the format (x, y, w, h)
    """
    result = {}
    x, y, w, h = gt_rect
    result['frame_id'] = frame_id
    result['class_id'] = class_id
    result['x1'] = x
    result['y1'] = y
    result['w'] = w
    result['h'] = h
    result['confidence'] = 1.0  # Ground truth has confidence of 1
    
    return pd.DataFrame([result])


if __name__ == "__main__":
    main()

