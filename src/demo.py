import argparse
from decimal import MAX_EMAX
import cv2
import numpy as np
import pandas as pd
import collections
import time

from utils.video_source import VideoSource
from datasets.utils.video import Video
from datasets.synthetic_dataset import SyntheticDataset
from datasets.manual_uav_dataset import ManualUAVDataset
from datasets.uav123_dataset import UAV123Dataset
from datasets.visdrone_dataset import VisDroneDataset
from datasets.vot_dataset import VOTDataset
from datasets.mmfw_dataset import MMFWDataset
from evaluation.metrics import match_boxes
from models.trackers.tracker_factory import create_tracker
from utils.paths import Paths
from utils.config import load_config

TRUE_COLOR = (0, 255, 0)  # Green for ground truth
FALSE_COLOR = (0, 0, 255)  # Red for predictions
TEXT_COLOR = (255, 255, 255) # White text

MIN_FRAMES_FOR_OOV = 1
MAX_FRAMES_FOR_OOV_REDETECT = 10

def main():
    parser = argparse.ArgumentParser(description="Tracking Demo")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file or image sequence")
    parser.add_argument('--model_config', type=str, required=True, help="Path to .toml file containing model's configuration. If only name is provided, it is assumed to be located in configs folder in the root dir of the project.")
    parser.add_argument('--gt_path', type=str, required=False, help="Path to the ground truth annotations file.")
    parser.add_argument('--data_type', type=str, required=False, choices=['synthetic', 'manual', 'uav123', 'visdrone', 'vot', 'mmfw'], help="Type of the dataset to use for demo")
    parser.add_argument('--tracker_results', type=str, required=False, help='Path to CSV file with tracker results to visualize. If not provided, the demo will create a tracker and run it in real time.')
    parser.add_argument('--debug', action='store_true', help='Whether to manually control the video playback (Default to false).')

    args = parser.parse_args()
    video = create_video_from_input_info(args.video_path)
    gt_rects = create_gt_rects(args.gt_path, args.data_type)
    debug_mode = args.debug

    if args.tracker_results:
        tracker = None
        tracker_results = pd.read_csv(args.tracker_results)
    else:
        model_config = load_config(args.model_config)
        state_dict_path = str(Paths.model_weights_dir() / model_config['weights']) if 'weights' in model_config else None
        tracker = create_tracker(model_config, state_dict=state_dict_path)
        tracker_results = None

    if len(video.source) == 0:
        print("Error: No frames found in the video source.")
        return

    if gt_rects:
        run_demo_with_gt(video, tracker, tracker_results, debug_mode, gt_rects)
    else:
        run_demo(video, tracker, tracker_results, debug_mode)


def run_demo_with_gt(video, tracker, tracker_results, debug_mode, gt_rects):
    init_counter = 0
    miss_counter = 0
    oov_total = 0
    oov_redetected = 0
    oov_run = 0            # length of the current run of all-zero GT rows
    oov_recovery = False   # True while we wait for the tracker to re-detect the object
    recovery_frames = 0    # frames spent in the current recovery window
    frame_times = collections.deque(maxlen=30)

    gt_by_frame = {idx: np.asarray(rect, dtype=float) for idx, rect in gt_rects}

    first_valid_idx = next((idx for idx, rect in gt_rects if np.any(rect)), None)
    if first_valid_idx is None:
        print("Error: ground truth contains only zero rows.")
        return

    cv2.namedWindow("Tracking Demo", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for frame_idx, frame in enumerate(video.source):
        # Skip all leading frames whose GT is all zeros
        if frame_idx < first_valid_idx:
            continue

        gt_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        results_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

        gt_rect = gt_by_frame.get(frame_idx, np.zeros((4,)))

        if frame_idx == first_valid_idx and not tracker_results:
            tracker.initialize(frame, gt_rect)

        # OOV detection and OOV end -> start of the recovery window
        if not gt_rect.any():
            if oov_run == 0:
                oov_total += 1
            oov_run += 1
        else:
            if oov_run >= MIN_FRAMES_FOR_OOV:   # a real OOV just ended
                oov_recovery = True
                recovery_frames = 0
                miss_counter = 0                # misses accumulated during OOV don't count
            oov_run = 0

        gt_frame = draw_bounding_box(gt_frame, gt_rect, color=TRUE_COLOR)
        if tracker_results:
            bbox_pred = tracker_results.iloc[[frame_idx + 1]]
            confidence = 1.0 # Add confidence to CSV file
            fps = None
        else:
            t0 = time.perf_counter()
            track_result = tracker.track(frame)
            frame_times.append(time.perf_counter() - t0)
            fps = len(frame_times) / sum(frame_times)
            bbox_pred = pd.DataFrame([{
                'x1': track_result.bbox.x,
                'y1': track_result.bbox.y,
                'x2': track_result.bbox.x + track_result.bbox.width,
                'y2': track_result.bbox.y + track_result.bbox.height
            }])
            confidence = track_result.confidence

        bbox_gt = convert_gt_for_evaluation(frame_idx + 1, gt_rect)
        tp, fp, fn, metrics_list = match_boxes(bbox_gt, bbox_pred)

        # OOV recovery: give the tracker MAX_FRAMES_FOR_OOV_REDETECT frames to re-detect
        if oov_recovery:
            recovery_frames += 1
            if tp > 0:
                oov_redetected += 1
                oov_recovery = False
            elif recovery_frames >= MAX_FRAMES_FOR_OOV_REDETECT:
                if gt_rect.any() and not tracker_results:
                    tracker.initialize(frame, gt_rect)
                    init_counter += 1
                    miss_counter = 0
                    oov_recovery = False

        if (fp > 0 or fn > 0) and gt_rect.any():
            miss_counter += 1
            if miss_counter > 20 and gt_rect.any() and not tracker_results:
                tracker.initialize(frame, gt_rect)
                init_counter += 1
                miss_counter = 0
        if tp > 0:
            miss_counter = 0

        bbox_to_draw = (bbox_pred['x1'].values[0],
                        bbox_pred['y1'].values[0],
                        bbox_pred['x2'].values[0] - bbox_pred['x1'].values[0],
                        bbox_pred['y2'].values[0] - bbox_pred['y1'].values[0])

        results_frame = draw_frame_number(results_frame, frame_idx + 1)
        results_frame = draw_metrics_info(results_frame, tp=tp, fp=fp, fn=fn, miss=miss_counter,
            init=init_counter, metrics_list=metrics_list)
        results_frame = draw_confidence_info(results_frame, confidence)
        results_frame = draw_oov_info(results_frame, oov_total, oov_redetected,
                                      in_oov=oov_run > 0, recovering=oov_recovery)

        if fps is not None:
            results_frame = draw_fps_info(results_frame, fps)

        if tp > 0 and gt_rect.any():
            prediction_color = TRUE_COLOR
        else:
            prediction_color = FALSE_COLOR

        results_frame = draw_bounding_box(results_frame, bbox_to_draw, color=prediction_color)

        #delimiter_image = np.zeros((20, frame.shape[1], 3), dtype=np.uint8)  # A black vertical bar as delimiter
        #frame_to_display = np.vstack([gt_frame, delimiter_image, results_frame])  # Just display the same frame twice for demo
        delimiter_image = np.zeros((frame.shape[0], 20, 3), dtype=np.uint8)  # vertical black bar
        frame_to_display = np.hstack([gt_frame, delimiter_image, results_frame])
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

def run_demo(video, tracker, tracker_results, debug_mode):
    frame_times = collections.deque(maxlen=30)
    cv2.namedWindow("Tracking Demo", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for frame_idx, frame in enumerate(video.source):
        results_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

        if frame_idx == 0 and not tracker_results:
            window_name = "Select Object to Track"
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            roi = cv2.selectROI(window_name, results_frame, fromCenter=False, showCrosshair=True)
            tracker.initialize(frame, roi)
            cv2.destroyWindow(window_name)

        if tracker_results:
            bbox_pred = tracker_results.iloc[[frame_idx + 1]]
            confidence = 1.0 # Add confidence to CSV file
            fps = None
        else:
            t0 = time.perf_counter()
            track_result = tracker.track(frame)
            frame_times.append(time.perf_counter() - t0)
            fps = len(frame_times) / sum(frame_times)
            bbox_pred = pd.DataFrame([{
                'x1': track_result.bbox.x,
                'y1': track_result.bbox.y,
                'x2': track_result.bbox.x + track_result.bbox.width,
                'y2': track_result.bbox.y + track_result.bbox.height
            }])
            confidence = track_result.confidence

        bbox_to_draw = (bbox_pred['x1'].values[0],
                        bbox_pred['y1'].values[0],
                        bbox_pred['x2'].values[0] - bbox_pred['x1'].values[0],
                        bbox_pred['y2'].values[0] - bbox_pred['y1'].values[0])

        results_frame = draw_frame_number(results_frame, frame_idx + 1)
        results_frame = draw_confidence_info(results_frame, confidence)

        if fps is not None:
            results_frame = draw_fps_info(results_frame, fps)

        prediction_color = TRUE_COLOR
        results_frame = draw_bounding_box(results_frame, bbox_to_draw, color=prediction_color)
        cv2.imshow("Tracking Demo", results_frame)

        if debug_mode:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print('Exiting demo.')
                exit(0)
            elif key == ord(' '):
                continue
        elif cv2.waitKey(30) & 0xFF == ord('q'):
            break


def draw_fps_info(frame, fps, relative_position=(0.01, 0.13)):
    text_scale = max(0.5, min(frame.shape[1], frame.shape[0]) / 1000)
    position = (int(frame.shape[1] * relative_position[0]), int(frame.shape[0] * relative_position[1]))
    frame = cv2.putText(frame, f"FPS: {fps:.1f}", position, cv2.FONT_HERSHEY_SIMPLEX, text_scale, TEXT_COLOR, 2)
    return frame


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


def draw_metrics_info(frame, tp, fp, fn, metrics_list, miss=0, init=0, relative_position=(0.21, 0.03)):
    text_scale = 1.0
    text_scale = max(0.5, min(frame.shape[1], frame.shape[0]) / 1000 * text_scale)
    position = (int(frame.shape[1] * relative_position[0]), int(frame.shape[0] * relative_position[1]))
    metrics = metrics_list[0] if metrics_list else {}
    metrics_str = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
    metrics_str += f", TP: {tp}, FP: {fp}, FN: {fn}"
    metrics_str += f", Miss: {miss}, Init: {init}"
    frame = cv2.putText(frame, metrics_str, position, cv2.FONT_HERSHEY_SIMPLEX, text_scale, TEXT_COLOR, 2)
    return frame

def draw_confidence_info(frame, confidence, relative_position=(0.01, 0.08), object_loss_threshold=0.5):
    text_scale = 1.0
    text_scale = max(0.5, min(frame.shape[1], frame.shape[0]) / 1000 * text_scale)
    position = (int(frame.shape[1] * relative_position[0]), int(frame.shape[0] * relative_position[1]))
    confidence_str = f"Confidence: {confidence:.3f}"
    frame = cv2.putText(frame, confidence_str, position, cv2.FONT_HERSHEY_SIMPLEX, text_scale, TEXT_COLOR, 2)

    if confidence < object_loss_threshold:  # Threshold for low confidence, can be adjusted
        frame = draw_object_loss_text(frame, relative_position=(0.20, relative_position[1]))
    return frame

def draw_oov_info(frame, oov_total, oov_redetected, in_oov=False, recovering=False,
                  relative_position=(0.01, 0.18)):
    text_scale = max(0.5, min(frame.shape[1], frame.shape[0]) / 1000)
    position = (int(frame.shape[1] * relative_position[0]), int(frame.shape[0] * relative_position[1]))
    text = f"OOV: {oov_total} (redetected: {oov_redetected})"
    if in_oov:
        text += " [OUT OF VIEW]"
    elif recovering:
        text += " [RE-DETECTING]"
    return cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, text_scale, TEXT_COLOR, 2)

def draw_object_loss_text(frame, relative_position=(0.30, 0.03)):
    text_scale = 1.0
    text_scale = max(0.5, min(frame.shape[1], frame.shape[0]) / 1000 * text_scale)
    position = (int(frame.shape[1] * relative_position[0]), int(frame.shape[0] * relative_position[1]))
    frame = cv2.putText(frame, "Object is lost", position, cv2.FONT_HERSHEY_SIMPLEX, text_scale, TEXT_COLOR, 2)
    return frame

def create_video_from_input_info(video_path: str) -> Video:
    video_source = VideoSource(video_path)
    return Video(str(video_path), video_source)

def create_gt_rects(gt_path: str, data_type: str):
    if data_type is None or gt_path is None:
        return None

    if data_type == 'synthetic':
        gt_rects = SyntheticDataset.parse_ground_truth(gt_path)
    elif data_type == 'manual':
        gt_rects = ManualUAVDataset.parse_ground_truth(gt_path)
    elif data_type == 'uav123':
        gt_rects = UAV123Dataset.parse_ground_truth(gt_path)
    elif data_type == 'visdrone':
        gt_rects = VisDroneDataset.parse_ground_truth(gt_path)
    elif data_type == 'vot':
        gt_rects = VOTDataset.parse_ground_truth(gt_path)
    elif data_type == 'mmfw':
        gt_rects = MMFWDataset.parse_ground_truth(gt_path)
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Supported types are: synthetic, manual, uav123, visdrone, vot, mmfw.")

    return gt_rects

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
