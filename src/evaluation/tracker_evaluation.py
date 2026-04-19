import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets.dataset_factory import create_dataset
from evaluation.metrics import match_boxes
from models.trackers.tracker import SingleObjectTrackerBase
from utils.config import Config


def _convert_gt_for_evaluation(frame_id, gt_rect, class_id=0, track_id=0):
    x, y, w, h = gt_rect
    return pd.DataFrame([{
        'frame_id': frame_id,
        'class_id': class_id,
        'x1': x,
        'y1': y,
        'w': w,
        'h': h,
        'confidence': 1.0
    }])


def _evaluate_single_video(tracker, dataset_label, video):
    gt_rects = video.gt_rects
    current_gt_index = 0
    tracker.to_device('cuda' if torch.cuda.is_available() else 'cpu')

    video_metrics = {
        'iou': [],
        'iog': [],
        'center_dist': [],
        'center_dist_norm': []
    }

    for frame_idx in tqdm(range(len(video.source)), desc=f"Processing video {video.label}", total=len(video.source)):
        frame = video.source[frame_idx]
        if frame_idx == 0:
            tracker.initialize(frame, gt_rects[0][1])
            current_gt_index += 1
            continue

        if current_gt_index < len(gt_rects) and gt_rects[current_gt_index][0] == frame_idx:
            gt_rect = gt_rects[current_gt_index][1]
            current_gt_index += 1
        else:
            gt_rect = np.zeros((4, ))

        track_result = tracker.track(frame)
        bbox_pred = pd.DataFrame([{
            'x1': track_result.bbox.x,
            'y1': track_result.bbox.y,
            'x2': track_result.bbox.x + track_result.bbox.width,
            'y2': track_result.bbox.y + track_result.bbox.height
        }])
        bbox_gt = _convert_gt_for_evaluation(frame_idx + 1, gt_rect)
        _, _, _, metrics_list = match_boxes(bbox_gt, bbox_pred)
        frame_metrics = metrics_list[0]
        for name, value in frame_metrics.items():
            video_metrics[name].append(value)

    for name in list(video_metrics.keys()):
        video_metrics[name] = np.mean(video_metrics[name]) if len(video_metrics[name]) > 0 else float('nan')

    return dataset_label, video.label, video_metrics


def evaluate_tracker(tracker: SingleObjectTrackerBase, config: Config, max_workers=None):
    test_paths_dict = config.get_test_paths()
    evaluation_results = {}
    all_videos = {label: create_dataset(label, path).parse() for label, path in test_paths_dict.items()}
    overall_number_of_videos = sum(len(videos) for videos in all_videos.values())
    pbar = tqdm(total=overall_number_of_videos, desc='Evaluating on test videos')

    worker_count = max_workers or max((os.cpu_count() or 2) - 1, 1)
    tracker.to_device('cpu')

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_video = {}
        for dataset_label, video_list in all_videos.items():
            evaluation_results[dataset_label] = {}
            for video in video_list:
                future = executor.submit(_evaluate_single_video, tracker, dataset_label, video)
                future_to_video[future] = (dataset_label, video.label)

        for future in as_completed(future_to_video):
            dataset_label, video_label, video_metrics = future.result()
            evaluation_results.setdefault(dataset_label, {})[video_label] = video_metrics
            pbar.update(1)
            pbar.set_description(f'Processed {pbar.n}/{overall_number_of_videos} videos')

    pbar.close()
    return evaluation_results


def calculate_average_metrics(evaluation_results):
    average_metrics = defaultdict(list)

    for videos in evaluation_results.values():
        for metrics in videos.values():
            for name in metrics.keys():
                average_metrics[name].append(metrics[name])
    
    for name in average_metrics.keys():
        average_metrics[name] = np.mean(average_metrics[name])
    return average_metrics
