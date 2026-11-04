"""Measure the speed of whole trackers (per-frame track() time) on a dataset.

Works with any tracker config. Each tracker starts from the first box where the target is visible
and then tracks every following frame, with no re-initialization and no scoring, so all trackers
see the same frames. Decoding and initialize() are not timed. Videos run one at a time because
parallel workers would compete for the CPU, and with several configs the order alternates per
video, so a device slowing down over the run (heat, power) affects them about equally.

Run from src/:
    uv run python -m evaluation.benchmark_speed `
        --model_config siamban_mnv4_fpn_tower_onnx.toml siamban_efficientnet_fpn_tower_onnx.toml `
        --dataset mmfw --dataset_path C:/Users/yevhe/PhDProjects/datasets/MMFW-UAV-Processed/Wide-Angle-Sensor-Subset

Speed depends on frame size and target size, so use footage close to the deployment resolution.
ONNX Runtime threads come from [params] of the config; find the best value with
benchmark_onnx_threads.py.
"""

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.dataset_factory import create_dataset
from models.trackers.tracker_factory import create_tracker
from utils.config import load_config


def load_tracker(config_name: str, device: str | None):
    config = load_config(config_name)
    params = config.setdefault('params', {})
    device = device or params.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # The checkpoint replaces the backbone's weights, so downloading pretrained ones is wasted.
    if 'weights' in config and isinstance(config.get('backbone'), dict):
        config['backbone']['pretrained'] = False

    tracker = create_tracker(config, device=device)
    return tracker, getattr(tracker, 'device', 'cpu') == 'cuda'


def first_visible_box(video):
    """(frame index, box) of the first annotation with a non-empty box, or None."""
    for frame_idx, box in video.gt_rects:
        if box is not None and box[2] > 0 and box[3] > 0:
            return int(frame_idx), box
    return None


def time_track(tracker, frame, sync: bool) -> float:
    """Milliseconds for one track() call, waiting for the GPU's queued work on both sides."""
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    tracker.track(frame)
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def warm_up(tracker, video, start: int, box, steps: int, sync: bool):
    """Untimed steps, since the first calls are slow (lazy allocation, GPU kernels, caches)."""
    frame = video.source[start]
    tracker.initialize(frame, box)
    for _ in range(steps):
        time_track(tracker, frame, sync)


def benchmark_video(tracker, video, start: int, box, max_frames: int, sync: bool) -> list[float]:
    """Frames are decoded one at a time, so a 4K video does not have to fit in RAM."""
    tracker.initialize(video.source[start], box)
    last = min(start + max_frames, len(video.source))
    return [time_track(tracker, video.source[i], sync) for i in range(start + 1, last)]


def report(name: str, per_video: list[list[float]]):
    times = np.concatenate(per_video)
    medians = [np.median(v) for v in per_video]
    median = np.median(times)
    print(f"\n{name}")
    print(f"  {len(times)} frames, {len(per_video)} videos")
    print(f"  median {median:7.2f} ms -> {1000 / median:6.1f} FPS")
    print(f"  mean   {times.mean():7.2f} ms, p95 {np.percentile(times, 95):.2f} ms")
    print(f"  per-video median: {min(medians):.2f} to {max(medians):.2f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_config', nargs='+', required=True, help="One or more tracker configs.")
    parser.add_argument('--dataset', required=True,
                        choices=['synthetic', 'uav123', 'manual', 'visdrone', 'vot', 'mmfw'])
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--max_frames', type=int, default=120, help="Frames per video, counted from the first box.")
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                        help="Default: the config's device, or cpu if it has none or no GPU is available. "
                             "ONNX Runtime and OpenCV trackers pick their own.")
    parser.add_argument('--warmup', type=int, default=20, help="Untimed track() steps per tracker.")
    args = parser.parse_args()

    videos = create_dataset(args.dataset, args.dataset_path).parse()

    # A video needs a visible box and at least one frame after it.
    runnable = []
    for video in videos:
        start_box = first_visible_box(video)
        if start_box and start_box[0] + 1 < len(video.source):
            runnable.append((video, *start_box))
    if not runnable:
        raise SystemExit("No video to run: check --dataset and --dataset_path.")

    first_frame = runnable[0][0].source[runnable[0][1]]
    print(f"{platform.machine()} | {platform.processor() or platform.platform()}")
    print(f"dataset {args.dataset}: {len(runnable)} of {len(videos)} videos usable, "
          f"first frame {first_frame.shape[1]}x{first_frame.shape[0]}")

    trackers = {}
    for name in args.model_config:
        tracker, sync = load_tracker(name, args.device)
        trackers[Path(name).stem] = (tracker, sync)
        warm_up(tracker, *runnable[0], args.warmup, sync)

    results = {name: [] for name in trackers}
    names = list(trackers)
    for i, (video, start, box) in enumerate(tqdm(runnable, desc='Videos')):
        offset = i % len(names)
        for name in names[offset:] + names[:offset]:
            tracker, sync = trackers[name]
            results[name].append(benchmark_video(tracker, video, start, box, args.max_frames, sync))

    for name, per_video in results.items():
        report(name, per_video)


if __name__ == '__main__':
    main()
