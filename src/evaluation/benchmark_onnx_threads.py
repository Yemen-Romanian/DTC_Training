"""Find the fastest ONNX Runtime thread settings for the siamban_onnx tracker on this machine.

Times each (threads, allow_spinning) combination two ways:
  network  - backbone on a search crop + head, exemplar features cached (median ms per frame)
  track    - the full tracker.track() call on real frames, if --video_path is given; this also
             counts crop extraction and decoding, which compete with ONNX Runtime for the cores

    python -m evaluation.benchmark_onnx_threads --model_config siamban_onnx.toml
    python -m evaluation.benchmark_onnx_threads --model_config siamban_onnx.toml --video_path <video> --threads 1 2 3 4

Put the winning values into [params] of the model config. Run it on the target device itself,
with the same cooling and power setup the tracker will run under.
"""

import argparse
import os
import platform
import time

import numpy as np
import onnxruntime as ort

from models.trackers.tracker_factory import create_tracker
from utils.config import load_config
from utils.video_source import VideoSource


def time_ms(fn, runs: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    return float(np.median(times)), float(np.percentile(times, 95))


def network_step(tracker):
    rng = np.random.default_rng(0)
    z = rng.random((1, 3, 127, 127), dtype=np.float32)
    x = rng.random((1, 3, 255, 255), dtype=np.float32)
    z_feat = tracker.backbone_session.run(['features'], {'image': z})[0]

    def step():
        x_feat = tracker.backbone_session.run(['features'], {'image': x})[0]
        tracker.head_session.run(['cls', 'reg'], {'z_feat': z_feat, 'x_feat': x_feat})
    return step


def track_step(tracker, frames, bbox):
    tracker.initialize(frames[0], bbox)
    index = iter(range(10**9))

    def step():
        # Cycle the frames; re-initializing at each wrap keeps the target where the bbox is.
        i = next(index) % (len(frames) - 1) + 1
        if i == 1:
            tracker.initialize(frames[0], bbox)
        tracker.track(frames[i])
    return step


def load_frames(video_path, count):
    source = VideoSource(video_path)
    return [source[i] for i in range(min(count, len(source)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_config', default='siamban_onnx.toml')
    parser.add_argument('--threads', type=int, nargs='+', default=None,
                        help="Thread counts to try (0 = ONNX Runtime default). Default: 0, 1, 2, ... up to the core count, at most 8.")
    parser.add_argument('--video_path', help="Video or frame folder for timing the full track() call.")
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                        help="Initial box on the first frame. Default: a box in the frame centre.")
    parser.add_argument('--frames', type=int, default=200, help="Frames to load from the video.")
    parser.add_argument('--runs', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=30)
    args = parser.parse_args()

    model_config = load_config(args.model_config)
    threads_list = args.threads or [0] + list(range(1, min(os.cpu_count() or 1, 8) + 1))

    frames, bbox = None, None
    if args.video_path:
        frames = load_frames(args.video_path, args.frames)
        h, w = frames[0].shape[:2]
        bbox = args.bbox or [w * 0.45, h * 0.45, w * 0.1, h * 0.1]

    print(f"{platform.machine()} | {platform.processor() or platform.platform()} | {os.cpu_count()} logical cores "
          f"| onnxruntime {ort.__version__}")
    print(f"models: {model_config.get('backbone_weights')}, {model_config.get('head_weights')}")
    if frames:
        print(f"video: {frames[0].shape[1]}x{frames[0].shape[0]}, {len(frames)} frames")
    header = f"{'threads':>7} {'spin':>5} | {'network ms':>10} {'p95':>6} {'FPS':>5}"
    if frames:
        header += f" | {'track ms':>8} {'p95':>6} {'FPS':>5}"
    print(header)
    print('-' * len(header))

    results = []
    for threads in threads_list:
        # Spinning only matters when there are worker threads to spin.
        for spin in ((True, False) if threads != 1 else (True,)):
            config = {**model_config, 'params': {**model_config.get('params', {}),
                                                 'threads': threads, 'allow_spinning': spin}}
            tracker = create_tracker(config)
            net, net_p95 = time_ms(network_step(tracker), args.runs, args.warmup)
            row = f"{threads or 'auto':>7} {'on' if spin else 'off':>5} | {net:>10.2f} {net_p95:>6.2f} {1000 / net:>5.0f}"
            score = net
            if frames:
                trk, trk_p95 = time_ms(track_step(tracker, frames, bbox), args.runs, args.warmup)
                row += f" | {trk:>8.2f} {trk_p95:>6.2f} {1000 / trk:>5.0f}"
                score = trk
            print(row)
            results.append((score, threads, spin))

    score, threads, spin = min(results)
    measured = 'track()' if frames else 'network'
    print(f"\nFastest by {measured}: threads = {threads}, allow_spinning = {str(spin).lower()} ({score:.2f} ms)")


if __name__ == '__main__':
    main()
