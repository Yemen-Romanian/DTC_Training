import sys
import math
from itertools import product
from pathlib import Path
import pandas as pd


def intersection_area(r1, r2):
    """
        r1, r2: tuples (x1, y1, x2, y2)
    """
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def area(r):
    width = max(0, r[2] - r[0])
    height = max(0, r[3] - r[1])
    return width * height


def union_area(r1, r2):
    return area(r1) + area(r2) - intersection_area(r1, r2)


def center_distance(r1, r2):
    x1_1, y1_1, x2_1, y2_1 = r1
    x1_2, y1_2, x2_2, y2_2 = r2
    cx1 = (x1_1 + x2_1) / 2
    cy1 = (y1_1 + y2_1) / 2
    cx2 = (x1_2 + x2_2) / 2
    cy2 = (y1_2 + y2_2) / 2
    return math.hypot(cx2 - cx1, cy2 - cy1)


def match_boxes(df1, df2):
    tp = 0
    m1 = m2 = set()
    metrics_list = []
    for (i, r1), (j, r2) in product(enumerate(df1.itertuples()), enumerate(df2.itertuples())):
        if not (i in m1 or j in m2):
            box1 = r1.x1, r1.y1, r1.x1 + r1.w, r1.y1 + r1.h
            box2 = r2.x1, r2.y1, r2.x2, r2.y2
            box1_area = area(box1)
            box2_area = area(box2)
            union = union_area(box1, box2)
            iou = intersection_area(box1, box2) / union if union > 0 else 0
            iog = intersection_area(box1, box2) / box1_area if box1_area > 0 else 0
            center_dist = center_distance(box1, box2)
            center_dist_norm = center_dist / math.sqrt(box1_area) if box1_area > 0 else 0

            print('---------------------')
            print('box1', box1, 'area', box1_area)
            print('box2', box2, 'area', box2_area)
            print('I/U', iou)
            print('I/G', iog)
            print('dist', center_dist)
            print('dist_norm', center_dist_norm)

            metrics_list.append({
                'iou': iou,
                'iog': iog,
                'center_dist': center_dist,
                'center_dist_norm': center_dist_norm
            })

            #if intersection_area(box1, box2) / union_area(box1, box2) > 0.5:
            #if intersection_area(box1, box2) / area(box1) > 0.5:
            if center_distance(box1, box2) / math.sqrt(area(box1)) < 1/3:
                m1.add(i)
                m2.add(j)
                tp += 1

    fp = len(df2) - tp
    fn = len(df1) - tp
    return tp, fp, fn, metrics_list


def calculate_metrics(df1, df2):
    """
        Returns: (tp, fp, fn, precision, recall)
    """
    tp, fp, fn = 0, 0, 0
    nframe = max(df1.iloc[-1].frame_id, df2.iloc[-1].frame_id)
    df1 = df1.set_index("frame_id")
    df2 = df2.set_index("frame_id")

    for frame in range(1, nframe + 1):
        df1_frame = df1.loc[[frame]] if frame in df1.index else df1.iloc[0:0]
        df2_frame = df2.loc[[frame]] if frame in df2.index else df2.iloc[0:0]
        tpi, fpi, fni, _ = match_boxes(df1_frame, df2_frame)
        tp += tpi
        fp += fpi
        fn += fni

        #if frame == 20: break

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return tp, fp, fn, precision, recall


def run(df1, df2):
    metrics = pd.DataFrame(columns=["conf", "tp", "fp", "fn", "precision", "recall"])

    for i in range(10, 11):
        conf = i / 10
        df2conf = df2[df2["conf"] >= conf]
        tp, fp, fn, precision, recall = calculate_metrics(df1, df2conf)
        new_row = {
            "conf":      conf,
            "tp":        tp,
            "fp":        fp,
            "fn":        fn,
            "precision": precision,
            "recall":    recall
        }
        metrics.loc[len(metrics)] = new_row

    return metrics


def main():
    if len(sys.argv) != 3:
        print("Usage: python metrics.py <ground_truth.csv> <tracking_results.csv>")
        sys.exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]

    df1 = pd.read_csv(path1, names=['frame_id', "track_id", "x1", 'y1', 'w', 'h', 'conf'])
    df2 = pd.read_csv(path2)

    metrics = run(df1, df2)
    metrics.to_csv(Path(path2).with_suffix('.metrics.3.csv'), index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
