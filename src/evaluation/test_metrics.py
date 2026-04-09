import pandas as pd
from metrics import match_boxes, calculate_metrics

def test_match_boxes1():
    df1 = pd.DataFrame([{"x1": 1, "y1": 1, 'w': 10, 'h': 10}])
    df2 = pd.DataFrame([{"x1": 3, "y1": 3, 'x2': 13, 'y2': 13}])

    tp, fp, fn, _ = match_boxes(df1, df2)
    assert (tp, fp, fn) == (1,0,0)


def test_match_boxes2():
    df1 = pd.DataFrame([
        {"x1": 1, "y1": 1, 'w': 10, 'h': 10},
        {"x1": 100, "y1": 1, 'w': 10, 'h': 10},
    ])
    df2 = pd.DataFrame([
        {"x1": 1, "y1": 1, 'x2': 10, 'y2': 10},
        {"x1": 1, "y1": 100, 'x2': 10, 'y2': 110},
    ])

    tp, fp, fn, _ = match_boxes(df1, df2)
    assert (tp, fp, fn) == (1,1,1)


def test_match_boxes3():
    df1 = pd.DataFrame([
        {"x1": 1, "y1": 1, 'w': 10, 'h': 10},
        {"x1": 100, "y1": 1, 'w': 10, 'h': 10},
    ])
    df2 = pd.DataFrame()

    tp, fp, fn, _ = match_boxes(df1, df2)
    assert (tp, fp, fn) == (0,0,2)


def test_calculate_metrics():
    df1 = pd.DataFrame([
        {'frame_id': 1, "x1": 1, "y1": 1, 'w': 10, 'h': 10},
        {'frame_id': 1, "x1": 100, "y1": 1, 'w': 10, 'h': 10},
        {'frame_id': 10, "x1": 1, "y1": 1, 'w': 10, 'h': 10},
        {'frame_id': 10, "x1": 100, "y1": 1, 'w': 10, 'h': 10},
    ])
    df2 = pd.DataFrame([
        {'frame_id': 1, "x1": 1, "y1": 1, 'x2': 10, 'y2': 10},
        {'frame_id': 1, "x1": 1, "y1": 100, 'x2': 10, 'y2': 110},
        {'frame_id': 10, "x1": 1, "y1": 1, 'x2': 10, 'y2': 10},
        {'frame_id': 10, "x1": 1, "y1": 100, 'x2': 10, 'y2': 110},
    ])

    tp, fp, fn, prec, recall = calculate_metrics(df1, df2)
    assert (tp, fp, fn) == (2,2,2)
