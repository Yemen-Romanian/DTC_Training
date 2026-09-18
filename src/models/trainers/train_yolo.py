"""Train a YOLO detector on the 4-class synthetic UAV dataset.

Run from src/:
    python -m models.trainers.train_yolo --epochs 50
    python -m models.trainers.train_yolo --epochs 1 --fraction 0.05
"""

import argparse
from multiprocessing import freeze_support

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from utils.paths import Paths

DEFAULT_DATA = r"..\..\datasets\Synthetic\train_new\yolo_dataset\data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15,
                        help="early stop after N epochs without val improvement")
    # 640 halves the long side of a 1280x720 frame, dropping the mean target from
    # ~99x37 px to ~50x18 and pushing the smallest third under 16 px.
    parser.add_argument("--imgsz", type=int, default=960)
    # Float batch = AutoBatch fraction of VRAM, which suits an 8 GiB card better
    # than a hardcoded integer that has to be retuned whenever imgsz changes.
    parser.add_argument("--batch", type=float, default=0.85)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of the train split to use; 0.05 for a quick smoke run")
    parser.add_argument("--device", default="0", help="'0' for first CUDA device, or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="yolo11n", help="run name under output/detection/")
    return parser.parse_args()


def disable_ultralytics_mlflow():
    """Turn off ultralytics' built-in MLflow logging.

    It switches itself on whenever mlflow is importable and defaults the tracking URI
    to a local file store, which recent MLflow versions refuse outright ("filesystem
    tracking backend is in maintenance mode") — so leaving it on crashes training.

    dict.__setitem__ bypasses ultralytics' JSONDict, whose __setitem__/update persist
    to the user's global settings file; this switch should last only for this process.
    """
    dict.__setitem__(SETTINGS, "mlflow", False)


def main():
    args = parse_args()
    disable_ultralytics_mlflow()

    # batch is int-or-float in ultralytics: >=1 means a literal batch size,
    # 0.0-1.0 means AutoBatch targeting that fraction of GPU memory.
    batch = int(args.batch) if args.batch >= 1 else args.batch

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=batch,
        fraction=args.fraction,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        single_cls=False,
        project=str(Paths.output_dir() / "detection"),
        name=args.name,
    )
    # No model.val() here: training already validates every epoch and reports the
    # best checkpoint. Re-running it on the same split only duplicates numbers.
    print("============ Training finished! =================")


if __name__ == "__main__":
    freeze_support()
    main()
