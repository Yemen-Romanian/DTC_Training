import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import argparse

@dataclass
class TrackingSampleInfo:
    is_valid: bool
    has_occlusions: bool
    occlusion_percentage: float
    number_of_frames: int

def verify_synthetic_dataset(folder: Path) -> Mapping[str, TrackingSampleInfo]:
    result = {}
    for full_path in folder.rglob('labels.txt'):
        print(f"Analyzing {full_path.parent.name}")
        sample_name = full_path.parent.name
        images_folder = full_path.parent / "images"
        list_of_images = list(images_folder.glob("*.jpg"))
        number_of_images = len(list_of_images)
        labels_df = pd.read_csv(
            full_path,
            header=None,
            names=["image_idx", "x", "y", "w", "h"],
            usecols=[0,2,3,4,5]
        )
        number_of_gt_frames = len(labels_df)
        number_of_occlusions = (labels_df[['x', 'y', 'w', 'h']] == 0).all(axis=1).sum()
        result[sample_name] = TrackingSampleInfo(
            is_valid=number_of_images==number_of_gt_frames,
            number_of_frames=number_of_images,
            has_occlusions=number_of_occlusions > 0,
            occlusion_percentage=number_of_occlusions/number_of_gt_frames * 100.0
        )

    return result


def verify_uav123_dataset(folder: Path):
    result = {}
    annotation_root = folder / "anno"
    images_root = folder / "data_seq" / "UAV123"

    for sample_image_folder in images_root.iterdir():
        if not sample_image_folder.is_dir():
            continue

        sample_name = sample_image_folder.name
        list_of_images = list(sample_image_folder.glob("*.jpg"))
        number_of_images = len(list_of_images)
        annotation_file = list(annotation_root.rglob(f"{sample_name}.txt"))
        if len(annotation_file) > 0:
            print(f"Found annotation: {annotation_file[0]}")
            labels_df = pd.read_csv(annotation_file[0], header=None, names=["x", "y", "w", "h"])
            labels_df = labels_df.fillna(0)
            number_of_gt_frames = len(labels_df)
            number_of_occlusions = (labels_df == 0).all(axis=1).sum()
            result[sample_name] = TrackingSampleInfo(
                is_valid=number_of_images==number_of_gt_frames,
                number_of_frames=number_of_images,
                has_occlusions=number_of_occlusions > 0,
                occlusion_percentage=number_of_occlusions/number_of_gt_frames * 100.0
            )
        else:
            print(f"No annotations for {sample_name}")
            result[sample_name] = TrackingSampleInfo(
                is_valid=False,
                number_of_frames=number_of_images,
                has_occlusions=True,
                occlusion_percentage=100.0
            )


    return result



def verify_visdrone_dataset(folder: Path):
    result = {}

    for ann_folder in folder.rglob("annotations"):
        print(f"{ann_folder}")
        seq_folder = ann_folder.parent / "sequences"
        print(seq_folder)
        for annotation_file in ann_folder.rglob("*.txt"):
            sample_name = annotation_file.stem
            print(f"Processing {sample_name}")
            images_folder = seq_folder / sample_name
            list_of_images = list(images_folder.glob("*.jpg"))
            number_of_images = len(list_of_images)
            labels_df = pd.read_csv(annotation_file, header=None, names=["x", "y", "w", "h"])
            labels_df = labels_df.fillna(0)
            number_of_gt_frames = len(labels_df)
            number_of_occlusions = (labels_df == 0).all(axis=1).sum()
            result[sample_name] = TrackingSampleInfo(
                is_valid=number_of_images==number_of_gt_frames,
                number_of_frames=number_of_images,
                has_occlusions=number_of_occlusions > 0,
                occlusion_percentage=number_of_occlusions/number_of_gt_frames * 100.0
            )

    return result

def verify_manual_dataset(folder: Path):
    """To be implemeted"""
    return {}

def main():
    parser = argparse.ArgumentParser(description="Verification statistics for a dataset")
    parser.add_argument('--datapath', type=str, required=True, help="Path to dataset")
    parser.add_argument('--data_type', type=str, required=False, choices=['synthetic', 'manual', 'uav123', 'visdrone'], help="Type of the dataset")
    args = parser.parse_args()

    dataset_path = Path(rf"{args.datapath}")
    datatype = args.data_type
    if datatype == 'synthetic':
        stats = verify_synthetic_dataset(dataset_path)
    elif datatype == 'uav123':
        stats = verify_uav123_dataset(dataset_path)
    elif datatype == 'visdrone':
        stats = verify_visdrone_dataset(dataset_path)
    elif datatype == 'manual':
        stats = verify_manual_dataset(dataset_path)
    else:
        raise ValueError(f"Invalid data type: {datatype}. Choose from ['synthetic', 'manual', 'uav123', 'visdrone']")

    invalid_sample = {sample: info for sample, info in stats.items() if not info.is_valid}
    samples_with_no_occlusion = {sample: info for sample, info in stats.items() if not info.has_occlusions}
    occlusion_samples = {sample: info for sample, info in stats.items() if info.has_occlusions}
    print("Invalid samples: ", list(invalid_sample.keys()))
    print("Samples with occlusions: ", list(occlusion_samples.keys()))
    print("Overall number of samples: ", len(stats))


if __name__ == '__main__':
    main()
