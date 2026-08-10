"""
Utility functions for MMFW-UAV Dataset
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

def mmfw_xml_gt_to_txt(dataset_folder: Path):
    """
    dataset_folder should contain following dirs: Annotations (.xml for each image)
    and Sequences (images grouped into separate videos)
    """
    seq_root = dataset_folder / "Sequences"
    ann_root = dataset_folder / "Annotations"
    ann_root_new = dataset_folder / "Annotations_new"
    ann_root_new.mkdir(exist_ok=True)

    for seq_folder in seq_root.iterdir():
        bboxes = []
        for image in sorted(seq_folder.glob("*.jpg")):
            print(f"Seq: {seq_folder.name}, image: {image.stem}")
            annotation_name = image.stem + ".xml"
            tree = ET.parse(ann_root / annotation_name)
            root = tree.getroot()

            for obj in root.findall("object"):
                box = obj.find("bndbox")
                xmin = int(box.find("xmin").text)
                xmax = int(box.find("xmax").text)
                ymin = int(box.find("ymin").text)
                ymax = int(box.find("ymax").text)
                x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin
                bboxes.append((x, y, w, h))

            df = pd.DataFrame(bboxes)
            df.to_csv(f"{ann_root_new / (seq_folder.name + '.txt')}", index=False, header=False)
