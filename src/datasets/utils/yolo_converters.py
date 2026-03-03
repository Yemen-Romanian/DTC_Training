from pathlib import Path
import random
import shutil

import pandas as pd
from tqdm import tqdm
from PIL import Image
import yaml

def synthetic_to_yolo_format(root_dir: Path, class_names: dict, train_ratio: float = 0.8):
    yolo_result_dir = root_dir / "yolo_dataset"
    for split in ['train', 'val']:
        (yolo_result_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_result_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    bbox_files = list(root_dir.rglob("bboxes.csv"))
    print(f"Found subdirectories with images: {len(bbox_files)}")

    for bbox_file in bbox_files:
        data_dir = bbox_file.parent
        dataset_prefix = data_dir.name
        
        df = pd.read_csv(bbox_file, header=None, names=['img_idx', 'class_id', 'x', 'y', 'w', 'h', 'unknown'])
        
        image_extensions = ('.jpg')
        all_images = [f for f in data_dir.iterdir() if f.suffix.lower() in image_extensions]

        for img_path in tqdm(all_images, desc=f"Processing {dataset_prefix}"):
            img_idx = int(img_path.stem)
            img_annots = df[df['img_idx'] == img_idx]
            if img_annots.empty:
                print(f"Empty bounding box for {img_path}, skipping")
                continue
            
            split = 'train' if random.random() < train_ratio else 'val'
            new_name_base = f"{dataset_prefix}_{img_path.stem}"
            new_img_path = yolo_result_dir / "images" / split / f"{new_name_base}{img_path.suffix}"
            new_label_path = yolo_result_dir / "labels" / split / f"{new_name_base}.txt"
            
            shutil.copy(img_path, new_img_path)
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            yolo_annots = []
            for _, row in img_annots.iterrows():
                x_center = (row['x'] + row['w'] / 2) / img_w
                y_center = (row['y'] + row['h'] / 2) / img_h
                w_norm = row['w'] / img_w
                h_norm = row['h'] / img_h
                
                yolo_annots.append(f"{max(0, int(row['class_id'])-1)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            with open(new_label_path, 'w') as f:
                f.write("\n".join(yolo_annots))
               
    # prepare YAML 
    yaml_data = {
        'path': str(yolo_result_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = yolo_result_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n--- Conversion finished ---")
    print(f"The configuration is saved into: {yaml_path}")
    print(f"Conversion finished! YOLO data directory: {yolo_result_dir}")

synthetic_to_yolo_format(Path(r"C:\Users\Zhenya\Projects\Datasets\Synthetic"), class_names={0: "drone"})