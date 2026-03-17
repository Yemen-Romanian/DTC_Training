import cv2
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a dataset.')
    parser.add_argument('-d', '--dataset', type=str, help='Path to the dataset directory.')

    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    image_folder = dataset_path / "images"
    labels = dataset_path / "labels.txt"

    labels_df = pd.read_csv(labels, sep=',', header=None,
                            names=['image_index', 'class', 'x', 'y', 'w', 'h', 'confidence'])
        
    for image_file in tqdm(sorted(image_folder.glob("*.jpg")), desc="Validating images"):
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_index = int(image_file.stem)

        bbox = labels_df[labels_df['image_index'] == image_index]
        if not bbox.empty:
            x, y, w, h = bbox[['x', 'y', 'w', 'h']].values[0]
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Video from dataset", image)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
