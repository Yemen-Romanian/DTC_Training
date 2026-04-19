from datasets.synthetic_dataset import SyntheticDataset
from datasets.manual_uav_dataset import ManualUAVDataset
from datasets.uav123_dataset import UAV123Dataset

def create_dataset(name: str, path):
    if name == 'synthetic':
        dataset = SyntheticDataset(path, csv_name="labels.txt")
    elif name == 'uav123':
        dataset = UAV123Dataset(path)
    elif name == 'manual':
        dataset = ManualUAVDataset(path, video_extension='.mp4')
    else:
        raise ValueError(f'Unsupported dataset {name}. Possible values: synthetic, uav123, manual')
    
    return dataset
