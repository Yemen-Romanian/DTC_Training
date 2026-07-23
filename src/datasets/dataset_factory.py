from datasets.synthetic_dataset import SyntheticDataset
from datasets.manual_uav_dataset import ManualUAVDataset
from datasets.uav123_dataset import UAV123Dataset
from datasets.visdrone_dataset import VisDroneDataset
from datasets.vot_dataset import VOTDataset

def create_dataset(name: str, path):
    if name == 'synthetic':
        dataset = SyntheticDataset(path, csv_name="labels.txt")
    elif name == 'uav123':
        dataset = UAV123Dataset(path)
    elif name == 'manual':
        dataset = ManualUAVDataset(path, video_extension='.mp4')
    elif name == 'visdrone':
        dataset = VisDroneDataset(path)
    elif name == 'vot':
        dataset = VOTDataset(path)
    else:
        raise ValueError(f'Unsupported dataset {name}. Possible values: synthetic, uav123, manual, visdrone, vot')

    return dataset
