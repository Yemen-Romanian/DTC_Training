from datasets.synthetic_dataset import SyntheticDataset
from datasets.manual_uav_dataset import ManualUAVDataset
from datasets.uav123_dataset import UAV123Dataset


class MixedDataset:
    """
    Composition of several datasets (Synthetic, UAV123, Manual).
    To actually parse and get videos use parse() method.

    paths_dict: dict -- names and paths for each dataset. Possible keys: 'synthetic', 'uav123', 'manual'
    """
    def __init__(self, paths_dict: dict):
        self._videos = list()
        self._paths_dict = paths_dict

    def parse(self):
        """
        Parse datasets from different sources, defined by paths_dict passed to constructor.   
        """
        print(self._paths_dict)
        for name, path in self._paths_dict.items():
            if name == 'synthetic':
                dataset = SyntheticDataset(path, csv_name="labels.txt")
            elif name == 'uav123':
                dataset = UAV123Dataset(path)
            elif name == 'manual':
                dataset = ManualUAVDataset(path, video_extension='.mp4')
            else:
                raise ValueError(f'Unsupported dataset {name}. Possible values: synthetic, uav123, manual')

            videos = dataset.parse()
            self._videos.extend(videos)
        
        return self._videos

    def __len__(self):
        return len(self._videos)
    
    def __getitem__(self, i):
        if i >= len(self._videos):
            raise IndexError(f"Index Out Of Bounds {i} with number of videos {len(self._videos)}")
        return self._videos[i]
