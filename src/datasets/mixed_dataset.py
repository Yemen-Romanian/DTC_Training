from datasets.dataset_factory import create_dataset


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
        for name, path in self._paths_dict.items():
            dataset = create_dataset(name, path)
            videos = dataset.parse()
            self._videos.extend(videos)
        
        return self._videos

    def __len__(self):
        return len(self._videos)
    
    def __getitem__(self, i):
        if i >= len(self._videos):
            raise IndexError(f"Index Out Of Bounds {i} with number of videos {len(self._videos)}")
        return self._videos[i]
