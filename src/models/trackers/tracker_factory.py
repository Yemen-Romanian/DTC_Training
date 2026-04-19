import torch

from models.trackers.siamfc import SiamFCNet, TrackerSiamFC
from models.trackers.feature_extractors import AlexNetFeatureExtractor


def create_tracker(name: str, state_dict: str|dict = None, device: str = 'cpu'):
    if name == 'siamfc':
        model = SiamFCNet(AlexNetFeatureExtractor())
        if isinstance(state_dict, str):
            weights = torch.load(state_dict)
            model.load_state_dict(weights)
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            raise TypeError(f"Inappropriate type for model state dict. Expected str path or dictionary, found {type(state_dict)}")
        
        tracker = TrackerSiamFC(model, device)
    else:
        raise ValueError(f"Unknown tracker type {name}. Available trackers: siamfc")
    
    return tracker
