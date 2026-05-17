import torch

from models.trackers.siamfc import TrackerSiamFC, SiamFCNet
from models.trackers.nano import TrackerNano
from models.trackers.vit import TrackerViT


def create_tracker(model_config: dict, state_dict: str|dict = None, device: str = None):
    model_id = model_config['id']
    device = device or model_config.get('params', {}).get('device', 'cpu')

    if model_id == 'siamfc':
        model = SiamFCNet.from_config(model_config)
        if isinstance(state_dict, str):
            weights = torch.load(state_dict, map_location=device)
            model.load_state_dict(weights)
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        elif state_dict is not None:
            raise TypeError(f"Inappropriate type for model state dict. Expected str path or dictionary, found {type(state_dict)}")
        
        tracker = TrackerSiamFC(model, device)

    elif model_id == 'nano':
        tracker = TrackerNano(device=device)
    elif model_id == 'vit':
        tracker = TrackerViT()
    else:
        raise ValueError(f"Unknown tracker type {model_id}. Available trackers: siamfc, nano")
    
    return tracker
