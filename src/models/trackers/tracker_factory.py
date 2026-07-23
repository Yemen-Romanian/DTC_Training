import logging
from pathlib import Path

import torch

from models.trackers.siamfc import TrackerSiamFC, SiamFCNet
from models.trackers.siamban import TrackerSiamBAN, SiamBANNet
from models.trackers.nano import TrackerNano
from models.trackers.vit import TrackerViT
from utils.paths import Paths

logger = logging.getLogger(__name__)


def _load_weights(model, state_dict, device):
    if isinstance(state_dict, str):
        model.load_state_dict(torch.load(state_dict, map_location=device))
    elif isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    elif state_dict is None:
        logger.warning(
            "No weights were given and the config has no 'weights' entry: the tracker keeps its "
            "randomly initialized head. Any evaluation of it is meaningless."
        )
    else:
        raise TypeError(f"Expected str path or dict for state_dict, got {type(state_dict)}")


def resolve_state_dict(model_config, state_dict=None):
    """Fall back to the config's `weights` entry, resolved like demo.py resolves it.

    An absolute path is used as is, a bare file name is looked up in output/model_weights.
    Returns None when neither a state dict nor a `weights` entry is given.
    """
    if state_dict is not None:
        return state_dict

    weights = model_config.get('weights') or model_config.get('params', {}).get('weights')
    if weights is None:
        return None

    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = Paths.model_weights_dir() / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Config lists weights '{weights}' but no such file exists (looked at {weights_path})."
        )

    logger.info(f"Loading tracker weights from {weights_path}")
    return str(weights_path)


def create_tracker(model_config: dict, state_dict: str | dict = None, device: str = None):
    model_id = model_config['id']
    device = device or model_config.get('params', {}).get('device', 'cpu')
    state_dict = resolve_state_dict(model_config, state_dict)

    if model_id == 'siamfc':
        model = SiamFCNet.from_config(model_config)
        _load_weights(model, state_dict, device)
        tracker = TrackerSiamFC(model, device)

    elif model_id == 'siamban':
        model = SiamBANNet.from_config(model_config)
        _load_weights(model, state_dict, device)
        tracker = TrackerSiamBAN(model, device)

    elif model_id == 'nano':
        tracker = TrackerNano(device=device)
    elif model_id == 'vit':
        tracker = TrackerViT()
    else:
        raise ValueError(f"Unknown tracker type {model_id}. Available trackers: siamfc, nano")

    return tracker
