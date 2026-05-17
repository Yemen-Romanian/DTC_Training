from models.abstract_trainable import AbstractTrainable
from models.trackers.siamfc import TrainableSiamFC


def create_trainable(model_config: dict) -> AbstractTrainable:
    """Creates trainable wrapper around raw nn.Module net that is used by Trainer"""
    model_id = model_config['id']
    if model_id == 'siamfc':
        return TrainableSiamFC(model_config)
    else:
        raise ValueError(f"Unsupported model id: {model_id}")