import logging
from models.trackers.siamfc import SiamFCNet
from models.trackers.feature_extractors import AlexNetFeatureExtractor, MobileNetV3FeatureExtractor
from models.abstract_trainable import AbstractTrainable
from models.trackers.siamfc import TrainableSiamFC

logger = logging.getLogger(__name__)

def create_net(model_config: dict):
    """
    Creates the PyTorch model (nn.Module) based on the provided configuration.
    """
    model_id = model_config['id']
    if model_id == 'siamfc':
        backbone_config = model_config.get('backbone', {})
        backbone_type = backbone_config.get('type', 'AlexNet')
        freeze_backbone = backbone_config.get('freeze', False)
        
        logger.info(f"Creating model: {model_id} with backbone: {backbone_type}")
        if freeze_backbone:
            logger.info(f"Backbone weights will be frozen.")

        # Instantiate Backbone
        if backbone_type == 'AlexNet':
            backbone = AlexNetFeatureExtractor()
        elif backbone_type == 'MobileNetV3':
            backbone = MobileNetV3FeatureExtractor(freeze_weights=freeze_backbone)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        model = SiamFCNet(backbone)
    else:
        raise ValueError(f"Unsupported model id: {model_id}")

    logger.info(f"Successfully created model {model_id}")
    return model


def create_trainable(model_config: dict) -> AbstractTrainable:
    """Creates trainable wrapper around raw nn.Module net that is used by Trainer"""
    model_id = model_config['id']
    if model_id == 'siamfc':
        return TrainableSiamFC(model_config)
    else:
        raise ValueError(f"Unsupported model id: {model_id}")

