import logging
from models.trackers.siamfc import SiamFCNet
from models.trackers.feature_extractors import AlexNetFeatureExtractor, MobileNetV3FeatureExtractor

logger = logging.getLogger(__name__)

def create_model(model_config: dict):
    """
    Creates the PyTorch model (nn.Module) based on the provided configuration.
    """
    model_id = model_config['id']
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

    if model_id == 'siamfc':
        model = SiamFCNet(backbone)
    else:
        raise ValueError(f"Unsupported model id: {model_id}")

    logger.info(f"Successfully created model {model_id}")
    return model
