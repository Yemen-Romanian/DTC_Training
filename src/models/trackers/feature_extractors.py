import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms

class AlexNetFeatureExtractor(nn.Module):
    """Feature extractor that was proposed in Bertinetto et.al paper"""
    
    def __init__(self):
        super(AlexNetFeatureExtractor, self).__init__()
        self.stride = 8 # Networks stride
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128)
        )
    def forward(self, x):
        return self.feature_extractor(x)
    

class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, device='cpu', freeze_weights=True, pretrained=True):
        super(MobileNetV3FeatureExtractor, self).__init__()
        self.device = device
        self.stride = 8 # Networks stride
        self.freeze_weights = freeze_weights
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None).to(self.device)
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        feature_layer_idx = 9
        self.feature_extractor = self.model.features[:feature_layer_idx]
        self.neck = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        ).to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_weights:
            for m in self.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        x_preprocessed = self.preprocess(x)
        features = self.feature_extractor(x_preprocessed)
        features = self.neck(features)
        return features
