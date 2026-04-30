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
    def __init__(self, device='cpu', freeze_weights=True):
        super(MobileNetV3FeatureExtractor, self).__init__()
        self.device = device
        self.stride = 16 # Networks stride
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(self.device)
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        feature_layer_idx = 10
        self.feature_extractor = self.model.features[:feature_layer_idx]
        
    def forward(self, x):
        x_preprocessed = self.preprocess(x)
        features = self.feature_extractor(x_preprocessed)
        return features

