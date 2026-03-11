import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trackers.feature_extractors import AlexNetFeatureExtractor

class SiamFCNet(nn.Module):
    def __init__(self, backbone):
        super(SiamFCNet, self).__init__()
        self.backbone = backbone
        self.scale_step = nn.Parameter(torch.ones(1)*0.001)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, z, x):
        """
        z: [B, 3, 127, 127] - examplar image
        x: [B, 3, 255, 255] - search image
        """
        
        z_features = self.backbone(z)
        x_features = self.backbone(x)
        
        # Batch-wise Cross Correlation
        b, c, h, w = x_features.size()
        _, _, hz, wz = z_features.size()
        
        x_features = x_features.view(1, b * c, h, w)
        z_features = z_features.view(b, c, hz, wz)
        
        score = F.conv2d(x_features, z_features, groups=b) # [1, b, 17, 17]
        score = score.view(b, 1, score.size(2), score.size(3))
        return score * self.scale_step + self.bias
    
z = torch.randn(1, 3, 127, 127)
x = torch.randn(1, 3, 255, 255)

model = SiamFCNet(backbone=AlexNetFeatureExtractor())
out = model(z, x)
print(out.shape)