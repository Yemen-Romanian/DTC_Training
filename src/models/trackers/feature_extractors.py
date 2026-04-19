import torch.nn as nn

class AlexNetFeatureExtractor(nn.Module):
    """Feature extractor that was proposed in Bertinetto et.al paper"""
    
    def __init__(self):
        super(AlexNetFeatureExtractor, self).__init__()
        
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
    
    
# fe = AlexNetFeatureExtractor()
# x = torch.randn((1, 3, 255, 255))
# z = torch.randn((1, 3, 127, 127))

# fx = fe(x)
# fz = fe(z)

# print(fx.shape)
# print(fz.shape)
