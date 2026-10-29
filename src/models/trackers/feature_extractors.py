from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision import transforms

class AlexNetFeatureExtractor(nn.Module):
    """Feature extractor that was proposed in Bertinetto et.al paper"""

    def __init__(self):
        super(AlexNetFeatureExtractor, self).__init__()
        self.stride = 8 # Networks stride
        self.out_channels = 128  # plain int, not a parameter - state_dict is unaffected

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
        self.out_channels = 48  # plain int, not a parameter - state_dict is unaffected
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


class MobileNetV3MultiLayerFeatureExtractor(nn.Module):
    """MobileNetV3-Small tapped at several depths, fused into one feature map.

    The single-tap extractor above reads features[:9], which runs at stride 16, and its neck
    upsamples x2 to reach the 32x32 search map the 17x17 response needs. That resolution is
    interpolated rather than measured: the head decodes stride-8 positions from features that
    only ever resolved stride 16.

    Tapping features[:4] as well adds a branch at *true* stride 8. In MobileNetV3-Small only
    features[2:4] run at stride 8, so features[:4] (24 channels) is the deepest such tap
    available - and it needs no upsampling at all, since it is natively 16x16 for a 127px
    exemplar and 32x32 for a 255px search crop.

    Each branch is projected to a common width by its own neck, then combined by a softmax over
    one learnable weight per branch. The fused width therefore stays independent of how many
    taps are used, so BANHead is unaffected. The learned weights are also a readable diagnostic
    after training: they say how much the network actually leans on each scale.

    Unlike MobileNetV3FeatureExtractor this does not retain the parent mobilenet as a submodule.
    Doing so puts the unused classifier and deep stages - plus a duplicate copy of the used
    stages - into every checkpoint, which accounted for 86% of the saved file.
    """

    TARGET_STRIDE = 8

    def __init__(self, taps=(4, 9), out_channels=48, freeze_weights=True, pretrained=True):
        super().__init__()

        self.taps = tuple(taps)
        if len(self.taps) < 2:
            raise ValueError(f"Expected at least two taps, got {self.taps}")
        if list(self.taps) != sorted(set(self.taps)):
            raise ValueError(f"Taps must be strictly increasing and unique, got {self.taps}")

        self.stride = self.TARGET_STRIDE
        self.out_channels = out_channels
        self.freeze_weights = freeze_weights

        # Local, not an attribute: only the slices below get registered as submodules, so the
        # unused deep stages and the classifier never reach the state dict.
        mnet = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        bounds = (0,) + self.taps
        self.stages = nn.ModuleList([
            nn.Sequential(*mnet.features[start:end])
            for start, end in zip(bounds[:-1], bounds[1:])
        ])

        if freeze_weights:
            for param in self.stages.parameters():
                param.requires_grad = False

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        stage_channels, upsample_factors = self._probe_stages()
        self.necks = nn.ModuleList([
            self._build_neck(channels, out_channels, factor)
            for channels, factor in zip(stage_channels, upsample_factors)
        ])

        # Zeros -> softmax gives every branch equal weight at the start.
        self.fusion_weights = nn.Parameter(torch.zeros(len(self.stages)))

    @torch.no_grad()
    def _probe_stages(self, probe_size: int = 255):
        """Measure each tap's width and grid size with one dummy forward.

        Derived rather than hardcoded so the numbers cannot go stale when the taps change.
        The stages are forced to eval first: at construction the module is in train mode, and
        pushing a dummy batch through BatchNorm in train mode would overwrite the pretrained
        running statistics.
        """
        was_training = self.stages.training
        self.stages.eval()
        try:
            activation = torch.zeros(1, 3, probe_size, probe_size)
            channels, sizes = [], []
            for stage in self.stages:
                activation = stage(activation)
                channels.append(activation.shape[1])
                sizes.append(activation.shape[-1])
        finally:
            self.stages.train(was_training)

        # Taps deepen monotonically, so the first one is the finest grid and sets the target.
        target = sizes[0]

        # The shallowest tap fixes the fused stride, and everything downstream assumes 8: the
        # dataset builds 17x17 targets, and TrackerSiamBAN.STRIDE converts response offsets to
        # pixels with it. A tap at stride 4 still aligns with the others by integer upsampling,
        # so without this the only symptom would be a silently wrong response size.
        expected = -(-probe_size // self.TARGET_STRIDE)  # ceil division
        if target != expected:
            raise ValueError(
                f"Shallowest tap features[:{self.taps[0]}] produces a {target}x{target} grid for a "
                f"{probe_size}px input, i.e. stride {probe_size / target:.3g}, but the fused output "
                f"must be stride {self.TARGET_STRIDE} ({expected}x{expected}). A different stride "
                f"changes the correlation response away from the 17x17 the training targets expect."
            )

        factors = []
        for tap, size in zip(self.taps, sizes):
            if size == 0 or target % size != 0:
                raise ValueError(
                    f"Tap features[:{tap}] produces a {size}x{size} grid, which does not divide "
                    f"the {target}x{target} grid of the shallowest tap features[:{self.taps[0]}]. "
                    f"The branches cannot be aligned by integer upsampling."
                )
            factors.append(target // size)
        return channels, factors

    @staticmethod
    def _build_neck(in_channels: int, out_channels: int, upsample_factor: int) -> nn.Sequential:
        """Project one branch to the fused width, upsampling only if that branch is coarser."""
        layers = []
        if upsample_factor > 1:
            layers.append(nn.Upsample(scale_factor=upsample_factor, mode='nearest'))
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_weights:
            # Every frozen stage, not just the first - the necks and fusion weights stay trainable.
            for module in self.stages.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def branch_weights(self) -> torch.Tensor:
        """Normalized per-branch contribution, in the same order as `taps`."""
        return torch.softmax(self.fusion_weights, dim=0)

    def forward(self, x):
        activation = self.preprocess(x)
        weights = self.branch_weights()

        fused = None
        for index, (stage, neck) in enumerate(zip(self.stages, self.necks)):
            activation = stage(activation)
            branch = neck(activation) * weights[index]
            fused = branch if fused is None else fused + branch
        return fused


class MobileNetV3FPNFeatureExtractor(nn.Module):
    """MobileNetV3-Small tapped at several depths, merged top-down by a feature pyramid.

    The default taps are C3, C4 and C5 - the last layer before each stride change:
    features[:4] (stride 8, 24 ch), features[:9] (stride 16, 48 ch), features[:12]
    (stride 32, 96 ch). features[12], the 1x1 expansion to 576 channels, is left out: it is
    sized for the ImageNet classifier.

    Unlike MobileNetV3MultiLayerFeatureExtractor, the branches do not compete through one
    scalar weight each. The FPN adds every deeper level, upsampled, into the 1x1-projected level
    above it, then smooths each sum with a 3x3 conv:

        P3 = conv3x3(lat(C3) + up(lat(C4) + up(lat(C5))))

    Only P3 is returned. It sits at true stride 8 - 16x16 for a 127px exemplar, 32x32 for a 255px
    search crop - so BANHead and the 17x17 targets are unaffected, and it carries the deep
    semantics at every position instead of letting the network turn the fine branch down.

    `dilated=True` builds the backbone with torchvision's dilated last stage (stride 1,
    dilation 2), which keeps C5 at stride 16: 8x8 on the exemplar instead of a 4x4 map that is
    almost all padding. The pretrained weights load unchanged, since dilation adds no parameters.

    `norm_layer` is passed to the FPN's lateral and output convs (torchvision's default: none).
    Normalizing keeps the lateral and top-down branches in comparable ranges, which matters for
    int8: each FPN sum is quantized with a single scale, so an unbounded branch would crush the
    other one's precision.
    """

    TARGET_STRIDE = 8

    def __init__(self, taps=(4, 9, 12), out_channels=48, freeze_weights=True, pretrained=True,
                 dilated=False, norm_layer=None):
        super().__init__()

        self.taps = tuple(taps)
        if len(self.taps) < 2:
            raise ValueError(f"Expected at least two taps, got {self.taps}")
        if list(self.taps) != sorted(set(self.taps)):
            raise ValueError(f"Taps must be strictly increasing and unique, got {self.taps}")

        self.stride = self.TARGET_STRIDE
        self.out_channels = out_channels
        self.freeze_weights = freeze_weights

        # Local, not an attribute, for the same checkpoint-size reason as the multi-layer
        # extractor: only the used slices are registered.
        mnet = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            dilated=dilated,
        )
        bounds = (0,) + self.taps
        self.stages = nn.ModuleList([
            nn.Sequential(*mnet.features[start:end])
            for start, end in zip(bounds[:-1], bounds[1:])
        ])

        if freeze_weights:
            for param in self.stages.parameters():
                param.requires_grad = False

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.fpn = FeaturePyramidNetwork(in_channels_list=self._probe_channels(),
                                         out_channels=out_channels, norm_layer=norm_layer)

    @torch.no_grad()
    def _probe_channels(self, probe_size: int = 255) -> list:
        """Measure each tap's width with one dummy forward, and check the finest one is stride 8.

        Stages are forced to eval first so the dummy batch does not overwrite the pretrained
        BatchNorm running statistics.
        """
        was_training = self.stages.training
        self.stages.eval()
        try:
            activation = torch.zeros(1, 3, probe_size, probe_size)
            channels, sizes = [], []
            for stage in self.stages:
                activation = stage(activation)
                channels.append(activation.shape[1])
                sizes.append(activation.shape[-1])
        finally:
            self.stages.train(was_training)

        # P3 inherits the grid of the shallowest tap, and everything downstream assumes stride 8.
        expected = -(-probe_size // self.TARGET_STRIDE)  # ceil division
        if sizes[0] != expected:
            raise ValueError(
                f"Shallowest tap features[:{self.taps[0]}] produces a {sizes[0]}x{sizes[0]} grid for "
                f"a {probe_size}px input, but P3 must be stride {self.TARGET_STRIDE} "
                f"({expected}x{expected}) to keep the 17x17 correlation response."
            )
        return channels

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_weights:
            # Only the backbone stages are frozen; the FPN always trains.
            for module in self.stages.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        activation = self.preprocess(x)
        levels = OrderedDict()
        for index, stage in enumerate(self.stages):
            activation = stage(activation)
            levels[f"c{index}"] = activation
        # The FPN returns one map per level under the input keys; "c0" is P3, the finest.
        return self.fpn(levels)["c0"]
