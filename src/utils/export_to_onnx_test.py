import time

import numpy as np
import onnxruntime as ort
import torch
from utils.paths import Paths
from models.trackers.tracker_factory import create_tracker


def benchmark(pt_model, ort_session, exemplar: torch.Tensor, search: torch.Tensor, runs: int = 200, warmup: int = 20):
    exemplar_np = exemplar.numpy()
    search_np = search.numpy()

    pt_model.eval()
    for _ in range(warmup):
        with torch.no_grad():
            pt_model(exemplar, search)

    pt_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_model(exemplar, search)
        pt_times.append(time.perf_counter() - t0)

    for _ in range(warmup):
        ort_session.run(['cls', 'reg'], {'exemplar': exemplar_np, 'search': search_np})

    ort_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        ort_session.run(['cls', 'reg'], {'exemplar': exemplar_np, 'search': search_np})
        ort_times.append(time.perf_counter() - t0)

    def stats(times):
        arr = np.array(times) * 1000  # ms
        return arr.mean(), arr.std()

    pt_mean, pt_std = stats(pt_times)
    ort_mean, ort_std = stats(ort_times)

    print(f"\n{'Model':<10} {'Mean (ms)':>12} {'Std (ms)':>12} {'FPS':>8}")
    print("-" * 46)
    print(f"{'PyTorch':<10} {pt_mean:>12.2f} {pt_std:>12.2f} {1000/pt_mean:>8.1f}")
    print(f"{'ONNX RT':<10} {ort_mean:>12.2f} {ort_std:>12.2f} {1000/ort_mean:>8.1f}")
    print(f"\nSpeedup: {pt_mean / ort_mean:.2f}x")


class BackboneWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        return self.backbone(image)


class HeadWrapper(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, z_feat, x_feat):
        return self.head(z_feat, x_feat)


model_config = {
    'id': 'siamban',
    'backbone': {
        'type': 'MobileNetV3',
        'freeze': True,
        'pretrained': True
    },
    'params': {
        'device': 'cpu'
    }
}
state_dict_path = str(Paths.model_weights_dir() / "siamban_synth_real_iou_0.6336_iog_0.7688_center_dist_33.2327_center_dist_norm_0.2996.pth")
tracker = create_tracker(model_config, state_dict=state_dict_path)
model = tracker.model

examplar = torch.randn((1, 3, 127, 127))
search = torch.randn((1, 3, 255, 255))

# Derive example feature tensors for the head export
model.eval()
with torch.no_grad():
    z_feat_example = model.backbone(examplar)
    x_feat_example = model.backbone(search)

backbone_path = str(Paths.model_weights_dir() / "siamban_backbone.onnx")
head_path = str(Paths.model_weights_dir() / "siamban_head.onnx")

torch.onnx.export(
    BackboneWrapper(model.backbone),
    examplar,
    backbone_path,
    input_names=['image'],
    output_names=['features'],
    dynamic_axes={
        'image':    {2: 'height', 3: 'width'},
        'features': {2: 'feat_height', 3: 'feat_width'},
    },
    opset_version=17,
)
print(f'Backbone exported to {backbone_path}')

torch.onnx.export(
    HeadWrapper(model.head),
    (z_feat_example, x_feat_example),
    head_path,
    input_names=['z_feat', 'x_feat'],
    output_names=['cls', 'reg'],
    opset_version=17,
)
print(f'Head exported to {head_path}')

# --- Verification ---
# Re-apply eval after export: torch.onnx.export's context manager restores the
# wrapper's original training=True, which propagates train(True) into the backbone
# and leaves the neck's BatchNorm in train mode despite the feature_extractor freeze.
model.eval()
with torch.no_grad():
    pt_z_feat = model.backbone(examplar)
    pt_x_feat = model.backbone(search)
    pt_cls, pt_reg = model.head(pt_z_feat, pt_x_feat)

backbone_session = ort.InferenceSession(backbone_path, providers=['CPUExecutionProvider'])
head_session = ort.InferenceSession(head_path, providers=['CPUExecutionProvider'])

ort_z_feat = backbone_session.run(['features'], {'image': examplar.numpy()})[0]  # 127×127
ort_x_feat = backbone_session.run(['features'], {'image': search.numpy()})[0]    # 255×255
ort_cls, ort_reg = head_session.run(['cls', 'reg'], {'z_feat': ort_z_feat, 'x_feat': ort_x_feat})

print(ort_z_feat.shape)
print(pt_z_feat.shape)

np.testing.assert_allclose(pt_z_feat.numpy(), ort_z_feat, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(pt_cls.numpy(), ort_cls, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(pt_reg.numpy(), ort_reg, rtol=1e-4, atol=1e-5)
print('Verification passed: backbone and head outputs match.')
