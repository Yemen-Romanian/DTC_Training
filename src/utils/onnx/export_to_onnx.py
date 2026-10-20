import time
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quant_pre_process
from onnxruntime.quantization.quantize import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static
)
from datasets.mixed_dataset import MixedDataset
from utils.paths import Paths
from models.trackers.tracker_factory import create_tracker
from utils.onnx.calibration_reader import (
    SiamBANBackboneCalibrationReader,
    SiamBANHeadCalibrationReader,
    sample_calibration_pairs
)


def benchmark(pt_model, backbone_session, head_session, exemplar: torch.Tensor, search: torch.Tensor,
              runs: int = 200, warmup: int = 20):
    """Time the per-frame tracking step: search backbone + head, with exemplar features cached.

    This is what TrackerSiamBAN.track() runs every frame; the exemplar pass happens once in
    initialize() and is left out.
    """
    search_np = search.numpy()

    pt_model.eval()
    with torch.no_grad():
        pt_z_feat = pt_model.backbone(exemplar)
    ort_z_feat = backbone_session.run(['features'], {'image': exemplar.numpy()})[0]

    def pt_step():
        with torch.no_grad():
            pt_model.head(pt_z_feat, pt_model.backbone(search))

    def ort_step():
        x_feat = backbone_session.run(['features'], {'image': search_np})[0]
        head_session.run(['cls', 'reg'], {'z_feat': ort_z_feat, 'x_feat': x_feat})

    def time_step(step):
        for _ in range(warmup):
            step()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            step()
            times.append(time.perf_counter() - t0)
        return times

    pt_times = time_step(pt_step)
    ort_times = time_step(ort_step)

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

def export_siamban_to_onnx(checkpoint_name: str, validation_paths: dict):
    state_dict_path = str(Paths.model_weights_dir() / checkpoint_name)
    tracker = create_tracker(model_config, state_dict=state_dict_path)
    model = tracker.model

    examplar = torch.randn((1, 3, 127, 127))
    search = torch.randn((1, 3, 255, 255))

    # Derive example feature tensors for the head export
    model.eval()
    with torch.no_grad():
        z_feat_example = model.backbone(examplar)
        x_feat_example = model.backbone(search)

    backbone_path = Paths.model_weights_dir() / "siamban_backbone.onnx"
    head_path = Paths.model_weights_dir() / "siamban_head.onnx"

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

    print(f"Running preprocessing")
    preprocessed_backbone_path = backbone_path.parent / (backbone_path.stem + "_prepr.onnx")
    preprocessed_head_path = head_path.parent / (head_path.stem + "_prepr.onnx")
    quant_pre_process(input_model=backbone_path, output_model_path=preprocessed_backbone_path)
    quant_pre_process(input_model=head_path, output_model_path=preprocessed_head_path)

    # Quantization
    pairs = sample_calibration_pairs(MixedDataset(validation_paths), num_pairs=200, seed=0)
    backbone_calibration_reader = SiamBANBackboneCalibrationReader(pairs=pairs)
    head_calibration_reader = SiamBANHeadCalibrationReader(pairs=pairs, fp32_backbone_path=backbone_path)
    quantized_backbone_path = backbone_path.parent / (backbone_path.stem + "_int8.onnx")
    quantized_head_path = head_path.parent / (head_path.stem + "_int8.onnx")
    quantize_static(
        preprocessed_backbone_path,
        quantized_backbone_path,
        backbone_calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax
    )

    quantize_static(
        preprocessed_head_path,
        quantized_head_path,
        head_calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax
    )

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

    # benchmark(model, backbone_session, head_session, examplar, search)

if __name__ == '__main__':
    export_siamban_to_onnx(
        "siamban_mixed_seed42_iou_0.6635_iog_0.7745_center_dist_5.3526_center_dist_norm_0.1766_lost_tracks_2.5000_evaluated_frames_626.8000.pth",
        {"visdrone": r"..\..\datasets\VisDrone\val"}
    )
