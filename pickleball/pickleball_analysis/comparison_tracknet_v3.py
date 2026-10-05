"""Official TrackNetV3 checkpoint adapter for the isolated comparison experiment.

The trained input is 512x288, eight RGB frames plus a median RGB background.
Larger ``input_size`` is an experimental fully convolutional inference mode;
it does not mean that the checkpoint was trained at that resolution.
"""
from pathlib import Path
import importlib.util

import cv2
import numpy as np
from PIL import Image
import torch


class TrackNetV3:
    trained_input_size = (512, 288)
    sequence_length = 8
    requires_background = True
    source_url = 'https://github.com/qaz812345/TrackNetV3'
    weight_url = 'https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view'

    def __init__(self, model_dir, device=None, input_size=(512, 288)):
        self.device = torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.input_size = tuple(map(int, input_size))
        if any(n <= 0 or n % 8 for n in self.input_size):
            raise ValueError('TrackNetV3 input dimensions must be positive multiples of 8')
        path = Path(__file__).parent / 'model_vendor' / 'tracknet_v3' / 'model.py'
        spec = importlib.util.spec_from_file_location('_comparison_tracknet_v3_upstream', path)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        checkpoint = torch.load(Path(model_dir) / 'TrackNet_best.pt', map_location='cpu', weights_only=True)
        params = checkpoint['param_dict']
        self.sequence_length = int(params['seq_len'])
        self.bg_mode = params['bg_mode']
        if self.bg_mode != 'concat':
            raise ValueError(f'Expected official concat checkpoint, got {self.bg_mode!r}')
        self.model = upstream.TrackNet((self.sequence_length + 1) * 3, self.sequence_length)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval().to(self.device)
        self._background_rgb = None

    def set_background(self, frames_bgr):
        """Estimate the original-size temporal median, then resize as upstream does.

        Supply a representative rally frame sample, not just one inference window.
        The stored median is RGB uint8. Original-resolution median before resize
        follows the official Dataset's frame_arr inference path exactly.
        """
        frames = np.asarray(frames_bgr)
        if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
            raise ValueError('Background requires a nonempty BGR frame sequence')
        self._background_rgb = np.median(frames[..., ::-1], axis=0).astype(np.uint8)

    def _rgb_chw(self, rgb):
        # Pillow Image.resize default BICUBIC matches the upstream Dataset.
        resized = np.asarray(Image.fromarray(rgb).resize(self.input_size))
        return np.moveaxis(resized, -1, 0)

    def infer(self, frames_bgr):
        """Return eight probability heatmaps in temporal order, shape (8,H,W)."""
        if len(frames_bgr) != self.sequence_length:
            raise ValueError(f'Expected {self.sequence_length} frames, got {len(frames_bgr)}')
        if self._background_rgb is None:
            raise ValueError('Call set_background with representative rally frames before infer')
        planes = [self._rgb_chw(self._background_rgb)]
        planes.extend(self._rgb_chw(np.asarray(frame)[..., ::-1]) for frame in frames_bgr)
        arr = np.concatenate(planes, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr[None]).to(self.device)
        with torch.inference_mode():
            heatmaps = self.model(tensor).cpu().numpy()[0]
        return heatmaps

    def candidates(self, frames_bgr, threshold=0.5, max_candidates=8):
        """Return all strong blobs, rather than silently selecting the largest one.

        Coordinates are mapped back to the original frame. Probability is a
        heatmap score, not a calibrated probability of correct ball localization.
        """
        heatmaps = self.infer(frames_bgr)
        h, w = np.asarray(frames_bgr[0]).shape[:2]
        return [heatmap_candidates(m, (w, h), threshold, max_candidates) for m in heatmaps]


def heatmap_candidates(heatmap, original_size, threshold=0.5, max_candidates=8):
    """Extract confidence-sorted connected components with weighted centroids."""
    binary = np.asarray(heatmap >= threshold, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    sx, sy = original_size[0] / heatmap.shape[1], original_size[1] / heatmap.shape[0]
    result = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        scores = heatmap[ys, xs]
        total = float(scores.sum())
        if total <= 0:
            continue
        result.append({'pixel': [float(np.dot(xs, scores) / total * sx),
                                 float(np.dot(ys, scores) / total * sy)],
                       'confidence': float(scores.max()),
                       'area': int(stats[label, cv2.CC_STAT_AREA]),
                       'status': 'detected'})
    return sorted(result, key=lambda item: item['confidence'], reverse=True)[:max_candidates]
