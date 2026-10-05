"""Official WASB tennis weights, isolated from the existing reconstruction.

infer takes exactly three chronological BGR uint8 frames. It returns three
sigmoid heatmaps, one for each input frame, cropped to original H × W. The
default preserves every source pixel; an explicit input_size=(512, 288) can
reproduce the upstream training resolution for comparison. Scores are heatmap
activations, not calibrated probabilities or measured accuracy.
"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import torch
import yaml

from model_vendor.wasb.hrnet import HRNet


class _Config(dict):
    """Small attribute/dictionary adapter for upstream HRNet's config accesses."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _config(value):
    if isinstance(value, dict):
        return _Config({key: _config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_config(item) for item in value]
    return value


class WASB:
    def __init__(self, model_dir, device="mps", input_size=None):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.input_size = input_size
        config_path = Path(__file__).parent / "model_vendor/wasb/wasb.yaml"
        self.config = _config(yaml.safe_load(config_path.read_text()))
        self.model = HRNet(self.config)
        self.checkpoint = self.model_dir / "tennis.pth.tar"
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        # Strict loading verifies all 428 official tensor names and shapes.
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval().to(self.device)
        self.last_shapes = None

    def infer(self, frames):
        if len(frames) != 3:
            raise ValueError("WASB expects exactly three chronological BGR frames")
        shape = frames[0].shape
        if len(shape) != 3 or shape[2] != 3 or any(frame.shape != shape for frame in frames):
            raise ValueError("Frames must share one H × W × 3 shape")
        if any(frame.dtype != np.uint8 for frame in frames):
            raise ValueError("Frames must be uint8 BGR images")
        source_h, source_w = shape[:2]
        width, height = self.input_size or (source_w, source_h)
        # Stage fusion upsamples by powers of two. Padding avoids shape mismatch
        # for arbitrary crops while preserving coordinates and image detail.
        padded_h, padded_w = (height + 7) // 8 * 8, (width + 7) // 8 * 8
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        images = []
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            # Exact official RGB, ToTensor /255, ImageNet normalization.
            rgb = frame[..., ::-1].astype(np.float32) / 255.0
            normalized = (rgb - mean) / std
            normalized = np.pad(normalized, ((0, padded_h-height), (0, padded_w-width), (0, 0)), mode="edge")
            images.append(normalized.transpose(2, 0, 1))
        inputs = np.concatenate(images, axis=0)[None]
        with torch.inference_mode():
            tensor = torch.from_numpy(inputs).to(self.device)
            logits = self.model(tensor)[0]
            maps = logits.sigmoid().cpu().numpy()[0, :, :height, :width]
        if (width, height) != (source_w, source_h):
            maps = np.stack([cv2.resize(hm, (source_w, source_h), interpolation=cv2.INTER_LINEAR) for hm in maps])
        self.last_shapes = {
            "source_shape": [3, source_h, source_w, 3],
            "input_shape": list(inputs.shape),
            "raw_output_shape": list(logits.shape),
            "heatmap_shape": list(maps.shape),
            "resized": (width, height) != (source_w, source_h),
            "padding_right_bottom": [padded_w-width, padded_h-height],
        }
        return maps

    def metadata(self):
        provenance_path = self.model_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        return {
            "name": "WASB tennis",
            "device": str(self.device),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "provenance": provenance,
            "training_input_size": [512, 288],
            "requested_input_size": list(self.input_size) if self.input_size else "native source resolution",
            "preprocessing": "three chronological RGB frames; /255; ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]; channel concatenation",
            "outputs": "three sigmoid heatmaps aligned to their respective input frames",
            "score_note": "heatmap activation, not a calibrated probability",
            "last_shapes": self.last_shapes,
        }


def heatmap_candidates(heatmap, threshold=0.5, max_candidates=8):
    """Keep separate weighted components rather than picking a paddle-sized blob."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((heatmap > threshold).astype(np.uint8), 8)
    candidates = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        weights = heatmap[ys, xs]
        mass = float(weights.sum())
        candidates.append({
            "pixel": [float(np.dot(xs, weights) / mass), float(np.dot(ys, weights) / mass)],
            "score": float(weights.max()),
            "mass": mass,
            "area": int(stats[label, cv2.CC_STAT_AREA]),
        })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_candidates]
