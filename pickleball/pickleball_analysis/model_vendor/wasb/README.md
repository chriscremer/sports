This directory contains only the upstream HRNet source and WASB architecture
configuration required for inference, plus its MIT license. No upstream runtime,
training pipeline, dataset, or nested repository is included.

Source: https://github.com/nttcom/WASB-SBDT
Commit: `923462cacdeb3353b84ddebdedb3f4b7a8553b0f`

Unmodified copied files:

- `src/models/hrnet.py` → `hrnet.py`
- `src/configs/model/wasb.yaml` → `wasb.yaml`
- `LICENSE.md`

The official tennis checkpoint is kept outside the code checkout, in the user's
analysis storage under `models/wasb/tennis.pth.tar`. Its source URL and SHA256 are
recorded beside it in `provenance.json`.

`comparison_wasb.py` adds a small CPU/MPS inference wrapper. The official CUDA-only
detector is not used. The original model was trained at 512 × 288. Native-resolution
inference uses the same convolution weights at a different image scale, which is
an experiment and does not imply better accuracy. The wrapper keeps all pixels
when `input_size=None`, pads right/bottom to multiples of eight, and crops the
output back to source coordinates. Optional resizing uses a direct affine resize;
for the 16:9 video this matches the official centered affine transform.

Normalization follows upstream `src/dataloaders/__init__.py`: RGB /255 followed
by ImageNet mean and standard deviation. Three chronological frames are
concatenated as channels. The three sigmoid heatmaps correspond to their
respective frames, rather than all referring to the central frame.
