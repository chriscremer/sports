# Rally Lab: pickleball video reconstruction

Local video viewer for https://www.youtube.com/watch?v=tdGSHF5FseE with detected stick figures over the source footage and a synchronized, orbitable 3D court. Code is in this directory. Downloaded video, model weights, tracking JSON, clips, Python environment, and review images live in `/Users/chriscremer/Downloads/pickleball_video_analysis`.

## Open the viewer

```sh
cd /Users/chriscremer/code/sports/pickleball/pickleball_analysis
./start.sh
```

Open http://127.0.0.1:8765. Use `./start.sh --port 8766` if the default port is occupied. No Node/build step is required; Three.js and OrbitControls are vendored locally. Fonts have system fallbacks. The local HTTP server supports byte ranges for seeking. It binds to loopback only.

Two real excerpts are analyzed at 15 Hz: **00:55–01:05.4** and **01:15–01:18.4**. The viewer offers these two analyzed excerpts. All views use the HTML video clock. The source footage and detections are drawn together on a canvas to avoid disappearing video presentation layers in the embedded browser; the controls below the footage handle playback, mute, and fullscreen. Player colours indicate court side, not named individuals.

The page includes pose and learned ball overlays, frame stepping, playback speed, looping, scrubbing, orbit/zoom, three camera presets, and movement trails. A first estimated 3D ball trajectory covers 00:56.523–00:59.720 in the opening excerpt. Later exchanges and the second excerpt have 2D detections where supported; they do not have reliable reconstructed height yet.

## Public GitHub Pages viewer

The published page is https://chriscremer.ca/sports/pickleball/pickleball_analysis/index.html.
The files at this directory's root (`index.html`, `app.js`, `style.css`, `clips.json`,
`vendor/`, and `media/`) form a static export; no Python server runs on the website.
Both analyzed excerpts and their tracking data are included. The Full match button and export controls have been removed; the header links to the original video on YouTube.

After editing `web/` or regenerating analysis, rebuild the public files:

```sh
python3 build_static.py
```

The export uses relative URLs so the page works within `/sports/pickleball/pickleball_analysis/`.
Downloaded sources and model weights stay in the original Downloads data directory;
only the two excerpt videos and their analysis are copied into the public export.
Publishing uses the `main` branch of `chriscremer/sports` and its existing GitHub Pages setup.

## Analysis

```sh
./setup.sh
/Users/chriscremer/Downloads/pickleball_video_analysis/.venv/bin/python analyze.py --start 55 --duration 10.4 --hz 15 --clip
/Users/chriscremer/Downloads/pickleball_video_analysis/.venv/bin/python analyze.py --start 75 --duration 3.4 --hz 15 --clip
```

`analyze.py` expects `source-720p.mp4`, `source.mp4` (audio and full-match playback), and `models/yolo11s-pose.pt` in the data directory. Use `--data` to change it, `--device cpu` without Apple MPS, and `--ffmpeg` for an alternate ffmpeg executable. The initial run used existing soccer-analysis dependencies; the separate pickleball environment now contains the dependencies for continued work.

The 720p YouTube source uses AV1. OpenCV decodes it, writes a temporary MPEG-4 clip, and ffmpeg produces browser-friendly H.264/AAC while preserving source audio from the 360p copy. Scratch clips are removed after successful conversion. To download on a new machine, use current yt-dlp with a supported JavaScript runtime:

```sh
yt-dlp --js-runtimes node:/absolute/path/to/node -f 'bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[ext=mp4]' --merge-output-format mp4 -o '/Users/chriscremer/Downloads/pickleball_video_analysis/source-720p.%(ext)s' 'https://www.youtube.com/watch?v=tdGSHF5FseE'
yt-dlp -f 18 -o '/Users/chriscremer/Downloads/pickleball_video_analysis/source.%(ext)s' 'https://www.youtube.com/watch?v=tdGSHF5FseE'
```

Court corners are manually calibrated for these excerpts, in `analyze.py`. **Do not analyze arbitrary timestamps with these corners.** The broadcast cuts and changes zoom. Additional footage needs its own calibration and camera-cut detection first.

## Learned ball inference

The original SavedModel lives in `models/tracknet-pickleball` under the data directory. Its public download is the **New Weights** folder linked by TrackNet-Pickleball. The local runtime uses an unchanged-weight PyTorch port for Apple MPS; no TensorFlow is needed for normal inference.

```sh
/Users/chriscremer/Downloads/pickleball_video_analysis/.venv/bin/python track_ball.py --start 55
/Users/chriscremer/Downloads/pickleball_video_analysis/.venv/bin/python reconstruct_ball.py --start 55
/Users/chriscremer/Downloads/pickleball_video_analysis/.venv/bin/python track_ball.py --start 75
```

`track_ball.py` runs at the native video rate (29.97 Hz) and adds `ball_track` to the existing rally JSON. Use `--device cpu` without MPS. Running inference again invalidates old 3D fits; rerun reconstruction after changing the tracker. Player reanalysis preserves ball results when the interval and calibration match exactly. Reconstruction needs an explicit `annotations/ball-events-START.json`; the existing reviewed annotations cover only the opening shots of the first excerpt.

For a fresh model setup, use `gdown.download_folder` on the public folder `https://drive.google.com/drive/folders/1EGsddY1fgEJ5ITrfF32aPCn6nml2Anzr`, saving to `DATA/models/tracknet-pickleball`. Run `convert_tracknet.py --models DATA/models` once in a separate Python 3.12 environment with `tensorflow==2.16.2` and `tf-keras==2.16.0`. It exports the model graph and arrays, without retraining. The original checkpoint normalizes **image width**, rather than channels, in its batch-normalization layers; the port preserves this. A CPU TensorFlow reference and MPS inference were compared on a real video triplet: maximum absolute heatmap difference 2.03e-6. This checks conversion fidelity, not tracking accuracy.

## What is observed, and what is estimated

- **Image pose:** actual [YOLO11s-pose detections](https://docs.ultralytics.com/tasks/pose/), with 17 COCO keypoints. Two overlapping far-court tiles improve small-player detections, including players up to 6 m behind the far baseline. Low-confidence joints are omitted.
- **Tracking:** Hungarian assignment with a court-side constraint and image-box continuity. Gaps of at most 0.4 seconds between observations of the same ID are interpolated, with a speed gate and no endpoint extrapolation. Interpolated samples have `tracking_status: interpolated`, null detection confidence, and supporting timestamps; they appear dashed in video and translucent in 3D. They are excluded from the position index. Longer gaps stay empty. Occlusions and missed detections can still fragment IDs. The output is not a verified player identity dataset.
- **Feet:** four manually marked court corners and an OpenCV homography yield planar positions, in metres. Feet may be off the ground, invisible, or inaccurately detected.
- **3D skeleton:** camera basis estimated from the court and an assumed focal length; joints are lifted into an upright plane through the player's foot position. Visible body height and sideways articulation are retained, but hidden limb depth is not recovered. The world chart is x-right, y-towards-near-baseline, height-up; the camera basis includes a handedness conversion.
- **2D ball:** the publicly released [TrackNet-Pickleball](https://github.com/AndrewDettor/TrackNet-Pickleball) checkpoint, without training on this match. It receives three consecutive RGB frames and returns three aligned heatmaps. Overlapping temporal windows are averaged, then a 0.5 heatmap threshold and temporal filters reject isolated spikes and stationary objects. Heatmap peaks are not calibrated probabilities. The model still misses later exchanges and can confuse paddles with the ball. No colour filter is used.
- **3D ball:** a continuous fit of gravity and per-flight linear drag to the learned observations in the reviewed opening shots. The event times and one occluded contact pixel are reviewed annotations in the data directory; bounce height is constrained to the ball radius and contact heights are explicit approximate priors. Shared event positions keep adjacent flights continuous. Fits with too little support, large reprojection error or implausible speed are omitted. Gaps longer than 0.35 s stay empty and no flight is extrapolated beyond its reviewed endpoints. The ball is enlarged in 3D for visibility. These height and velocity estimates are **not training ground truth**, and low reprojection error does not establish accurate 3D depth. The JSON separates `ball` model observations from `reconstruction` estimates.
- **Score:** a transparent position heuristic, **not** a trained value function or rally-win probability. For complete four-player observations: `clip(50 + 8 × (far mean kitchen distance − near mean kitchen distance), 0, 100)`. It compares proximity, not tactical correctness. No action recommendations or counterfactual simulation have been trained.

## Verification

```sh
/Users/chriscremer/Downloads/pickleball_video_analysis/.venv/bin/python -m unittest discover -s tests
```

25 tests cover court mapping, positive skeleton height, reprojection of lifted joints, HTTP byte ranges, path confinement, bounded player interpolation, ball RGB triplet alignment, heatmap localization, temporal spike/stationary filtering, synthetic flight recovery, gravity/drag endpoints, unsupported ball gaps, comparison candidate scaling and temporal padding, independent 1080p reconstruction, input preservation and unsupported-flight omission. Browser verification covers source video loading, timeline synchronization, camera changes, overlays, excerpt switching, comparison clips, pixel inspection and prediction export.

## Separate 1080p model comparison

Open http://127.0.0.1:8765/ball-models.html for the isolated comparison. The original viewer, ball observations, and reviewed 3D flights are unchanged. The new page uses a genuine 1920 × 1080 YouTube format-137 source, with two frame-aligned H.264 excerpts. Inference decodes the original download rather than the preview encodes.

The default excerpt is the original viewer's 00:55–01:05.4 clip: the same 312 frames, starting at the beginning at 1× speed. Excerpt bounds are read from the original analysis metadata. The displayed match clock follows the original viewer; inference and flight fitting retain actual decoded source timestamps.

Six configurations compare TrackNet-Pickleball at 512 × 288, official [TrackNetV3](https://github.com/qaz812345/TrackNetV3) badminton weights at 512 × 288, 960 × 544 and native 1920 × 1080, and official [WASB](https://github.com/nttcom/WASB-SBDT) tennis weights at 512 × 288 and native 1920 × 1080. All checkpoints were trained at 512 × 288. Larger inputs are experimental; native resolution can impair detection because it changes apparent ball size. Toggle model overlays, step by one decoded frame, and compare candidate crops and synchronized courts beside the source footage.

All generated results and source files live under the data directory; `comparison/manifest.json` describes input sizes and frame timing. `comparison/rally-START-models.json` preserves multiple thresholded candidate regions, peak activations, empty frames and model metadata. Candidate coverage is not accuracy. This experiment does not alter the original reconstruction, train models, or identify rally winners.

The page now highlights three setups in synchronized 3D courts: TrackNet-Pickleball 512 × 288, TrackNetV3 512 × 288 and TrackNetV3 960 × 544. These have the three highest combined candidate counts across the two excerpts, not verified accuracy scores. Each ball is fitted independently from that setup's 1080p predictions, using the same original calibration and reviewed opening events/heights. Common original player poses provide context. Camera orbit/zoom and perspective/top/side presets are linked across the three courts; stick figures and ball trails can be toggled. Other configurations remain under the 2D references disclosure.

```sh
/Users/chriscremer/Downloads/pickleball_video_analysis/.venv/bin/python reconstruct_comparison.py
```

Reconstruction output is isolated in `comparison/reconstruction-manifest.json` and `comparison/rally-START-3d.json`. The opening reviewed interval is 00:56.523–00:59.720. Missing support skips individual flights without discarding neighboring flights. Unsupported gaps and times outside accepted flights show unknown height. Hollow markers preserve the strongest 2D candidate in each court view by applying the original court homography under a ground-plane assumption; `detection_display` stores `ground_xy`, null height, and a method label. These markers are not reconstructed ball positions or training ground truth. Candidates whose image residual from a supported fit exceeds 18 pixels remain visible beside the fitted ball. Missing detections produce no marker. The pixel inspector is removed, and playback controls sit directly beneath the source footage. The second excerpt has no reviewed ball events and shows common player poses and hollow 2D detection projections on the courts. Reprojection thresholds use the original 1280 × 720 calibration space; displayed errors are converted to 1080p pixels. Heights and distances between fits are estimates, not training ground truth. Raw 2D predictions and 3D fits have separate JSON downloads.

```sh
/Users/chriscremer/Downloads/pickleball_video_analysis/.venv/bin/python compare_ball_models.py
```

The runner skips completed model/clip pairs. `--refresh` rebuilds only comparison artifacts. Model weights are under `models/wasb/tennis.pth.tar` and `models/tracknet-v3/TrackNet_best.pt`; provenance and official source licenses are retained locally. WASB uses ImageNet-normalized chronological RGB triplets. TrackNetV3 uses eight RGB frames and a median original-resolution background from 16 uniformly spaced excerpt frames. Each configuration uses non-overlapping windows; only the final window is padded, and padded frames are not emitted. No interpolation or smoothing is applied. The original viewer uses its existing averaging and temporal filters, so its results need not match the raw reference card.

## Further work

Improve the pretrained ball tracker on held-out labeled pickleball footage, evaluate precision/recall against reviewed image labels, and extend reviewed 3D flights through the remaining exchanges. Refine camera calibration and quantify height uncertainty before using these estimates as training features. For a learned score, label rally winners and build leakage-free pre-shot states with ball height and velocity. Train and validate on held-out matches before displaying probabilities or action rankings.

Ultralytics code/model use is governed by its published licensing; Three.js and OrbitControls use the MIT license. Video is retained locally for this analysis.
