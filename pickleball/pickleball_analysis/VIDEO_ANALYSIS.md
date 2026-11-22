# Pickleball video analysis workflow

The local v2 viewer displays a source replay, player poses, reviewed ball positions, estimated 3D flights, and a timeline of the rally and paddle contacts. The process combines model inference with a visual review of the original frames. Keep the original recording, model output, and review decisions as separate inputs so a later review can be replayed against the same evidence.

## 1. Prepare the video and pick an excerpt

Run the commands below from this project directory. Create a separate data folder per recording under `/Users/chriscremer/Downloads/pickleball_video_analysis`. The examples use Bourne shell syntax. Set `START` and `DURATION` to the actual excerpt you choose.

```sh
DEMO=/Users/chriscremer/Downloads/pickleball_video_analysis
DATA="$DEMO/new-recording"
PY="$DEMO/.venv/bin/python"
mkdir -p "$DATA/v2-ball-review"
ln -s "$DEMO/models" "$DATA/models"
```

Use `./setup.sh` and the model setup in [README.md](README.md) if the environment or weights are missing. The model directory contains the YOLO pose weights, converted TrackNet-Pickleball weights, and TrackNetV3 and WASB checkpoints. Use `--device cpu` on machines without Apple MPS.

Place the highest quality playable copy as `source-1080p.mp4`. For a YouTube recording, use the 1080p video-only stream when available. The comparison runner currently expects 1920 × 1080 constant-frame-rate footage. Normalize variable-frame-rate footage before analysis, then use that normalized copy as the source of truth for frame indices and timestamps. Derive the 720p input from the same copy with the same frame order and clock.

```sh
ffmpeg -i "$DATA/source-1080p.mp4" -map 0:v:0 -an -vf scale=1280:720 \
  -c:v libx264 -crf 17 "$DATA/source-720p.mp4"
```

Scan the wide view for the serve, exchanges, and stoppage. Select a short excerpt with a little footage on both sides of the rally. Split at cuts or material camera movement and calibrate each stable view. Retain the original match clock when selecting the window. For the demo, the clip starts around match time `00:55`, while the serve is `1.57s` on the clip timeline.

## 2. Calibrate the court and run the models

Create `camera.json` in the new data folder. Record the source `id`, `title`, and optional `url` under `source`. Set `corners_normalized` to four `[x/width, y/height]` points ordered **far-left, far-right, near-right, near-left**, as viewed in the image. Use the outside court corners. Optionally set `far_crops_normalized` to a list of `[left, top, right, bottom]` rectangles to recover small far-side players. Keep space behind the baseline for moving players. Inspect these points on frames at both ends of the clip before inference. The demo's calibration is the default when `--camera` is absent.

```sh
START=120          # Example, measured on the full recording
DURATION=10.4      # Example, includes the serve through the stoppage
"$PY" analyze.py --data "$DATA" --camera "$DATA/camera.json" \
  --start "$START" --duration "$DURATION" --hz 15
"$PY" track_ball.py --data "$DATA" --start "$START"
"$PY" compare_ball_models.py --data "$DATA" --clip-start "$START"
```

`analyze.py` runs YOLO pose detection and links players within a court side. `track_ball.py` scores consecutive frames with TrackNet-Pickleball at the source frame rate on the 720p input. The comparison runner decodes 1080p, saves time-aligned candidates from six model setups, and generates the exact-frame browser preview. It reuses cached model runs, so use a fresh folder or `--refresh` when the recording or excerpt changes. For a nonstandard camera, check that the current player and ball image bounds still contain the court and run-off. The comparison runner's `--ffmpeg` flag can select another binary.

Once the baseline is ready, keep `analysis/rally-START.json` fixed during review. Re-running ball inference invalidates the review's baseline fingerprint. Re-check review observations against the new baseline before reusing them.

## 3. Review ball positions, rally boundaries, and hits

1. Scan wide chronological grids with every third or fourth frame. Review the serve, direction changes, apparent gaps, net exchanges, and the end in cropped grids with every frame. Confirm the ball in adjacent frames before placing its center. Mark occlusions or ambiguity in review notes and leave those frames empty. Audit some frames with existing detections for false positives. A marker on a paddle is not evidence of a hit.
2. Record the serve at paddle contact. Follow the incoming and outgoing ball to identify each subsequent contact and its camera side (`near` or `far`). Use the last decisive event for the end, such as the second bounce, a visibly out landing, or a failed net return dropping on the hitter's side. For uncertain timing, save the earliest and latest plausible frames. Leave unclear contacts out and record why. The serve also counts as a near- or far-team hit.
3. Preserve reviewed frame indices, original source frame numbers, source timestamps, and the evidence sheet. Clip time is the frame index divided by the native FPS. Use full-recording timestamps when tagging the source. Use clip-relative seconds in the viewer.

```sh
"$PY" review_ball.py --data "$DATA" --clip-start "$START" \
  --start 0 --stop 48 --step 4 --crop 200 150 1500 750 \
  --scale .4 --output "$DATA/v2-ball-review/scan-000.png"
```

`--start` and `--stop` here are **excerpt frame indices**, with an exclusive stop. Adapt the crop (`x y width height`, in native pixels) to the shot. The PNG's companion JSON stores each tile's origin and dimensions. Map a clicked tile coordinate back to native pixels using `crop_origin + (sheet_point - tile_origin) × crop_size / tile_size`, independently for x and y.

Save these inputs beside the PNGs in the review folder:

- `observations.json` includes `baseline_sha256`, `inspected_frames` (excerpt indices), and `observations`. Each addition has `index`, `source_frame`, `source_seconds`, `pixel_1080p`, `visibility: "visible"`, `sheet`, and `sheet_pixel`. Compute the fingerprint with `shasum -a 256 "$DATA/analysis/rally-$START.json"`. Use the exact generated analysis filename if `START` was fractional. The replay currently adds confirmed misses to empty baseline frames. Keep suspected wrong positives in review notes for a separate correction pass.
- `rally-events.json` includes `clip` (the ID in `comparison/rally-START-models.json`) and `events`, ordered from `serve` through `hit` to `end`. Each event has `kind`, `frame`, `source_frame`, `source_seconds`, `team` (`near`, `far`, or `null` for the end), inclusive `frame_interval`, `sheet`, and `evidence`. A single file describes one complete rally. The renderer validates the event times against the exact preview frame clock.
- Optional `flight-windows.json` includes `windows`. For each window, supply ordered `anchors` and an `evidence` note. Each anchor uses the excerpt `frame`, a `kind` such as `contact`, `bounce`, or `visible_endpoint`, and a `height_prior_m`. Set `pin_visible_pixel` only when the corresponding ball center was reviewed and is available in the track. A bounce uses the ball radius (`0.037` m). Use plausible ranges to reason about contact height and reject a fit if small changes in assumptions move it implausibly. Keep windows separate across cuts and uncertain contacts. The replay skips poor fits and withholds height across unsupported stretches.
- Optional `bounce-events.json` includes `bounces` for ground contacts visible outside a fitted flight. Record `frame`, `ground_pixel_1080p` at the point touching the floor, `sheet`, and `evidence`. The replay combines these contacts with fitted bounce anchors and shows a brief court ripple. Mark actual ground contacts rather than inferring them from a missing ball or a net touch alone.

For the worked example, see `v2-ball-review/observations.json`, `rally-events.json`, `flight-windows.json`, and the named evidence grids under the demo data folder. The demo also has five earlier baseline flight fits, generated by `reconstruct_ball.py` from `annotations/ball-events-55.json`. A new recording can start with only the v2 flight windows.

## 4. Build and verify the local review

```sh
"$PY" replay_ball_review.py "$DATA/v2-ball-review/observations.json" \
  --data "$DATA" --clip-start "$START"
PICKLEBALL_DATA_DIR="$DATA" PICKLEBALL_PYTHON="$PY" ./start.sh --port 8766
```

Open `http://127.0.0.1:8766/media/v2-ball-review/index.html`. Click the start, end, and both teams' hit markers. Check that the video, playhead, event label, and 3D view agree on each contact. Inspect the marked frames and the neighboring frames. Verify that the timeline uses clip seconds and the footage panel labels match time separately. The outlined markers indicate a wider timing interval. The 3D view interpolates player joints within the same track and animates supported ball flights at display speed. Very short, compatible gaps between flight fits can be interpolated for playback. Detections with unknown height appear only in the source footage. Compare reviewed additions against the baseline track in the analysis JSON and check transitions into and out of occlusions.

Use `report.json` to record reviewed additions and the number of frames with estimated height. Position counts measure coverage. Estimate tracking accuracy on separately labeled rallies by scoring visible-ball recall, wrong detections, pixel error, and missing spans. A single camera leaves depth uncertain, and the timeline records team contacts rather than player identity or the official score.

## Opening five-minute match view

The demo's `match-five/camera-windows.json` records usable camera windows in match seconds and names each calibration under `cameras`. Verify both sides of every cut. For fixed end-court views, mark the four outer court corners. For moving sideline views, mark the four kitchen intersections on several reference frames in the same shot. The tracker matches court features to nearby references and omits frames where it cannot locate the court reliably. `build_match_review.py --hz 15` runs pose tracking and builds `match-five/index.html`. It combines wide and cropped detections, smooths body position and joint offsets separately, and bridges short gaps within a player track. Use `--moving-only` to rerun sideline calibration and tracking while reusing fixed-angle samples at the same rate, or `--assemble-only` to rebuild the page from cached samples.

The embedded player and local fallback share the full match clock. The reconstruction overlay projects the displayed 3D joints, player ground positions, and ball into the current camera shot. A ring with a center dot marks a predicted ball position. Assembly caches the source camera matrices in `match-five/overlay-cameras.json` and exports them in `match.json`.

Review the footage in chronological frame grids to find each serve and end of play. Recheck those boundaries on denser grids, including across camera cuts. Record additional spans in `match-five/rally-spans.json` under `rallies` and set `approximate` when the boundary times are uncertain. Set `reviewed_through` to the last match second continuously checked from the start. Within that coverage, gaps between Ball in play spans are treated as breaks in play.

The Analyzed lane highlights rallies with frame-aligned contacts. To add a hand-reviewed timeline, save a JSON file in `match-five` with the event fields from `v2-ball-review/rally-events.json` and a `source_clip` object with `id`, `first_frame`, `frame_count`, `fps`, `source_start`, and `duration`. Set the span's `hit_review` to that filename.

To process the ball across the match view, run the main TrackNet-Pickleball pass, then the court crop and larger TrackNetV3 passes to recover some misses.

```sh
"$PY" analyze_match_ball.py --data "$DATA"
"$PY" analyze_match_ball.py --data "$DATA" --model crop
"$PY" analyze_match_ball.py --data "$DATA" --model v3
"$PY" fit_match_ball.py --data "$DATA" --candidates
```

Use the printed direction changes as clues for reviewing contacts against the footage. Sharp audio transients can corroborate some hits. Align audio to a visible paddle contact first because the broadcast can have an audio delay. Record approximate contacts in `match-five/hit-candidates.json` under `rallies`. Map each rally's original start time formatted to three decimal places to a chronological list of `[source_seconds, team]` pairs starting with the serve. A span's hand-reviewed `hit_review` takes precedence when supplied.

```sh
"$PY" fit_match_ball.py --data "$DATA"
"$PY" build_match_review.py --data "$DATA" --assemble-only
```

The fitter caches camera projections, estimates flights from supported detections and gravity, and checks reprojection error and continuity at nearby flight boundaries. The match builder exports a position for every source frame in a marked rally. It refines the entire path against image rays and uses gravity, contact timing, and net clearance between opposite-side hits to guide the reconstruction. It downweights model detections that disagree with neighboring evidence. A translucent ball and an approximate height label identify predictions. Verify localization by withholding some reviewed frames from the fit and projecting the resulting 3D path back into the video. This check measures image alignment. Measuring depth accuracy needs an independent view or calibrated 3D positions.

To fill gaps, make chronological grids from the 1080p source and record visible ball centers in `match-five/ball-grid-observations.json` under `observations`. Each entry contains the native source `frame`, `pixel_960` in the 960 × 540 reference image, and the evidence `sheet`. Use neighboring frames to confirm the ball before replacing a model candidate. The fit weights reviewed centers and can recover shorter flights around them. Rerun `fit_match_ball.py` and `--assemble-only` after editing the file.

A pink ripple marks an airborne paddle contact. Add grid-confirmed contacts under `volleys` with the source `frame` and evidence `sheet`. When a fit cannot locate an overhead contact, an optional `pixel_960` and `height_prior_m` place an approximate ripple. The fitter also marks volleys when the incoming flight is continuously supported above the floor. Yellow ripples mark ground bounces. Blue ripples mark other paddle contacts on the Detailed hits timeline, including serves. A white ripple flags an abrupt horizontal reversal with no nearby marked contact. Verify those reversals against the footage before recording additional hits or bounces.
