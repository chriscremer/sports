"""Run isolated pretrained-model experiments on genuine 1080p source frames.

Outputs are separate from the existing viewer's analysis and reconstruction.
No tracking accuracy is claimed: these are raw predictions without labels.
"""
import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from comparison_tracknet_v3 import TrackNetV3, heatmap_candidates
from comparison_wasb import WASB, heatmap_candidates as wasb_candidates
from tracknet import TrackNet, preprocess

CONFIGS = [
    dict(id='pickleball-512', name='TrackNet · Pickleball', size=[512,288], family='pickleball', color='#ffbf47', training='Pickleball', url='https://github.com/AndrewDettor/TrackNet-Pickleball'),
    dict(id='v3-512', name='TrackNetV3 · Reference', size=[512,288], family='v3', color='#5edac7', training='Badminton', url='https://github.com/qaz812345/TrackNetV3'),
    dict(id='v3-960', name='TrackNetV3 · Larger input', size=[960,544], family='v3', color='#79adff', training='Badminton', url='https://github.com/qaz812345/TrackNetV3'),
    dict(id='v3-1080', name='TrackNetV3 · Native 1080p', size=[1920,1080], family='v3', color='#c6a0ff', training='Badminton', url='https://github.com/qaz812345/TrackNetV3'),
    dict(id='wasb-512', name='WASB · Reference', size=[512,288], family='wasb', color='#ff869c', training='Tennis', url='https://github.com/nttcom/WASB-SBDT'),
    dict(id='wasb-1080', name='WASB · Native 1080p', size=[1920,1080], family='wasb', color='#f5e681', training='Tennis', url='https://github.com/nttcom/WASB-SBDT'),
]


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False))
    temporary.replace(path)


def windows(cap, start_frame, count, length):
    """Yield consecutive groups, repeating the final frame only as padding."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for offset in range(0, count, length):
        frames, timestamps = [], []
        for _ in range(min(length, count-offset)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Source ended before the requested interval')
            frames.append(frame)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        valid = len(frames)
        frames.extend([frames[-1]]*(length-valid))
        yield frames, timestamps, valid


def run_model(config, clip, source, data, device):
    family, size = config['family'], tuple(config['size'])
    if family == 'v3':
        model = TrackNetV3(data/'models/tracknet-v3', device, size)
        cap = cv2.VideoCapture(str(source))
        samples = []
        for index in np.linspace(clip['first_frame'], clip['first_frame']+clip['frame_count']-1, 16, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok: raise RuntimeError('Could not read background sample')
            samples.append(frame)
        model.set_background(samples)
        del samples
        length = model.sequence_length
    elif family == 'wasb':
        model = WASB(data/'models/wasb', device, size)
        length = 3
    else:
        model = TrackNet(data/'models').eval().to(device)
        length = 3
    cap = cv2.VideoCapture(str(source))
    rows = []
    started = time.monotonic()
    for frames, timestamps, valid in windows(cap, clip['first_frame'], clip['frame_count'], length):
        if family == 'pickleball':
            with torch.inference_mode():
                maps = model(torch.from_numpy(preprocess(frames)[None]).to(device)).cpu().numpy()[0]
        else:
            maps = model.infer(frames)
        for i in range(valid):
            raw = wasb_candidates(maps[i]) if family == 'wasb' else heatmap_candidates(maps[i], (clip['width'],clip['height']))
            candidates = [dict(pixel=[round(float(x),2) for x in c['pixel']], score=round(float(c.get('score',c.get('confidence'))),4), area=c['area']) for c in raw]
            rows.append(dict(t=round(timestamps[i]-clip['source_start'],6), source_t=round(timestamps[i],6), peak=round(float(maps[i].max()),4), candidates=candidates))
        del maps, frames
        if len(rows) % (length*20) == 0:
            print(f"{clip['id']} {config['id']}: {len(rows)}/{clip['frame_count']}", flush=True)
    cap.release()
    result = dict(frames=rows, frame_count=len(rows), frames_with_candidates=sum(bool(r['candidates']) for r in rows), inference_seconds=round(time.monotonic()-started,2), threshold=0.5, sequence_length=length, input_size=list(size), temporal_mode='non-overlapping chronological windows; final window padded by repeating last frame; no interpolation or filtering')
    if family == 'wasb': result['metadata'] = model.metadata()
    if family == 'v3': result['metadata'] = dict(checkpoint='TrackNet_best.pt', training_input_size=[512,288], background='16 uniformly spaced original-resolution frames; RGB temporal median', upstream_commit='6eda442ada1740573f200f836d93edc9a541ee86', checkpoint_sha256='df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a')
    del model
    gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'))
    parser.add_argument('--ffmpeg', default='/Users/chriscremer/miniconda3/bin/ffmpeg')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    output = args.data/'comparison'
    output.mkdir(exist_ok=True)
    source = args.data/'source-1080p.mp4'
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width,height) != (1920,1080): raise ValueError(f'Expected genuine 1080p source, got {width}x{height}')
    clips = []
    for start in [55,75]:
        original_clip=json.loads((args.data/'analysis'/f'rally-{start}.json').read_text())['clip']
        duration=original_clip['end']-original_clip['start']
        first, count = round(start*fps), round(duration*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        ok, _ = cap.read()
        if not ok: raise RuntimeError('Could not seek source')
        source_start = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        end=original_clip['end']
        clip = dict(id=f'rally-{start}', label=f'{start//60:02}:{start%60:02}–{int(end)//60:02}:{end%60:04.1f} · {duration:.1f}s', viewer_start=original_clip['start'], viewer_end=end, original_viewer_video=original_clip['video'], first_frame=first, frame_count=count, source_start=source_start, fps=fps, width=width, height=height, duration=count/fps, video=f'comparison/rally-{start}-1080p.mp4', predictions=f'comparison/rally-{start}-models.json')
        preview = args.data/clip['video']
        if args.refresh or not preview.exists():
            # Exact frame-index trim, rather than an approximate keyframe seek.
            subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(source),'-an','-vf',f'trim=start_frame={first}:end_frame={first+count},setpts=PTS-STARTPTS','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(preview)], check=True)
        clips.append(clip)
    cap.release()
    manifest = dict(source=dict(file='source-1080p.mp4', width=width,height=height,fps=fps,url='https://www.youtube.com/watch?v=tdGSHF5FseE',youtube_format='137',note='All inference uses the downloaded original 1920×1080 frames. Browser clips are H.264 previews.'),models=CONFIGS,clips=clips,threshold=0.5,note='Unlabeled experiments. Coverage measures frames with candidates, not localization accuracy. Larger inputs are experimental; every checkpoint was trained at 512×288. No synthetic gap filling or 3D coordinates.')
    atomic_json(output/'manifest.json', manifest)
    for clip in clips:
        path = args.data/clip['predictions']
        results = json.loads(path.read_text()) if path.exists() and not args.refresh else dict(clip=clip, models={})
        results['clip']=clip
        for config in CONFIGS:
            if config['id'] in results['models']: continue
            print(f"Starting {clip['id']} {config['id']}", flush=True)
            results['models'][config['id']] = run_model(config, clip, source, args.data, args.device)
            atomic_json(path, results)
            summary = results['models'][config['id']]
            print(f"Done {config['id']}: {summary['frames_with_candidates']}/{summary['frame_count']} frames have candidates ({summary['inference_seconds']}s)", flush=True)
        atomic_json(path,results)
    print('Ready: http://127.0.0.1:8765/ball-models.html', flush=True)


if __name__ == '__main__':
    main()
