"""Track the ball on the native match clock for each reviewed rally."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compare_ball_models import atomic_json, windows
from comparison_tracknet_v3 import TrackNetV3
from tracknet import TrackNet, heatmap_detection, preprocess

WIDTH, HEIGHT = 960, 540


def intervals(output):
    review = json.loads((output / 'rally-spans.json').read_text())
    rallies = review['rallies'] if isinstance(review, dict) else review
    return sorted(rallies, key=lambda rally: rally['start'])


def shot_windows(output):
    return json.loads((output / 'camera-windows.json').read_text())['wide_windows']


def samples(fps, start, end, cameras):
    for camera in cameras:
        first = max(round(start * fps), int(np.ceil(camera['start'] * fps)))
        last = min(round(end * fps), int(np.ceil(camera['end'] * fps)))
        if last <= first:
            continue
        yield camera, first, last


def track(data, output, device, refresh, model_name='pickleball'):
    if model_name == 'v3':
        model = TrackNetV3(data / 'models/tracknet-v3', device, (960, 544))
        length = model.sequence_length
    else:
        model = TrackNet(data / 'models').eval().to(device)
        length = 3
    cap = cv2.VideoCapture(str(data / 'source-1080p.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError('Could not read the 1080p source clock')
    cameras = shot_windows(output)
    suffix = '' if model_name == 'pickleball' else f'-{model_name}'
    cache = output / f'ball-detections{suffix}'
    cache.mkdir(exist_ok=True)
    try:
        for rally in intervals(output):
            start, end = rally['start'], rally['end']
            path = cache / f'{start:.3f}.json'
            if path.exists() and not refresh:
                continue
            detections = []
            for shot, first, last in samples(fps, start - .30, end + .26, cameras):
                if model_name == 'v3':
                    background = []
                    for index in np.linspace(first, last - 1, 12, dtype=int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                        ok, image = cap.read()
                        if not ok:
                            raise ValueError(f'Could not read background frame {index}')
                        background.append(cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
                    model.set_background(background)
                index = first
                for frames, _, valid in windows(cap, first, last - first, length):
                    if model_name == 'crop':
                        h, w = frames[0].shape[:2]
                        x, y = .20, .15 if shot['camera'] == 'end' else .18
                        frames = [frame[round(y*h):round((y+.60)*h), round(x*w):round((x+.60)*w)] for frame in frames]
                    if model_name == 'v3':
                        maps = model.infer(frames)
                    else:
                        with torch.inference_mode():
                            tensor = torch.from_numpy(preprocess(frames)[None]).to(device)
                            maps = model(tensor).cpu().numpy()[0]
                    for heatmap in maps[:valid]:
                        ball = heatmap_detection(heatmap, WIDTH, HEIGHT, threshold=.55 if model_name == 'v3' else .43)
                        if ball and model_name == 'v3':
                            ball['kind'] = 'TrackNetV3'
                        if ball and model_name == 'crop':
                            u, v = ball['pixel']
                            ball['pixel'] = [round(x * WIDTH + .60 * u, 2), round(y * HEIGHT + .60 * v, 2)]
                            ball['kind'] = 'TrackNet-Pickleball (court crop)'
                        detections.append({'frame': index, 't': round(index / fps, 6),
                                           'shot': shot['start'], 'ball': ball})
                        index += 1
            atomic_json(path, {'fps': fps, 'frames': detections})
            observed = sum(row['ball'] is not None for row in detections)
            print(f'{start:.2f}–{end:.2f}: {observed}/{len(detections)} model detections', flush=True)
    finally:
        cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'))
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--model', choices=['pickleball', 'v3', 'crop'], default='pickleball')
    args = parser.parse_args()
    track(args.data, args.data / 'match-five', args.device, args.refresh, args.model)
