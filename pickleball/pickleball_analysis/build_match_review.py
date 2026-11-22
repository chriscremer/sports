"""Analyze reviewed wide camera windows for the five-minute match viewer."""
import argparse
import json
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from complete_ball_track import match_ball_playback
from fit_match_ball import HEIGHT as PROJECTION_HEIGHT, WIDTH as PROJECTION_WIDTH, camera_samples
from geometry import LENGTH, WIDTH, calibrate, ground, lift_pose
from match_contacts import match_contact_ripples
from moving_camera import MovingCourt
from rally_timeline import rally_timeline
from refine_match_players import refine_match_players


def calibration(corners, width, height):
    pixels = np.array(corners, dtype=float) * [width, height]
    return calibrate(width, height, pixels)


def detect_players(batches, H, K, R, C):
    players = {'near': [], 'far': []}
    for result, offset, far_only in batches:
        if result.keypoints is None:
            continue
        for box, raw_joints in zip(result.boxes.data.cpu().numpy(), result.keypoints.data.cpu().numpy()):
            joints = raw_joints.copy()
            joints[:, :2] += offset
            ankles = [point for point in joints[15:17] if point[2] > .28]
            if ankles:
                # Keep lateral support centered but use the lowest visible foot
                # for depth when the other foot is lifted during a lunge.
                pixel = [float(np.average([p[0] for p in ankles], weights=[p[2] for p in ankles])),
                         min(box[3] + offset[1] + 3, max(p[1] for p in ankles) + .025 * (box[3] - box[1]))]
            else:
                pixel = [(box[0] + box[2]) / 2 + offset[0], box[3] + offset[1]]
            xy = ground(H, pixel)
            if not (-1.1 < xy[0] < WIDTH + 1.1 and -5 < xy[1] < LENGTH + 4):
                continue
            side = 'near' if xy[1] > LENGTH / 2 else 'far'
            if far_only and side != 'far':
                continue
            quality = float(box[4]) + .18 * float(joints[5:, 2].mean())
            players[side].append({'team': side, 'xy': xy.round(4).tolist(), 'joints3d': lift_pose(K, R, C, xy, joints),
                                  'confidence': round(float(box[4]), 3), 'quality': quality})
    selected = {}
    for side, candidates in players.items():
        selected[side] = []
        for candidate in sorted(candidates, key=lambda p: p['quality'], reverse=True):
            if any(np.linalg.norm(np.subtract(candidate['xy'], p['xy'])) < .47 for p in selected[side]):
                continue
            del candidate['quality']
            selected[side].append(candidate)
            if len(selected[side]) == 2:
                break
    return selected


class CameraRun:
    def __init__(self, window, camera, width, height, index):
        self.index = index
        self.fixed = calibration(camera['corners_normalized'], width, height) if 'corners_normalized' in camera else None
        self.previous = {'near': [], 'far': []}
        self.velocities = {}
        self.next_id = 0
        self.frames = []

    def record(self, t, batches, mapped):
        if mapped is None:
            return
        H, K, R, C, _ = mapped
        sides = detect_players(batches, H, K, R, C)
        rows = []
        for side, players in sides.items():
            prior = [(stamp, player) for stamp, player in self.previous[side] if t - stamp <= .65]
            if prior and players:
                distances = np.array([[np.linalg.norm(np.asarray(old['xy']) +
                    self.velocities.get(old['id'], np.zeros(2)) * min(.25, t - stamp) - new['xy'])
                    for new in players] for stamp, old in prior])
                first, second = linear_sum_assignment(distances)
                for i, j in zip(first, second):
                    if distances[i, j] <= .55 + 7 * (t - prior[i][0]):
                        players[j]['id'] = prior[i][1]['id']
                        stamp, previous = prior[i]
                        velocity = np.subtract(players[j]['xy'], previous['xy']) / (t - stamp)
                        previous_velocity = self.velocities.get(previous['id'], velocity)
                        self.velocities[previous['id']] = .45 * np.clip(velocity, -6, 6) + .55 * previous_velocity
            for p in players:
                if 'id' not in p:
                    self.next_id += 1
                    p['id'] = f'{self.index}-{self.next_id}'
                p['tracking_status'] = 'detected'
            ids = {p['id'] for p in players}
            self.previous[side] = [(t, p) for p in players] + [(stamp, p) for stamp, p in prior if p['id'] not in ids]
            rows.extend(players)
        self.frames.append({'t': round(t, 6), 'players': rows})


def build(data, output, samples_per_second, device, moving_only=False):
    config = json.loads((output / 'camera-windows.json').read_text())
    duration = config['duration']
    width, height = 1280, 720
    source_path = str(data / 'source-1080p.mp4')
    source = cv2.VideoCapture(source_path)
    fps = source.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError('Could not read the 1080p source clock')
    pose = YOLO(str(data / 'models/yolo11s-pose.pt'))
    all_frames = []
    if moving_only:
        cached_rate = float((output / 'pose-rate.txt').read_text())
        if cached_rate != samples_per_second:
            raise ValueError('Use the same sample rate as the existing fixed-camera run')
        fixed = [w for w in config['wide_windows'] if not config['cameras'][w['camera']].get('references')]
        all_frames = [f for f in json.loads((output / 'poses.json').read_text())
                      if any(w['start'] <= f['t'] < w['end'] for w in fixed)]
    for index, window in enumerate(config['wide_windows']):
        start, end = window['start'], window['end']
        camera = config['cameras'][window['camera']]
        moving = camera.get('references')
        if moving_only and not moving:
            continue
        run = CameraRun(window, camera, width, height, f"{window['camera']}-{index}")
        registration = None
        if moving:
            references = cv2.VideoCapture(source_path)
            try:
                registration = MovingCourt(references, moving, width, height)
            finally:
                references.release()
        first = int(np.ceil(start * fps))
        native = first
        source.set(cv2.CAP_PROP_POS_FRAMES, native)
        images, times = [], []

        def flush():
            if images:
                results = pose.predict(images, device=device, imgsz=768, conf=.27, verbose=False)
                batches = [[(result, (0, 0), False)] for result in results]
                crops_to_run = [(crop, True) for crop in camera.get('far_crops_normalized', [])]
                if registration:
                    crops_to_run.append(((.10, .20, .90, .85), False))
                for (x0, y0, x1, y1), far_only in crops_to_run:
                    x, top, right, bottom = round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)
                    crops = [image[top:bottom, x:right] for image in images]
                    detections = pose.predict(crops, device=device, imgsz=640, conf=.25, verbose=False)
                    for per_frame, result in zip(batches, detections, strict=True):
                        per_frame.append((result, (x, top), far_only))
                for time, image, per_frame in zip(times, images, batches, strict=True):
                    mapped = registration.locate(image, time) if registration else run.fixed
                    run.record(time, per_frame, mapped)
                images.clear()
                times.clear()

        for i in range(int(np.ceil((end - start) * samples_per_second))):
            target = max(first, round((start + i / samples_per_second) * fps))
            if target / fps >= end:
                break
            while native < target:
                if not source.grab():
                    raise ValueError(f'Source ended at frame {native}')
                native += 1
            ok, frame = source.read()
            if not ok:
                raise ValueError(f'Could not decode source frame {native}')
            native += 1
            images.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
            times.append(target / fps)
            if len(images) == 8:
                flush()
        flush()
        all_frames.extend(run.frames)
        print(f'{start:.2f}–{end:.2f}: {len(run.frames)} calibrated pose samples', flush=True)
    source.release()
    all_frames.sort(key=lambda frame: frame['t'])
    path = output / 'poses.partial.json'
    path.write_text(json.dumps(all_frames, separators=(',', ':')))
    path.replace(output / 'poses.json')
    (output / 'pose-rate.txt').write_text(str(samples_per_second))
    print(f'Saved {len(all_frames)} samples across {len(config["wide_windows"])} camera windows ({duration}s review).', flush=True)


def covered_windows(frames, windows, hz):
    spans = []
    tolerance = .85 / hz
    for window in windows:
        times = [f['t'] for f in frames if window['start'] <= f['t'] < window['end']]
        if not times:
            continue
        start = previous = times[0]
        for stamp in times[1:] + [None]:
            if stamp is None or stamp - previous > 2 * tolerance:
                spans.append({**window, 'start': max(window['start'], start - tolerance),
                              'end': min(window['end'], previous + tolerance)})
                start = stamp
            previous = stamp
    return spans


def source_projection(data, output, config, fps):
    windows = config['wide_windows']
    frame_ranges = [{'shot': window['start'], 'frame': frame}
                    for window in windows
                    for frame in (ceil(window['start'] * fps), ceil(window['end'] * fps) - 1)]
    matrices = camera_samples(data, output, [{'fps': fps, 'frames': frame_ranges}],
                              windows, config['cameras'], cache_name='overlay-cameras.json')
    shots = []
    for window in windows:
        start, end = window['start'], window['end']
        shot = matrices.get(start, {})
        if -1 in shot:
            shots.append({'start': start, 'end': end, 'matrix': shot[-1].ravel().tolist()})
        elif shot:
            samples = [[frame / fps, matrix.ravel().tolist()] for frame, matrix in sorted(shot.items())]
            shots.append({'start': start, 'end': end, 'samples': samples})
    return {'width': PROJECTION_WIDTH, 'height': PROJECTION_HEIGHT, 'shots': shots}


def assemble(data, output):
    config = json.loads((output / 'camera-windows.json').read_text())
    duration = config['duration']
    frames = json.loads((output / 'poses.json').read_text())
    hz = float((output / 'pose-rate.txt').read_text()) if (output / 'pose-rate.txt').exists() else 8
    initial = data / 'v2-ball-review'
    replay = json.loads((initial / 'replay-rally-55.json').read_text())
    reviewed = replay['ball_review']['rally_events']
    clip = json.loads((data / 'comparison/rally-55-models.json').read_text())['clip']
    camera_windows = covered_windows(frames, config['wide_windows'], hz)
    refine_match_players(frames, config['wide_windows'])
    rallies, timelines, rally_contacts = [], [], []
    ball_path = output / 'match-ball.json'
    match_ball = json.loads(ball_path.read_text()) if ball_path.exists() else {}

    def add_review(review, clip, span=None):
        events = review['events']
        rally_contacts.extend(event for event in events if event['kind'] != 'end')
        start, end = events[0]['source_seconds'], events[-1]['source_seconds']
        section = len(timelines)
        timelines.append(rally_timeline(review, clip, clip['source_start'], heading='Detailed hits', section=section))
        rallies.append({**(span or {}), 'start': start, 'end': end, 'section': section,
                        'detail_end': end + 1 / clip['fps']})

    add_review(reviewed, clip)
    reviewed_through = 0
    extra_rallies = output / 'rally-spans.json'
    if extra_rallies.exists():
        span_review = json.loads(extra_rallies.read_text())
        if isinstance(span_review, dict):
            reviewed_through = span_review['reviewed_through']
            span_review = span_review['rallies']
        for span in span_review:
            review_file = span.pop('hit_review', None)
            generated = match_ball.get('reviews', {}).get(f"{span['start']:.3f}")
            if review_file:
                section_review = json.loads((output / review_file).read_text())
                add_review(section_review, section_review['source_clip'], span)
            elif generated:
                add_review(generated, generated['source_clip'], span)
            else:
                rallies.append(span)
    rallies.sort(key=lambda rally: rally['start'])
    previous_end = 0
    for rally in rallies:
        if not previous_end <= rally['start'] < rally['end'] <= duration:
            raise ValueError(f"Invalid or overlapping rally span: {rally}")
        previous_end = rally['end']
    if not 0 <= reviewed_through <= duration:
        raise ValueError('Rally review coverage must end within the match viewer')
    source = {**replay['source'], 'local_video': '/media/source.mp4'}
    match = {
        'source': source, 'clip': {'start': 0, 'end': duration, 'hz': hz, 'width': 1280, 'height': 720},
        'frames': frames, 'camera_windows': camera_windows, 'reviewed_rallies': rallies,
        'rally_reviewed_through': reviewed_through,
        **{key: replay[key] for key in ('calibration', 'ball_events', 'ball_summary', 'ball_track', 'ball_segments', 'ball_review')},
    }
    # Both media players use match time. The replay's samples and flight endpoints
    # already carry source timestamps, while the standalone excerpt has an offset.
    match['ball_review']['source_start'] = 0
    if match_ball:
        match['ball_track'] = sorted(match['ball_track'] + match_ball['ball_track'], key=lambda row: row['t'])
        match['ball_segments'] = sorted(match['ball_segments'] + match_ball['segments'], key=lambda row: row['start'])
        match['ball_review']['bounces'] = sorted(match['ball_review']['bounces'] + match_ball['bounces'],
                                                key=lambda row: row['t'])
        match['ball_review']['volleys'] = match_ball.get('volleys', [])
        match['ball_events']['events'].extend({'t': event['source_seconds'], 'kind': event['kind']}
                                             for review in match_ball['reviews'].values() for event in review['events'])
        match['ball_summary']['model'] = 'TrackNet-Pickleball + TrackNetV3'
    match['ball_playback'] = match_ball_playback(match, rally_contacts)
    match['ball_review']['ripples'] = match_contact_ripples(
        match['ball_playback'], rally_contacts, match['ball_review'], config['wide_windows'])
    match['source_projection'] = source_projection(data, output, config, match['ball_review']['fps'])
    (output / 'match.json').write_text(json.dumps(match, separators=(',', ':')) + '\n')
    template = Path(__file__).with_name('web') / 'match-five.html'
    (output / 'index.html').write_text(template.read_text().replace('<!-- REVIEWED_RALLIES -->', ''.join(timelines)))
    print(f'Built five-minute review with {len(frames)} pose samples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'))
    parser.add_argument('--hz', type=float, default=8)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--assemble-only', action='store_true')
    parser.add_argument('--moving-only', action='store_true', help='Reuse the fixed-camera samples and process moving windows')
    args = parser.parse_args()
    output = args.data / 'match-five'
    if not args.assemble_only:
        build(args.data, output, args.hz, args.device, args.moving_only)
    assemble(args.data, output)
