"""Replay visually confirmed ball additions beside the original reconstruction."""
import argparse
import hashlib
import json
import math
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from ball_geometry import fit_flights, sample_segment
from rally_timeline import rally_timeline


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n')


def fit_review_windows(track, aligned, calibration, windows, first_allowed_time):
    segments = []
    for window in windows:
        events = []
        for anchor in window['anchors']:
            frame = anchor['frame']
            row = aligned[frame]
            event = {'t': row['t'], 'kind': anchor['kind'], 'height_prior_m': anchor['height_prior_m']}
            if anchor.get('pin_visible_pixel'):
                event['reviewed_pixel'] = row['ball']['pixel']
            events.append(event)
        times = [event['t'] for event in events]
        if times[0] < first_allowed_time or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError(f"Overlapping or unordered review window: {window['evidence']}")
        first_allowed_time = times[-1]
        fitted = fit_flights(track, events, calibration)
        for segment in fitted:
            if segment['reprojection_rmse_px'] > 6:
                continue
            segment.update(max_gap_s=.14, max_support_distance_s=.06, evidence=window['evidence'],
                           method='Approximate monocular flight with visually reviewed intervals and height priors')
            segments.append(segment)
    return segments


def build(data, review_path, clip_start=55):
    clip_id = f'rally-{clip_start:g}'
    baseline_path = data / f'analysis/{clip_id}.json'
    baseline_bytes = baseline_path.read_bytes()
    baseline_sha = hashlib.sha256(baseline_bytes).hexdigest()
    replay = json.loads(baseline_bytes)
    predictions = read_json(data / f'comparison/{clip_id}-models.json')
    clip = predictions['clip']
    review = read_json(review_path)
    events_path = review_path.with_name('rally-events.json')
    event_review = read_json(events_path) if events_path.exists() else None
    timeline = rally_timeline(event_review, clip) if event_review else ''
    if review['baseline_sha256'] != baseline_sha:
        raise ValueError('The baseline changed after this review')
    observations = review['observations']
    track = replay['ball_track']
    base_count = sum(bool(row['ball']) for row in track)
    aligned = {}
    for frame in range(clip['frame_count']):
        time = clip['source_start'] + frame / clip['fps']
        row = min(track, key=lambda row: abs(row['t'] - time))
        if abs(row['t'] - time) > .5 / clip['fps']:
            raise ValueError(f'Baseline is out of sync at excerpt frame {frame}')
        aligned[frame] = row

    matched = {name: 0 for name in predictions['models']}
    any_match = 0
    for obs in observations:
        frame = obs['index']
        if frame not in aligned or obs['source_frame'] != clip['first_frame'] + frame:
            raise ValueError(f'Invalid source frame for observation {frame}')
        row = aligned[frame]
        if row['ball'] is not None or abs(obs['source_seconds'] - row['t']) > .5 / clip['fps']:
            raise ValueError(f'Observation {frame} overlaps an existing mark or has the wrong time')
        x, y = obs['pixel_1080p']
        if obs['visibility'] != 'visible' or not (0 <= x < clip['width'] and 0 <= y < clip['height']):
            raise ValueError(f'Observation {frame} has no visible ball center')
        row['ball'] = {
            'pixel': [round(x * replay['clip']['width'] / clip['width'], 2),
                      round(y * replay['clip']['height'] / clip['height'], 2)],
            'kind': 'source frame grid review', 'tracking_status': 'reviewed',
            'height': None, 'evidence': obs,
        }
        hits = []
        for name, model in predictions['models'].items():
            candidates = model['frames'][frame]['candidates']
            if any(math.dist([x, y], candidate['pixel']) <= 12 for candidate in candidates):
                matched[name] += 1
                hits.append(name)
        obs['cached_models_with_candidate_within_12px'] = hits
        any_match += bool(hits)

    windows_path = review_path.with_name('flight-windows.json')
    windows = read_json(windows_path)['windows'] if windows_path.exists() else []
    last_v1_time = max((segment['end'] for segment in replay.get('ball_segments', [])), default=0)
    additional = fit_review_windows(track, aligned, replay['calibration'], windows, last_v1_time)
    v1_sampled = sum(bool(row.get('reconstruction')) for row in track)
    additional_sampled = 0
    for row in track:
        segment = next((s for s in additional if s['start'] <= row['t'] <= s['end']), None)
        if segment and min(abs(row['t'] - t) for t in segment['support_times']) <= segment['max_support_distance_s']:
            sample = sample_segment([segment], row['t'], max_gap=segment['max_gap_s'])
            if sample:
                row['review_reconstruction'] = sample
                additional_sampled += 1
    projection_only = sum(row['ball'] is not None and not (row.get('reconstruction') or row.get('review_reconstruction')) for row in track)

    # Use the existing preview of these exact source frames and keep v1 fits separate.
    replay['clip']['video'] = f"/media/{clip['video']}"
    replay['ball_review'] = {
        'source_start': clip['source_start'], 'fps': clip['fps'],
        'reviewed_additions': len(observations), 'baseline_sha256': baseline_sha,
        'additional_segments': additional,
        'rally_events': event_review,
    }
    report = {
        'review_only': True, 'inspected_frames': len(review['inspected_frames']),
        'baseline_markers': base_count, 'reviewed_additions': len(observations),
        'replay_markers': base_count + len(observations), 'frames': len(track),
        'candidate_radius_1080p': 12, 'matches_by_model': matched, 'matches_any_model': any_match,
        'original_fit_frames': v1_sampled, 'additional_fit_frames': additional_sampled,
        'additional_flights': len(additional), 'projection_only_frames': projection_only,
    }
    output = review_path.parent
    replay_name = f'replay-{clip_id}.json'
    write_json(output / replay_name, replay)
    write_json(output / 'report.json', report)

    template = Path(__file__).with_name('web') / 'index.html'
    html = template.read_text().replace('<body>', f'<body data-review-url="./{replay_name}">')
    if timeline:
        html = html.replace('</head>', '<link rel="stylesheet" href="/rally-timeline.css"></head>')
    html = html.replace('Rally Lab · Pickleball reconstruction', 'Rally Lab · V2 ball review')
    original_link = ('<a href="https://www.youtube.com/watch?v=tdGSHF5FseE" target="_blank" '
                     'rel="noreferrer">Original on YouTube ↗</a>')
    source_url = replay['source'].get('url') or ''
    source_link = ''
    if urlsplit(source_url).scheme in ('http', 'https'):
        source_link = f'<a href="{escape(source_url, quote=True)}" target="_blank" rel="noreferrer">Original video ↗</a>'
    html = html.replace(original_link, source_link)
    html = html.replace('<span id="source-time">00:55.000</span>',
                        '<span>Match time <span id="source-time">00:55.000</span></span>')
    html = html.replace('● Ball</span></div>', '● Fitted flight &nbsp; ○ Ground projection</span></div>')
    html = html.replace('Drag to orbit · Scroll to zoom</span></div>',
                        'Drag to orbit · Scroll to zoom</span><span id="ball-depth-note" class="stage-label" '
                        'style="top:auto; left:auto; right:12px; bottom:12px; letter-spacing:0; font-size:10px"></span></div>')
    html = html.replace('<section class="views">', timeline + '<section class="views">')
    (output / 'index.html').write_text(html)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    parser.add_argument('--data', type=Path, default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'))
    parser.add_argument('--clip-start', type=float, default=55, help='Nominal match time at the start of the review clip')
    args = parser.parse_args()
    build(args.data, args.review, args.clip_start)
