"""Review paddle contacts and fit match ball flights in the calibrated court."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, medfilt
from scipy.sparse import lil_matrix

from analyze_match_ball import HEIGHT, WIDTH, intervals
from ball_geometry import flight, sample_segment
from compare_ball_models import atomic_json
from geometry import LENGTH, WIDTH as COURT_WIDTH, calibrate
from moving_camera import MovingCourt

NET = LENGTH / 2
DRAG = .65


def court_projection(H):
    """Keep the measured court homography exact while estimating a vertical axis."""
    floor = np.linalg.inv(H)
    center = np.array([WIDTH / 2, HEIGHT / 2])
    horizontal = floor[:2, :2] - center[:, None] * floor[2, :2]
    a, b = horizontal.T
    z, w = floor[2, :2]
    lhs = np.array([a @ b, a @ a - b @ b])
    rhs = np.array([-z * w, w * w - z * z])
    inverse_focal2 = (lhs @ rhs) / max(lhs @ lhs, 1e-12)
    focal = np.clip(1 / np.sqrt(max(inverse_focal2, 1e-9)), .55 * WIDTH, 2.5 * WIDTH)
    K = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]])
    basis = np.linalg.solve(K, floor)
    floor /= np.sign(basis[2] @ [COURT_WIDTH / 2, NET, 1]) * np.linalg.norm(basis[:, :2], axis=0).mean()
    basis = np.linalg.solve(K, floor)
    vertical = np.cross(basis[:, 0], basis[:, 1])
    vertical /= np.linalg.norm(vertical)
    if np.linalg.solve(np.column_stack((basis[:, :2], vertical)), -basis[:, 2])[2] < 0:
        vertical *= -1
    return np.column_stack((floor[:, :2], K @ vertical, floor[:, 2]))


def camera_samples(data, output, tracks, windows, cameras, *, cache_name='ball-cameras.json'):
    """Sample calibration on the same clock as the detections, separately per shot."""
    cache = output / cache_name
    by_shot = group_shots(tracks)
    sizes = {str(shot): [min(r['frame'] for r in rows), max(r['frame'] for r in rows)]
             for shot, rows in by_shot.items()}
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get('method') == 2 and saved['windows'] == windows and saved['cameras'] == cameras and saved['sizes'] == sizes:
            return {float(shot): {int(frame): np.array(P) for frame, P in rows.items()} for shot, rows in saved['matrices'].items()}
    source = data / 'source-1080p.mp4'
    cap = cv2.VideoCapture(str(source))
    references = cv2.VideoCapture(str(source))
    matrices = {}
    try:
        for window in windows:
            shot = window['start']
            rows = by_shot.get(shot)
            if not rows:
                continue
            camera = cameras[window['camera']]
            if 'corners_normalized' in camera:
                H, _, _, _, _ = calibrate(WIDTH, HEIGHT, np.array(camera['corners_normalized']) * [WIDTH, HEIGHT])
                matrices[shot] = {-1: court_projection(H)}
                continue
            moving = MovingCourt(references, camera['references'], WIDTH, HEIGHT)
            matrices[shot] = {}
            first, last = rows[0]['frame'], rows[-1]['frame']
            selected = range(first, last + 1, 4)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            native = first
            fps = tracks[0]['fps']
            for frame in selected:
                while native < frame and cap.grab():
                    native += 1
                ok, image = cap.read()
                if not ok or native != frame:
                    break
                native += 1
                image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                try:
                    located = moving.locate(image, frame / fps)
                except (ValueError, cv2.error):
                    continue
                if located is None:
                    continue
                matrices[shot][frame] = court_projection(located[0])
    finally:
        cap.release()
        references.release()
    atomic_json(cache, {'method': 2, 'windows': windows, 'cameras': cameras, 'sizes': sizes,
                        'matrices': {shot: {frame: P.tolist() for frame, P in rows.items()} for shot, rows in matrices.items()}})
    return matrices


def group_shots(tracks):
    shots = {}
    for track in tracks:
        for row in track['frames']:
            shots.setdefault(row['shot'], []).append(row)
    return shots


def calibrated_rows(track, matrices, require_neighbors=True):
    populated = []
    for row in track:
        if not row['ball']:
            continue
        shot = matrices.get(row['shot'], {})
        frame = row['frame']
        if -1 in shot:
            matrix = shot[-1]
        else:
            before = next((i for i in range(frame, frame - 9, -1) if i in shot), None)
            after = next((i for i in range(frame, frame + 9) if i in shot), None)
            if before is None and after is None:
                continue
            if before is None or after is None or before == after:
                matrix = shot[before if before is not None else after]
            else:
                weight = (frame - before) / (after - before)
                matrix = (1 - weight) * shot[before] + weight * shot[after]
        pixel = np.array(row['ball']['pixel'])
        try:
            base, step = pixel_ray(matrix, pixel)
        except np.linalg.LinAlgError:
            continue
        proxy = base + .65 * step
        if not (-4 <= proxy[0] <= COURT_WIDTH + 4 and -8 <= proxy[1] <= LENGTH + 7):
            continue
        populated.append({**row, 'P': matrix, 'pixel': pixel, 'base': base, 'step': step, 'proxy': proxy})

    if not require_neighbors:
        return populated
    # A supported detection has at least one nearby observation in the same shot.
    result = []
    for i, row in enumerate(populated):
        neighbors = populated[max(i - 3, 0):i] + populated[i + 1:i + 4]
        if any(n['shot'] == row['shot'] and abs(n['frame'] - row['frame']) <= 3
               and np.linalg.norm(n['pixel'] - row['pixel']) < 8 + 26 * abs(n['frame'] - row['frame'])
               for n in neighbors):
            result.append(row)
    return result


def pixel_ray(matrix, pixel):
    equations = matrix[:2] - np.asarray(pixel)[:, None] * matrix[2]
    return np.linalg.solve(equations[:, :2], -equations[:, [3, 2]]).T


def add_secondary(primary, secondary):
    """Fill misses where the second model continues a supported local trajectory."""
    support = {row['frame']: row for row in primary}
    nearby = {row['frame']: row for row in secondary}
    for row in secondary:
        frame = row['frame']
        if frame in support:
            continue
        baseline = [support[i] for i in range(frame - 4, frame + 5) if i in support and support[i]['shot'] == row['shot']]
        others = [nearby[i] for i in range(frame - 2, frame + 3) if i != frame and i in nearby and nearby[i]['shot'] == row['shot']]
        if baseline and not any(np.linalg.norm(reference['pixel'] - row['pixel']) <= 6 + 17 * abs(reference['frame'] - frame)
                                for reference in baseline):
            continue
        if len(others) < 2 or not any(np.linalg.norm(other['pixel'] - row['pixel']) < 30 for other in others):
            continue
        support[frame] = row
    return sorted(support.values(), key=lambda row: row['frame'])


def contiguous(rows, max_gap=7):
    groups = []
    for row in rows:
        if not groups or row['shot'] != groups[-1][-1]['shot'] or row['frame'] - groups[-1][-1]['frame'] > max_gap:
            groups.append([])
        groups[-1].append(row)
    return groups


def fit_arc(rows, start, end, start_height=None, end_height=None):
    """Fit an airborne arc from the pixels on their individual calibrated rays."""
    reviewed = np.array([r['ball']['tracking_status'] == 'reviewed' for r in rows])
    minimum = 4 if reviewed.sum() >= 3 else 5
    if len(rows) < minimum or end - start < .12:
        return None
    times = np.array([r['t'] for r in rows])
    pixels = np.stack([r['pixel'] for r in rows])
    projection = np.stack([r['P'] for r in rows])
    confidence = np.where(reviewed, 2.5, 1.)
    duration = end - start
    fraction = -np.expm1(-DRAG * (times - start)) / -np.expm1(-DRAG * duration)
    baseline = flight(np.zeros(3), np.zeros(3), duration, DRAG, times - start)[0]
    eq = projection[:, :2] - pixels[:, :, None] * projection[:, 2, None, :]
    directions = eq[:, :, :3]
    rhs = -eq[:, :, 3] - np.einsum('nij,nj->ni', directions, baseline)
    design = np.concatenate((directions * (1 - fraction[:, None, None]), directions * fraction[:, None, None]), axis=2)
    low, high = np.tile([-3.5, -7.5, .027], 2), np.tile([COURT_WIDTH + 3.5, LENGTH + 6, 4.5], 2)
    priors = [start_height or (.65, 2.), end_height or (.65, 2.)]
    prior_values, prior_weights = np.array(priors).T
    anchors = np.zeros((2, 6))
    anchors[0, 2] = anchors[1, 5] = 1.
    initial = np.linalg.lstsq(np.vstack((design.reshape(-1, 6) / 15, anchors * prior_weights[:, None])),
                             np.r_[rhs.ravel() / 15, prior_values * prior_weights], rcond=None)[0]

    def errors(parameters):
        xyz = baseline + (1 - fraction[:, None]) * parameters[:3] + fraction[:, None] * parameters[3:]
        image = np.einsum('nij,nj->ni', projection, np.column_stack((xyz, np.ones(len(xyz)))))
        return image[:, :2] / np.maximum(.1, image[:, 2:]) - pixels

    def residual(parameters, keep):
        return np.r_[(errors(parameters)[keep] * confidence[keep, None]).ravel(), (parameters[[2, 5]] - prior_values) * prior_weights]

    fit = least_squares(residual, np.clip(initial, low + 1e-5, high - 1e-5), args=(slice(None),),
                        bounds=(low, high), loss='soft_l1', f_scale=3, max_nfev=50)
    keep = np.linalg.norm(errors(fit.x), axis=1) <= 9
    if keep.sum() < max(minimum, .62 * len(rows)) or np.ptp(times[keep]) < .10:
        return None
    if not keep.all():
        fit = least_squares(residual, fit.x, args=(keep,), bounds=(low, high), loss='soft_l1', f_scale=3, max_nfev=30)
    p0, p1 = fit.x.reshape(2, 3)
    path, velocities = flight(p0, p1, duration, DRAG, np.linspace(0, duration, 25))
    rmse = np.sqrt(np.mean(np.sum(errors(fit.x)[keep] ** 2, axis=1)))
    if rmse > 7 or path[:, 2].max() > 4.5 or np.linalg.norm(velocities, axis=1).max() > 44:
        return None
    if start_height and start_height[1] > 80 and p0[2] > .15:
        return None
    if end_height and end_height[1] > 80 and p1[2] > .15:
        return None
    return {'start': start, 'end': end, 'p0': p0.round(5).tolist(), 'p1': p1.round(5).tolist(),
            'camera_shot': rows[0]['shot'], 'drag_per_second': DRAG, 'support_times': times[keep].tolist(),
            'reprojection_rmse_px': round(float(rmse), 2), 'observations': int(keep.sum()),
            'max_gap_s': .26, 'max_support_distance_s': .105,
            'method': 'estimated flight from detected pixels and court calibration'}


def flight_fragments(rows, start, end, start_height, end_height):
    """Try a single flight, then a floor bounce, then shorter supported fits."""
    single = fit_arc(rows, start, end, start_height, end_height)
    if single and single['reprojection_rmse_px'] <= 3.5 and single['observations'] >= .9 * len(rows):
        return [single], []
    best = None
    floor = (.037, 145.)
    minimum = 5
    if len(rows) >= 2 * minimum and end - start > .3:
        for index in range(minimum, len(rows) - minimum + 1, 2):
            time = (rows[index - 1]['t'] + rows[index]['t']) / 2
            left = fit_arc(rows[:index], start, time, start_height, floor)
            if left is None:
                continue
            right = fit_arc(rows[index:], time, end, floor, end_height)
            if right is None:
                continue
            mismatch = np.linalg.norm(np.subtract(left['p1'], right['p0']))
            if mismatch > .31 or min(left['p0'][2], right['p1'][2]) < .12:
                continue
            score = sum(f['reprojection_rmse_px'] * f['observations'] + 9 for f in (left, right))
            score += 9 * (len(rows) - left['observations'] - right['observations'])
            if best is None or score < best[0]:
                best = score, left, right
    if single:
        cost = single['reprojection_rmse_px'] * single['observations'] + 9 * (len(rows) - single['observations'])
        if best is None or cost <= best[0]:
            return [single], []
    if best:
        _, left, right = best
        bounce = ((np.asarray(left['p1']) + right['p0']) / 2).round(5).tolist()
        left['p1'] = right['p0'] = bounce
        return [left, right], [{'t': left['end'], 'xy': bounce[:2]}]
    if len(rows) < 12:
        return [], []
    middle = len(rows) // 2
    time = (rows[middle - 1]['t'] + rows[middle]['t']) / 2
    left, left_bounces = flight_fragments(rows[:middle], start, time, start_height, None)
    right, right_bounces = flight_fragments(rows[middle:], time, end, None, end_height)
    return left + right, left_bounces + right_bounces


def reconstruct(rows, boundaries):
    segments, bounces = [], []
    for before, after in zip(boundaries, boundaries[1:]):
        a, b = before['source_seconds'], after['source_seconds']
        between = [r for r in rows if a - .000001 <= r['t'] <= b + .000001]
        for group in contiguous(between, 9):
            start, end = max(group[0]['t'] - .017, a), min(group[-1]['t'] + .017, b)
            # Contacts constrain height loosely. The final event may be a fault or out ball.
            first = (.78 if before['kind'] == 'serve' else .65, 7.) if start - a < .10 else None
            last = (.65, 7.) if after['kind'] == 'hit' and b - end < .10 else None
            found, contacts = flight_fragments(group, start, end, first, last)
            segments.extend(found)
            bounces.extend(contacts)
    segments = join_flights(segments, rows, bounces)
    segments = recover_reviewed_arcs(rows, segments, boundaries)
    return segments, [event for event in bounces if any(s['end'] == event['t'] for s in segments)
                      and any(s['start'] == event['t'] for s in segments)]


def recover_reviewed_arcs(rows, segments, events):
    """Recover short arcs around confirmed centers within gaps in the first fit."""
    limits = sorted({event['source_seconds'] for event in events}
                    | {t for segment in segments for t in (segment['start'], segment['end'])})
    added = []
    for start, end in zip(limits, limits[1:]):
        if end - start < .125 or any(s['start'] < end and start < s['end'] for s in segments):
            continue
        missing = [row for row in rows if start <= row['t'] <= end]
        for group in contiguous(missing, 5):
            first = 0
            while len(group) - first >= 4:
                recover = None
                for stop in range(min(len(group), first + 12), first + 3, -1):
                    part = group[first:stop]
                    if sum(row['ball']['tracking_status'] == 'reviewed' for row in part) < 2:
                        continue
                    a, b = max(start, part[0]['t'] - .016), min(end, part[-1]['t'] + .016)
                    nearest = [s for s in segments + added if s['camera_shot'] == part[0]['shot']]
                    before = max((s for s in nearest if s['end'] <= a and a - s['end'] < .15),
                                 key=lambda s: s['end'], default=None)
                    after = min((s for s in nearest if s['start'] >= b and s['start'] - b < .15),
                                key=lambda s: s['start'], default=None)
                    p0 = (before['p1'][2], 12.) if before else None
                    p1 = (after['p0'][2], 12.) if after else None
                    arc = fit_arc(part, a, b, p0, p1)
                    if not arc or arc['reprojection_rmse_px'] > 5.5:
                        continue
                    if before and np.linalg.norm(np.subtract(before['p1'], arc['p0'])) > .6 + 18 * (a - before['end']):
                        continue
                    if after and np.linalg.norm(np.subtract(after['p0'], arc['p1'])) > .6 + 18 * (after['start'] - b):
                        continue
                    arc['method'] = 'short flight supported by reviewed image centers'
                    recover = arc
                    break
                if recover:
                    added.append(recover)
                    first = stop
                else:
                    first += 1
    return sorted(segments + added, key=lambda segment: segment['start'])


def join_flights(segments, rows, bounces):
    """Share the position at an adjacent contact if both image tracks support it."""
    chains = []
    for segment in segments:
        if (not chains or segment['camera_shot'] != chains[-1][-1]['camera_shot']
                or abs(segment['start'] - chains[-1][-1]['end']) > .000001):
            chains.append([])
        chains[-1].append(segment)

    by_time = {row['t']: row for row in rows}
    bounce_times = {event['t'] for event in bounces}
    for chain in chains:
        if len(chain) < 2:
            continue
        middle = [(np.array(left['p1']) + right['p0']) / 2 for left, right in zip(chain, chain[1:])]
        initial = np.array([chain[0]['p0'], *middle, chain[-1]['p1']])
        stamps = [segment['start'] for segment in chain] + [chain[-1]['end']]
        weights = np.full(initial.shape, 2.)
        for i, stamp in enumerate(stamps):
            if stamp in bounce_times:
                initial[i, 2] = .037
                weights[i, 2] = 145.

        groups = []
        for segment in chain:
            supported = [by_time[time] for time in segment['support_times']]
            times = np.array([row['t'] for row in supported]) - segment['start']
            groups.append((times, np.stack([row['pixel'] for row in supported]),
                           np.stack([row['P'] for row in supported]), segment['end'] - segment['start'],
                           np.array([2.5 if row['ball']['tracking_status'] == 'reviewed' else 1. for row in supported])))

        def projection_errors(nodes, i):
            times, pixels, matrices, duration, _ = groups[i]
            xyz, _ = flight(nodes[i], nodes[i + 1], duration, DRAG, times)
            image = np.einsum('nij,nj->ni', matrices, np.column_stack((xyz, np.ones(len(xyz)))))
            return image[:, :2] / np.maximum(.1, image[:, 2:]) - pixels

        def residual(flat):
            nodes = flat.reshape(-1, 3)
            return np.concatenate([*((projection_errors(nodes, i) * groups[i][-1][:, None]).ravel() for i in range(len(chain))),
                                   ((nodes - initial) * weights).ravel()])

        count = sum(2 * len(group[0]) for group in groups)
        sparsity = lil_matrix((count + initial.size, initial.size), dtype=bool)
        start = 0
        for i, group in enumerate(groups):
            stop = start + 2 * len(group[0])
            sparsity[start:stop, 3*i:3*i+6] = True
            start = stop
        indices = np.arange(initial.size)
        sparsity[count + indices, indices] = True
        low = np.tile([-3.5, -7.5, .027], len(initial))
        high = np.tile([COURT_WIDTH + 3.5, LENGTH + 6, 4.5], len(initial))
        fit = least_squares(residual, initial.ravel(), bounds=(low, high), loss='soft_l1',
                            f_scale=3, jac_sparsity=sparsity.tocsr(), max_nfev=65)
        nodes = fit.x.reshape(-1, 3)
        errors = [np.sqrt(np.mean(np.sum(projection_errors(nodes, i) ** 2, axis=1))) for i in range(len(chain))]
        if any(error > 8 for error in errors):
            continue
        for i, segment in enumerate(chain):
            duration = segment['end'] - segment['start']
            path, velocity = flight(nodes[i], nodes[i + 1], duration, DRAG, np.linspace(0, duration, 12))
            if path[:, 2].max() > 4.5 or np.linalg.norm(velocity, axis=1).max() > 44:
                break
        else:
            for i, (segment, error) in enumerate(zip(chain, errors, strict=True)):
                segment.update(p0=nodes[i].round(5).tolist(), p1=nodes[i + 1].round(5).tolist(),
                               reprojection_rmse_px=round(float(error), 2))
            for event in bounces:
                if event['t'] in stamps:
                    event['xy'] = nodes[stamps.index(event['t']), :2].round(5).tolist()

    kept = []
    for segment in segments:
        if kept and segment['camera_shot'] == kept[-1]['camera_shot']:
            gap = segment['start'] - kept[-1]['end']
            distance = np.linalg.norm(np.subtract(segment['p0'], kept[-1]['p1']))
            if gap < .12 and distance > .75 + 25 * gap:
                if segment['observations'] <= kept[-1]['observations']:
                    continue
                kept.pop()
        kept.append(segment)
    return kept


def contact_candidates(rows):
    candidates = []
    for group in contiguous(rows, 12):
        if len(group) < 7:
            continue
        start, end = group[0]['frame'], group[-1]['frame']
        frames = np.arange(start, end + 1)
        longitudinal = np.interp(frames, [r['frame'] for r in group], [r['proxy'][1] for r in group])
        smooth = medfilt(np.pad(longitudinal, 2, mode='edge'), 5)[2:-2]
        for team, sign in [('near', 1), ('far', -1)]:
            peaks, details = find_peaks(sign * smooth, prominence=.7, distance=6, wlen=45)
            for peak, strength in zip(peaks, details['prominences']):
                if sign * (smooth[peak] - NET) < -.15:
                    continue
                row = min(group, key=lambda r: abs(r['frame'] - frames[peak]))
                candidates.append({'team': team, 't': row['t'], 'strength': round(float(strength), 2), 'y': round(float(smooth[peak]), 2)})
    return sorted(candidates, key=lambda row: row['t'])


def apply_grid_review(track, rows, review, matrices):
    reviewed = []
    for source in track:
        point = review.get(source['frame'])
        if point:
            ball = {'pixel': point['pixel_960'], 'tracking_status': 'reviewed',
                    'kind': 'Source frame grid review', 'sheet': point['sheet'], 'height': None}
            reviewed.append({**source, 'ball': ball})
    if not reviewed:
        return rows
    revised = {row['frame']: row for row in rows}
    anchors = calibrated_rows(reviewed, matrices, require_neighbors=False)
    for anchor in anchors:
        revised[anchor['frame']] = anchor
    # Between close reviewed centers, discard an inconsistent model peak.
    for before, after in zip(anchors, anchors[1:]):
        first, last = before['frame'], after['frame']
        if before['shot'] != after['shot'] or not 1 < last - first <= 5:
            continue
        for frame in range(first + 1, last):
            row = revised.get(frame)
            if row is None:
                continue
            amount = (frame - first) / (last - first)
            interpolated = (1 - amount) * before['pixel'] + amount * after['pixel']
            if np.linalg.norm(row['pixel'] - interpolated) > 10:
                del revised[frame]
    return sorted(revised.values(), key=lambda row: row['frame'])


def datasets(data, output, grid_review=None):
    config = json.loads((output / 'camera-windows.json').read_text())
    rallies = intervals(output)
    tracks = [json.loads((output / 'ball-detections' / f"{rally['start']:.3f}.json").read_text()) for rally in rallies]
    matrices = camera_samples(data, output, tracks, config['wide_windows'], config['cameras'])
    for rally, track in zip(rallies, tracks, strict=True):
        primary = calibrated_rows(track['frames'], matrices)
        for variant in ('crop', 'v3'):
            secondary = output / f'ball-detections-{variant}' / f"{rally['start']:.3f}.json"
            if secondary.exists():
                primary = add_secondary(primary, calibrated_rows(json.loads(secondary.read_text())['frames'], matrices))
        yield rally, track, apply_grid_review(track['frames'], primary, grid_review or {}, matrices)


def hit_review(rally, listed, fps):
    first = round((rally['start'] - .32) * fps)
    last = round((rally['end'] + .30) * fps)
    clip = {'id': f"rally-{rally['start']:.3f}", 'first_frame': first, 'frame_count': last - first,
            'source_start': first / fps, 'duration': (last - first) / fps, 'fps': fps}
    events = []
    contacts = [(t, team) for t, team in listed] + [(rally['end'], None)]
    for i, (time, team) in enumerate(contacts):
        frame = round(time * fps)
        kind = 'end' if team is None else 'serve' if i == 0 else 'hit'
        evidence = ('End of play estimated from the source review.' if kind == 'end' else
                    'Serve timing estimated from the broadcast.' if kind == 'serve' else
                    'Approximate paddle contact from ball motion and available broadcast audio.')
        events.append({'kind': kind, 'team': team, 'source_frame': frame, 'source_seconds': round(frame / fps, 6),
                       'frame': frame - first, 'frame_interval': [max(0, frame - first - 4), min(last - first - 1, frame - first + 4)],
                       'evidence': evidence})
    return {'clip': clip['id'], 'source_clip': clip, 'events': events}


def volley_contacts(boundaries, flights, bounces, rows, manual):
    ripples = []
    # The first two returns after the serve are preceded by a bounce.
    for previous, event in zip(boundaries[2:], boundaries[3:]):
        if event['kind'] != 'hit':
            continue
        start, end = previous['source_seconds'], event['source_seconds']
        path = [s for s in flights if s['start'] >= start - .02 and s['end'] <= end + .02 and s['end'] > start]
        if not path or path[0]['start'] - start > .075 or end - path[-1]['end'] > .075:
            continue
        if min(min(s['p0'][2], s['p1'][2]) for s in path) < .18 or any(start < b['t'] < end for b in bounces):
            continue
        if any(b['start'] - a['end'] > .055 or np.linalg.norm(np.subtract(a['p1'], b['p0'])) > .35
               for a, b in zip(path, path[1:])):
            continue
        support = sorted({t for segment in path for t in segment['support_times'] if start <= t <= end})
        position = path[-1]['p1']
        side = 1 if event['team'] == 'near' else -1
        if max(np.diff([start, *support, end])) <= .14 and side * (position[1] - NET) >= -.4:
            ripples.append({'t': end, 'xyz': path[-1]['p1'], 'method': 'unbroken airborne flight fit'})

    for note in manual:
        frame, time = note['frame'], note['t']
        if not boundaries[0]['source_seconds'] <= time <= boundaries[-1]['source_seconds']:
            continue
        position = sample_segment(flights, time, max_gap=.26)
        if position:
            xyz = position['xyz']
        elif note.get('pixel_960'):
            nearest = min(rows, key=lambda row: abs(row['frame'] - frame), default=None)
            if nearest is None or abs(nearest['frame'] - frame) > 8:
                continue
            height = note['height_prior_m']
            base, step = pixel_ray(nearest['P'], note['pixel_960'])
            xyz = np.r_[base + height * step, height].round(5).tolist()
        else:
            nearest = min((row for row in rows if row['ball']['tracking_status'] == 'reviewed'),
                          key=lambda row: abs(row['frame'] - frame), default=None)
            if nearest is None or abs(nearest['frame'] - frame) > 3:
                continue
            xyz = np.r_[nearest['base'] + .95 * nearest['step'], .95].round(5).tolist()
        ripples = [r for r in ripples if abs(r['t'] - time) > .15]
        ripples.append({'t': time, 'xyz': xyz, 'method': 'grid-reviewed volley'})
    return sorted(ripples, key=lambda r: r['t'])


def build(data, output, hits, candidates_only):
    reviewed = hits.get('rallies', {})
    review_file = output / 'ball-grid-observations.json'
    grid_review = json.loads(review_file.read_text()) if review_file.exists() else {}
    grid = {point['frame']: point for point in grid_review.get('observations', [])}
    volley_notes = grid_review.get('volleys', [{'frame': frame} for frame in grid_review.get('volley_frames', [])])
    match = {'ball_track': [], 'segments': [], 'bounces': [], 'volleys': [], 'reviews': {}, 'rallies': []}
    for rally, track, rows in datasets(data, output, grid):
        start, end = rally['start'], rally['end']
        key = f'{start:.3f}'
        if candidates_only:
            candidates = contact_candidates(rows)
            print(f'{start:.2f}–{end:.2f}  {len(rows)}/{len(track["frames"])}: ' +
                  ' '.join(f"{r['t']:.2f}{r['team'][0]}({r['strength']:.1f})" for r in candidates), flush=True)
            continue
        if rally.get('hit_review'):
            review = json.loads((output / rally['hit_review']).read_text())
        else:
            review = hit_review(rally, reviewed[key], track['fps'])
            match['reviews'][key] = review
        boundaries = review['events']
        a, b = boundaries[0]['source_seconds'], boundaries[-1]['source_seconds']
        flights, bounces = reconstruct(rows, boundaries)
        manual_volleys = [{**note, 't': round(note['frame'] / track['fps'], 6)} for note in volley_notes]
        volleys = volley_contacts(boundaries, flights, bounces, rows, manual_volleys)
        match['segments'].extend(flights)
        match['bounces'].extend(bounces)
        match['volleys'].extend(volleys)
        supported = 0
        observed = {r['frame']: r for r in rows}
        first, last = boundaries[0]['source_frame'], boundaries[-1]['source_frame']
        for frame in range(first, last + 1):
            t = round(frame / track['fps'], 6)
            detection = observed.get(frame)
            ball = {**detection['ball']} if detection else None
            estimate = sample_segment(flights, t, max_gap=.26)
            if estimate:
                supported += 1
            if ball:
                xy = detection['base']
                ball['court_ray'] = [xy.round(5).tolist(), detection['step'].round(5).tolist()]
                if (-3 <= xy[0] <= COURT_WIDTH + 3 and -4 <= xy[1] <= LENGTH + 4):
                    ball['ground_xy'] = xy.round(5).tolist()
                # Reject candidates which disagree with a supported flight.
                if estimate and ball['tracking_status'] != 'reviewed':
                    position = detection['P'] @ np.r_[estimate['xyz'], 1]
                    if np.linalg.norm(position[:2] / position[2] - detection['pixel']) > 10:
                        ball = None
            match['ball_track'].append({'t': t, 'ball': ball, 'reconstruction': estimate})
        match['rallies'].append({'start': a, 'end': b, 'hits': sum(e['kind'] != 'end' for e in boundaries),
                                 'supported_frames': supported, 'total_frames': last - first + 1})
        print(f'{a:.2f}–{b:.2f}: {len(flights)} flights, {supported}/{last - first + 1} frames with estimated height; '
              f'{len(bounces)} inferred bounces, {len(volleys)} volleys', flush=True)
    if not candidates_only:
        opening = data / 'v2-ball-review' / 'replay-rally-55.json'
        if opening.exists():
            clip = json.loads(opening.read_text())
            review = clip['ball_review']
            flights = sorted(clip['ball_segments'] + review['additional_segments'], key=lambda s: s['start'])
            match['volleys'].extend(volley_contacts(review['rally_events']['events'], flights, review['bounces'], [], []))
            match['volleys'].sort(key=lambda event: event['t'])
        atomic_json(output / 'match-ball.json', match)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'))
    parser.add_argument('--candidates', action='store_true')
    args = parser.parse_args()
    output = args.data / 'match-five'
    hits = {} if args.candidates else json.loads((output / 'hit-candidates.json').read_text())
    build(args.data, output, hits, args.candidates)
