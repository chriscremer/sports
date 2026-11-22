"""Fill the match playback track with labeled, evidence-guided predictions."""
from itertools import groupby

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from ball_geometry import sample_segment
from geometry import LENGTH, WIDTH

NET = LENGTH / 2
BALL_RADIUS = .037
MAX_SPEED = 42.
LOW = np.array([-3.5, -6., BALL_RADIUS])
HIGH = np.array([WIDTH + 3.5, LENGTH + 6., 5.])


def source_times(start, end, fps):
    first = round(start * fps)
    last = round(end * fps)
    frames = (round(frame / fps, 6) for frame in range(first, last + 1))
    return np.array([start, *(time for time in frames if start < time < end), end])


def fitted_sample(segments, time):
    for segment in segments:
        if segment['start'] <= time <= segment['end']:
            sample = sample_segment([segment], time, max_gap=segment.get('max_gap_s', .35))
            if sample:
                return sample
    return None


def contact_position(event, guide, poses):
    team = event['team']
    to_near = team == 'near'
    serve = event['kind'] == 'serve'
    backup = guide.copy()
    y = LENGTH - .4 if to_near else .4
    backup[1] = y if serve else (max(NET + .24, guide[1]) if to_near else min(NET - .24, guide[1]))
    backup[2] = np.clip(backup[2], .45, 1.8)
    if not poses:
        return backup
    nearby = min(poses, key=lambda row: abs(row['t'] - event['source_seconds']))
    if abs(nearby['t'] - event['source_seconds']) > .32:
        return backup
    players = [p for p in nearby['players'] if p['team'] == team]
    if serve:
        players = [p for p in players if abs(p['xy'][1] - y) <= 3.5]
    wrists = [np.array(wrist) for player in players for wrist in player['joints3d'][9:11] if wrist is not None]
    if not wrists:
        return backup
    wrist = min(wrists, key=lambda point: np.linalg.norm(point - backup))
    # The wrists are pose estimates. Keep the neighboring ball flight influential.
    result = .72 * wrist + .28 * backup
    result[2] = np.clip(result[2], .4, 2.3)
    return result


def extend_edge(sample, seconds):
    xyz, velocity = np.asarray(sample['xyz']), np.asarray(sample['velocity'])
    # Dampen open-ended extrapolation, especially after the last visible return.
    speed = np.linalg.norm(velocity)
    if speed > 18:
        velocity *= 18 / speed
    elapsed = np.sign(seconds) * .27 * -np.expm1(-abs(seconds) / .27)
    position = xyz + velocity * elapsed + np.array([0., 0., -4.905 * elapsed**2])
    return np.clip(position, LOW, HIGH)


def reach_landmark(landmarks, time, position):
    before = max(t for t in landmarks if t < time)
    after = min(t for t in landmarks if t > time)
    return (np.linalg.norm(position - landmarks[before]) <= MAX_SPEED * (time - before) + .25
            and np.linalg.norm(position - landmarks[after]) <= MAX_SPEED * (after - time) + .25)


def arc_between(times, start, end, start_position, end_position):
    fraction = (times - start) / (end - start)
    positions = start_position + fraction[:, None] * (end_position - start_position)
    # A bounded gravity arc fills the depth gap between the endpoints.
    rise = min(1.25, 9.81 * (end - start)**2 / 8)
    positions[:, 2] += 4 * fraction * (1 - fraction) * rise
    y0, y1 = start_position[1], end_position[1]
    if (y0 - NET) * (y1 - NET) < 0:
        crossing = (NET - y0) / (y1 - y0)
        height = (1 - crossing) * start_position[2] + crossing * end_position[2] + 4 * crossing * (1 - crossing) * rise
        # Raise the predicted arc only near a crossing, preserving its endpoints.
        if .06 < crossing < .94 and height < .88:
            ramp = np.minimum(fraction / crossing, (1 - fraction) / (1 - crossing))
            positions[:, 2] += (.88 - height) * ramp
    return positions


def legacy_ray(ball, projection):
    pixel = np.asarray(ball['pixel'])
    equations = projection[:2] - pixel[:, None] * projection[2]
    try:
        return np.linalg.solve(equations[:, :2], -equations[:, [3, 2]]).T
    except np.linalg.LinAlgError:
        return None


def net_crossings(path, times, contacts):
    hit_times = [event['source_seconds'] for event in contacts]
    for i, (a, b) in enumerate(zip(path, path[1:])):
        if (a[1] - NET) * (b[1] - NET) >= 0:
            continue
        u = (NET - a[1]) / (b[1] - a[1])
        x, _, z = a * (1 - u) + b * u
        if not 0 <= x <= WIDTH or z >= .89:
            continue
        time = (1 - u) * times[i] + u * times[i + 1]
        following = np.searchsorted(hit_times, time)
        if not 0 < following < len(contacts):
            continue
        sender, receiver = contacts[following - 1]['team'], contacts[following]['team']
        if sender != receiver and (b[1] > a[1]) == (receiver == 'near'):
            yield i, u


def refine_path(positions, times, detections, legacy_projection, contacts, contact_times, predicted):
    """Optimize one continuous path under image-ray and motion constraints."""
    rays = []
    for i, time in enumerate(times):
        ball = detections.get(time)
        if not ball:
            continue
        ray = ball.get('court_ray')
        if ray is None and legacy_projection is not None:
            ray = legacy_ray(ball, legacy_projection)
        if ray is not None:
            rays.append((i, *np.asarray(ray), ball['tracking_status'] == 'reviewed', ball.get('model_score', .5)))
    if not rays:
        return positions

    row_ids, columns, coefficients, targets = [], [], [], []

    def constraint(indices, values, target, weight):
        row_ids.extend([len(targets)] * len(indices))
        columns.extend(indices)
        coefficients.extend(weight * value for value in values)
        targets.append(weight * target)

    for i, (point, uncertain) in enumerate(zip(positions, predicted, strict=True)):
        for axis, target in enumerate(point):
            constraint([3*i + axis], [1.], target, .39 if uncertain else .95)

    rate = np.median(np.diff(times))
    for i in range(1, len(times) - 1):
        before, after = times[i] - times[i - 1], times[i + 1] - times[i]
        weights = [rate / before, -rate / before - rate / after, rate / after]
        near_contact = any(abs(times[i] - time) < .075 for time in contact_times)
        for axis in range(3):
            gravity = -9.81 * (before + after) * rate / 2 if axis == 2 and not near_contact else 0.
            constraint([3*(i - 1) + axis, 3*i + axis, 3*(i + 1) + axis], weights, gravity,
                       2.6 if near_contact else 12.)

    fixed_rows, fixed_terms = len(targets), len(coefficients)
    path = positions
    for _ in range(4):
        del targets[fixed_rows:]
        del row_ids[fixed_terms:], columns[fixed_terms:], coefficients[fixed_terms:]
        for i, base, step, reviewed, score in rays:
            scale = np.sqrt(1. + step**2)
            residual = (path[i, :2] - base - path[i, 2] * step) / scale
            deviation = np.linalg.norm(residual)
            # Trackers sometimes follow a paddle, court lettering, or a spectator.
            # Reviewed centers stay strong even if the first path misses them.
            weight = 24. if reviewed else (2.2 + 1.7 * score) / (1. + (deviation / .42)**2)
            for axis in range(2):
                constraint([3*i + axis, 3*i + 2], [1., -step[axis]], base[axis], weight / scale[axis])
        for i, point in enumerate(path):
            for axis in np.flatnonzero((point < LOW) | (point > HIGH)):
                constraint([3*i + axis], [1.], np.clip(point[axis], LOW[axis], HIGH[axis]), 20.)
        # Apply clearance between opposite-side hits. The last ball may hit the net.
        for i, fraction in net_crossings(path, times, contacts):
            constraint([3*i + 2, 3*(i + 1) + 2], [1 - fraction, fraction], .89, 14.)
        system = coo_matrix((coefficients, (row_ids, columns)), shape=(len(targets), positions.size)).tocsr()
        path = spsolve((system.T @ system).tocsc(), system.T @ targets).reshape(-1, 3)
    return np.clip(path, LOW, HIGH)


def fill_gap(times, samples, first, stop, contacts, impacts, poses):
    left = samples[first - 1] if first > 0 else None
    right = samples[stop] if stop < len(samples) else None
    start = times[first - 1] if left is not None else times[first]
    end = times[stop] if right is not None else times[stop - 1]
    p0 = np.asarray(left['xyz']) if left else None
    p1 = np.asarray(right['xyz']) if right else None
    if p0 is None:
        p0 = extend_edge(right, start - end) if right else np.array([WIDTH / 2, 0., .7])
    if p1 is None:
        p1 = extend_edge(left, end - start) if left else np.array([WIDTH / 2, LENGTH, .45])

    landmarks = {start: p0, end: p1}
    if first == 0:
        serve = next((event for event in contacts if event['kind'] == 'serve'), None)
        if serve:
            landmarks[start] = contact_position(serve, p0, poses)
    for event in contacts:
        time = event['source_seconds']
        if event['kind'] != 'hit' or not start + .09 < time < end - .09:
            continue
        guide = p0 + (time - start) / (end - start) * (p1 - p0)
        position = contact_position(event, guide, poses)
        if reach_landmark(landmarks, time, position):
            landmarks[time] = position
    for event in impacts:
        time = event['t']
        if start < time < end:
            # A confirmed contact position supersedes a nearby pose-based estimate.
            remaining = {t: position for t, position in landmarks.items()
                         if t in (start, end) or abs(time - t) >= .11}
            position = np.asarray(event['xyz'])
            if reach_landmark(remaining, time, position):
                landmarks = {**remaining, time: position}

    gap_times = times[first:stop]
    positions = np.repeat(landmarks[start][None, :], len(gap_times), axis=0)
    ordered = sorted(landmarks)
    for a, b in zip(ordered, ordered[1:]):
        keep = (a <= gap_times) & (gap_times <= b)
        positions[keep] = arc_between(gap_times[keep], a, b, landmarks[a], landmarks[b])
    return np.clip(positions, LOW, HIGH)


def compatible_flight(segment, landmarks):
    """Reject an arc that is physically too far from a nearby reviewed impact."""
    for time, position in landmarks:
        closest = np.clip(time, segment['start'], segment['end'])
        if abs(closest - time) > .18:
            continue
        sample = fitted_sample([segment], closest)
        if sample and np.linalg.norm(np.subtract(sample['xyz'], position)) > 1.1 + MAX_SPEED * abs(closest - time):
            return False
    return True


def limit_jumps(positions, times, predicted):
    """Smooth conflicts between independent fits while favoring supported points."""
    path = positions.copy()
    limits = np.diff(times) * MAX_SPEED
    freedom = np.where(predicted, 1., .35)
    for _ in range(24):
        changed = False
        for indices in (range(len(limits)), range(len(limits) - 1, -1, -1)):
            for i in indices:
                delta = path[i + 1] - path[i]
                length = np.linalg.norm(delta)
                if length <= limits[i] + .002:
                    continue
                ratio = freedom[i] / (freedom[i] + freedom[i + 1])
                correction = delta * (1 - limits[i] / length)
                path[i] += correction * ratio
                path[i + 1] -= correction * (1 - ratio)
                changed = True
        if not changed:
            break
    moved = np.linalg.norm(path - positions, axis=1) > .08
    return path, np.asarray(predicted) | moved


def match_ball_playback(match, contacts):
    """Provide an observed or predicted position at every live source frame."""
    fits = sorted(match['ball_segments'] + match['ball_review']['additional_segments'], key=lambda s: s['start'])
    observations = {round(row['t'], 6): row['ball'] for row in match['ball_track'] if row['ball']}
    review = match['ball_review']
    bounces = [{'t': event['t'], 'xyz': [*event['xy'], BALL_RADIUS]} for event in review.get('bounces', [])]
    impacts = bounces + review.get('volleys', [])
    legacy = review['rally_events']['events']
    K, R, C = (np.asarray(match['calibration'][key]) for key in ('K', 'R', 'C'))
    legacy_projection = K @ R @ np.column_stack((np.eye(3), -C))
    fps = review['fps']

    playback = []
    for span in match['reviewed_rallies']:
        start, end = span['start'], span['end']
        times = source_times(start, end, fps)
        local_contacts = [event for event in contacts if start <= event['source_seconds'] <= end]
        poses = [frame for frame in match['frames'] if start - .35 <= frame['t'] <= end + .35]
        local_fits = [segment for segment in fits if segment['start'] < end and segment['end'] >= start]
        trusted_impacts = [(event['t'], event['xyz']) for event in impacts
                           if start <= event['t'] <= end and event.get('method') == 'grid-reviewed volley']
        if local_fits and local_contacts and local_contacts[0]['kind'] == 'serve':
            first_fit = local_fits[0]
            serve = contact_position(local_contacts[0], np.asarray(first_fit['p0']), poses)
            trusted_impacts.append((start, serve))
        local_fits = [segment for segment in local_fits if compatible_flight(segment, trusted_impacts)]
        samples = [fitted_sample(local_fits, time) for time in times]
        predicted = [sample is None for sample in samples]
        positions = np.zeros((len(times), 3))
        projection = legacy_projection if start == legacy[0]['source_seconds'] else None
        for missing, indices in groupby(range(len(times)), key=lambda i: predicted[i]):
            members = list(indices)
            first, stop = members[0], members[-1] + 1
            if missing:
                positions[first:stop] = fill_gap(times, samples, first, stop, local_contacts, impacts, poses)
            else:
                positions[first:stop] = [sample['xyz'] for sample in samples[first:stop]]
        contact_times = ([event['source_seconds'] for event in local_contacts]
                         + [event['t'] for event in impacts if start <= event['t'] <= end])
        refined = refine_path(positions, times, observations, projection, local_contacts, contact_times, predicted)
        adjusted = np.linalg.norm(refined - positions, axis=1) > .25
        positions, predicted = limit_jumps(refined, times, np.asarray(predicted) | adjusted)
        if not np.isfinite(positions).all():
            raise ValueError(f'Invalid ball playback position in rally at {start}')
        playback.append({'start': start, 'end': end,
                         'samples': [[time, *xyz.round(5).tolist(), int(is_predicted)]
                                     for time, xyz, is_predicted in zip(times, positions, predicted, strict=True)]})
    return playback
