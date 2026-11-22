"""Stabilize reconstructed player tracks on the match clock."""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def weighted_positions(times, points, deviation):
    delta = times[:, None] - times[None, :]
    weights = np.exp(-.5 * (delta / deviation)**2) * (abs(delta) <= 2.5 * deviation)
    return weights @ points / weights.sum(axis=1)[:, None]


def linear_joints(left, right, weight):
    return [((1 - weight) * np.array(a) + weight * np.array(b)).round(4).tolist()
            if a is not None and b is not None else None
            for a, b in zip(left, right, strict=True)]


def stabilize_track(frames, observations):
    times = np.array([frames[i]['t'] for i, _ in observations])
    roots = np.array([p['xy'] for _, p in observations])
    joints = np.array([[joint if joint is not None else [np.nan] * 3 for joint in p['joints3d']]
                       for _, p in observations])
    relatives = joints - np.column_stack((roots, np.zeros(len(roots))))[:, None, :]

    repaired = roots.copy()
    for i in range(1, len(roots) - 1):
        span = times[i + 1] - times[i - 1]
        if span > .24:
            continue
        weight = (times[i] - times[i - 1]) / span
        linear = (1 - weight) * roots[i - 1] + weight * roots[i + 1]
        if np.linalg.norm(roots[i] - linear) > .36 + 1.4 * span:
            repaired[i] = linear
    smoothed = weighted_positions(times, repaired, .075)

    # Smooth limbs relative to the body so camera jitter does not look like a
    # change in stance. Wrists use a shorter window to keep fast paddle motion.
    for j in range(joints.shape[1]):
        valid = np.flatnonzero(np.isfinite(relatives[:, j, 0]))
        if len(valid) < 2:
            continue
        deviations = .045 if j in (7, 8, 9, 10, 15, 16) else .07
        offsets = relatives[valid, j].copy()
        for k in range(1, len(valid) - 1):
            a, b, c = valid[k - 1:k + 2]
            if times[c] - times[a] > .24:
                continue
            midpoint = np.interp(times[b], times[[a, c]], [0, 1])
            estimate = offsets[k - 1] * (1 - midpoint) + offsets[k + 1] * midpoint
            if np.linalg.norm(offsets[k] - estimate) > .47:
                offsets[k] = estimate
        relatives[valid, j] = weighted_positions(times[valid], offsets, deviations)
    joints = relatives + np.column_stack((smoothed, np.zeros(len(smoothed))))[:, None, :]
    joints[:, :, 2] = np.clip(joints[:, :, 2], 0, 2.5)
    for (_, player), root, points in zip(observations, smoothed, joints, strict=True):
        player['xy'] = root.round(4).tolist()
        player['joints3d'] = [point.round(4).tolist() if np.isfinite(point).all() else None
                              for point in points]


def bridge_track(frames, observations):
    for (first, left), (last, right) in zip(observations, observations[1:]):
        t0, t1 = frames[first]['t'], frames[last]['t']
        if last - first <= 1 or t1 - t0 > .32 or np.linalg.norm(np.subtract(left['xy'], right['xy'])) > .25 + 5.5 * (t1 - t0):
            continue
        for frame in frames[first + 1:last]:
            if sum(p['team'] == left['team'] for p in frame['players']) >= 2:
                continue
            fraction = (frame['t'] - t0) / (t1 - t0)
            xy = (1 - fraction) * np.array(left['xy']) + fraction * np.array(right['xy'])
            frame['players'].append({'id': left['id'], 'team': left['team'], 'xy': xy.round(4).tolist(),
                                     'joints3d': linear_joints(left['joints3d'], right['joints3d'], fraction),
                                     'confidence': None, 'tracking_status': 'interpolated'})


def link_camera_cut(previous, current):
    if not previous or not current or current[0]['t'] - previous[-1]['t'] > .20:
        return
    replacements = {}
    for team in ('near', 'far'):
        before = [p for p in previous[-1]['players'] if p['team'] == team]
        after = [p for p in current[0]['players'] if p['team'] == team]
        if not before or not after:
            continue
        distances = np.array([[np.linalg.norm(np.subtract(a['xy'], b['xy'])) for b in after] for a in before])
        for i, j in zip(*linear_sum_assignment(distances), strict=True):
            if distances[i, j] <= .8:
                replacements[after[j]['id']] = before[i]['id']
    for frame in current:
        for player in frame['players']:
            player['id'] = replacements.get(player['id'], player['id'])


def refine_match_players(frames, windows):
    previous = []
    for window in windows:
        shot = [frame for frame in frames if window['start'] <= frame['t'] < window['end']]
        tracks = defaultdict(list)
        for index, frame in enumerate(shot):
            for player in frame['players']:
                tracks[player['id']].append((index, player))
        for observations in tracks.values():
            if len(observations) >= 3:
                stabilize_track(shot, observations)
            bridge_track(shot, observations)
        link_camera_cut(previous, shot)
        previous = shot
