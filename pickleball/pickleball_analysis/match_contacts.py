"""Place match-view ripples at marked contacts and abrupt unclassified turns."""
import numpy as np
from scipy.signal import find_peaks


def match_contact_ripples(playback, contacts, review, camera_windows):
    events = [{'t': event['t'], 'xyz': [*event['xy'], .017], 'kind': 'bounce'}
              for event in review.get('bounces', [])]
    paddle = [{'t': event['source_seconds'], 'kind': event['kind'],
               'approximate': event['frame_interval'][1] - event['frame_interval'][0] > 1}
              for event in contacts]
    for volley in review.get('volleys', []):
        closest = min(paddle, key=lambda event: abs(event['t'] - volley['t']), default=None)
        if closest and closest['kind'] == 'hit' and abs(closest['t'] - volley['t']) <= .14:
            closest['kind'] = 'volley'
        else:
            paddle.append({'t': volley['t'], 'kind': 'volley'})
    events.extend(paddle)

    # A strong horizontal turn can expose a missing contact or a bad fit.
    # Keep it separate from identified impacts, and ignore the apex of a lob.
    turns = []
    cuts = [window[key] for window in camera_windows for key in ('start', 'end')]
    for rally in playback:
        samples = np.asarray(rally['samples'])
        time = samples[:, 0]
        horizontal = samples[:, 1:3]
        stride = 3
        middle = slice(stride, -stride)
        if len(samples) <= 2 * stride + 2:
            continue
        previous = time[middle] - time[:-2 * stride]
        following = time[2 * stride:] - time[middle]
        incoming = (horizontal[middle] - horizontal[:-2 * stride]) / previous[:, None]
        outgoing = (horizontal[2 * stride:] - horizontal[middle]) / following[:, None]
        speed_in = np.linalg.norm(incoming, axis=1)
        speed_out = np.linalg.norm(outgoing, axis=1)
        heading = np.sum(incoming * outgoing, axis=1) / np.maximum(speed_in * speed_out, 1e-9)
        reversal = np.minimum(speed_in, speed_out) * np.maximum(0, -heading)
        for peak in find_peaks(reversal, height=1.65, distance=7)[0]:
            index = peak + stride
            t = float(time[index])
            if any(abs(t - cut) < .14 for cut in cuts) or any(abs(t - event['t']) < .18 for event in events):
                continue
            turns.append({'t': t, 'kind': 'turn'})
    return sorted(events + turns, key=lambda event: event['t'])
