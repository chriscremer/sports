"""Make chronological contact sheets for reviewing a rally at source resolution."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def grid(data, output, start, stop, step, crop, columns, scale, clip_start=55):
    clip_id = f'rally-{clip_start:g}'
    clip = json.loads((data / f'comparison/{clip_id}-models.json').read_text())['clip']
    baseline = json.loads((data / f'analysis/{clip_id}.json').read_text())['ball_track']
    stop = min(start + 48, clip['frame_count']) if stop is None else stop
    if step < 1:
        raise ValueError('Step must be at least one frame')
    indices = range(start, stop, step)
    if not indices or min(indices) < 0 or max(indices) >= clip['frame_count']:
        raise ValueError('Review frame range is outside the excerpt')
    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h, columns) < 1 or scale <= 0 or x + w > clip['width'] or y + h > clip['height']:
        raise ValueError('Invalid crop, scale, or column count')
    tile_w, tile_h = round(w * scale), round(h * scale)
    border, label = 3, 24
    rows = (len(indices) + columns - 1) // columns
    sheet = np.full((rows * (tile_h + label + border), columns * (tile_w + border), 3), 20, np.uint8)
    receipts = []
    frames = cv2.VideoCapture(str(data / 'source-1080p.mp4'))
    source_frame = clip['first_frame'] + start
    try:
        if not frames.set(cv2.CAP_PROP_POS_FRAMES, source_frame):
            raise ValueError(f'Could not seek to source frame {source_frame}')
        for tile, index in enumerate(indices):
            wanted = clip['first_frame'] + index
            while source_frame < wanted:
                if not frames.grab():
                    raise ValueError(f'Could not skip source frame {source_frame}')
                source_frame += 1
            ok, frame = frames.read()
            source_frame += 1
            if not ok or frame.shape[:2] != (clip['height'], clip['width']):
                raise ValueError(f'Could not read 1080p source frame {wanted}')
            time = clip['source_start'] + index / clip['fps']
            prior = min(baseline, key=lambda row: abs(row['t'] - time))
            if abs(prior['t'] - time) > .5 / clip['fps']:
                raise ValueError(f'Baseline is out of sync at frame {wanted}')
            left, top = tile % columns * (tile_w + border), tile // columns * (tile_h + label + border)
            pixels = frame[y:y+h, x:x+w]
            view = cv2.resize(pixels, (tile_w, tile_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            sheet[top+label:top+label+tile_h, left:left+tile_w] = view
            status = 'v1 marker' if prior['ball'] else 'v1 empty'
            cv2.putText(sheet, f'{index:03d}  {time:06.3f}s  {status}', (left+5, top+17),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (237, 237, 237), 1, cv2.LINE_AA)
            receipts.append({'index': index, 'source_frame': wanted, 'source_seconds': round(time, 6),
                             'image_origin': [left, top+label], 'v1': prior['ball']})
    finally:
        frames.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f'Could not write {output}')
    output.with_suffix('.json').write_text(json.dumps({
        'crop_xywh_1080p': crop, 'tile_wh': [tile_w, tile_h], 'frames': receipts,
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, help='Exclusive excerpt frame index, defaults to 48 frames past --start')
    parser.add_argument('--step', type=int, default=4)
    parser.add_argument('--crop', type=int, nargs=4, default=[200, 150, 1500, 750], metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--columns', type=int, default=3)
    parser.add_argument('--scale', type=float, default=.4)
    parser.add_argument('--clip-start', type=float, default=55, help='Nominal match time at the start of the review clip')
    args = parser.parse_args()
    grid(args.data, args.output, args.start, args.stop, args.step, args.crop, args.columns, args.scale, args.clip_start)
