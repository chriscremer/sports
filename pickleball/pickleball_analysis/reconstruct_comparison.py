"""Independent 3D flight fits for the three selected comparison setups.

Uses existing reviewed events/calibration/poses as shared controls. Never writes
to the original analysis files or changes raw model predictions.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np

from ball_geometry import fit_flights, sample_segment, project
from compare_ball_models import atomic_json

SELECTED = ['pickleball-512', 'v3-512', 'v3-960']


def detection_display_rows(model, clip, analysis, segments):
    """Keep 2D-only candidates visible without turning assumed ground into height."""
    calibration=analysis['calibration']
    H=np.asarray(calibration['H'])
    K,R,C=[np.asarray(calibration[key]) for key in ['K','R','C']]
    scale=np.array([analysis['clip']['width']/clip['width'],analysis['clip']['height']/clip['height']])
    rows=[]
    for row in model['frames']:
        candidate=row['candidates'][0] if row['candidates'] else None
        display=dict(t=row['t'],source_t=row['source_t'],projection=None)
        if candidate:
            pixel=np.asarray(candidate['pixel'])*scale
            projected=H@np.r_[pixel,1.]
            if abs(projected[2])>1e-8:
                xy=projected[:2]/projected[2]
                if np.isfinite(xy).all():
                    fit=sample_segment(segments,row['source_t'])
                    error=float(np.linalg.norm(project(fit['xyz'],K,R,C)-pixel)/scale[0]) if fit else None
                    display['projection']=dict(ground_xy=xy.round(5).tolist(),height=None,pixel=candidate['pixel'],score=candidate['score'],fit_residual_1080p_px=round(error,2) if error is not None else None,show_with_fit=error is not None and error>18,method='2D candidate mapped by court homography under a ground-plane assumption; ball height and true horizontal position are unknown')
        rows.append(display)
    return rows


def supported_event_groups(track, events):
    """Fit adjacent supported flights jointly; leave unsupported flights empty."""
    groups, rejected, current = [], [], []
    for a, b in zip(events, events[1:]):
        count = sum(row['ball'] is not None and a['t'] <= row['t'] <= b['t'] for row in track)
        # Match the existing fitter's entry requirement, even for short flights.
        if count >= 5:
            if not current: current = [a]
            current.append(b)
        else:
            if current: groups.append(current)
            current = []
            rejected.append(dict(start=a['t'],end=b['t'],observations=count,reason='Fewer than five model detections'))
    if current: groups.append(current)
    return groups, rejected


def reconstruct_model(model, clip, analysis, review):
    sx = analysis['clip']['width']/clip['width']
    sy = analysis['clip']['height']/clip['height']
    # Preserve the existing fitter's pixel thresholds by working in the original
    # calibration's 1280x720 image coordinates. World output remains metres.
    track = [dict(t=row['source_t'],ball=dict(pixel=[row['candidates'][0]['pixel'][0]*sx,row['candidates'][0]['pixel'][1]*sy]) if row['candidates'] else None) for row in model['frames']]
    segments, skipped = [], []
    if review:
        groups, skipped = supported_event_groups(track, review['events'])
        for events in groups:
            fitted = fit_flights(track, events, analysis['calibration'])
            segments.extend(fitted)
            accepted = {(s['start'],s['end']) for s in fitted}
            skipped.extend(dict(start=a['t'],end=b['t'],reason='Fit rejected by reprojection, support, height or speed checks') for a,b in zip(events,events[1:]) if (a['t'],b['t']) not in accepted)
    supported = sum(sample_segment(segments,row['t']) is not None for row in track)
    count = sum(s['observations'] for s in segments)
    weighted_rmse = (sum(s['reprojection_rmse_px']**2*s['observations'] for s in segments)/count)**.5 if count else None
    return dict(segments=segments,skipped_flights=skipped,reconstructed_frames=supported,total_frames=len(track),fit_rmse_calibration_px=round(weighted_rmse,2) if weighted_rmse is not None else None,fit_rmse_1080p_px=round(weighted_rmse/sx,2) if weighted_rmse is not None else None,calibration_image_size=[analysis['clip']['width'],analysis['clip']['height']],fit_observations=count,status='estimated flights' if segments else 'No reviewed supported 3D ball flight in this excerpt')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'))
    args=parser.parse_args()
    root=args.data
    manifest=json.loads((root/'comparison/manifest.json').read_text())
    output=dict(selected=SELECTED,selection_reason='Three strongest candidate-coverage setups across both excerpts. Coverage is not measured accuracy; native 1080p variants and WASB remain available as 2D references.',method='Independent model detections + identical existing calibration and reviewed event/height constraints; gravity and linear drag. Common existing player poses. Hollow ground-projection markers preserve 2D candidates with unknown height; projections are display aids, not reconstructed ball positions.',clips=[])
    for clip in manifest['clips']:
        original=json.loads((root/'analysis'/f"{clip['id']}.json").read_text())
        predictions=json.loads((root/clip['predictions']).read_text())
        annotations=root/'annotations'/f"ball-events-{clip['id'].split('-')[-1]}.json"
        review=json.loads(annotations.read_text()) if annotations.exists() else None
        # Only shared player poses are copied; never reuse the original ball fit.
        poses=[dict(t=round(f['t']-clip['source_start'],6),players=copy.deepcopy(f['players'])) for f in original['frames']]
        data=dict(clip=clip,poses=poses,events=review,models={})
        for model_id in SELECTED:
            data['models'][model_id]=reconstruct_model(predictions['models'][model_id],clip,original,review)
            data['models'][model_id]['detection_display']=detection_display_rows(predictions['models'][model_id],clip,original,data['models'][model_id]['segments'])
            print(clip['id'],model_id,len(data['models'][model_id]['segments']),'flights;',data['models'][model_id]['reconstructed_frames'],'supported frames')
        path=root/'comparison'/f"{clip['id']}-3d.json"
        atomic_json(path,data)
        output['clips'].append(dict(id=clip['id'],file=f'comparison/{path.name}'))
    atomic_json(root/'comparison/reconstruction-manifest.json',output)


if __name__=='__main__': main()
