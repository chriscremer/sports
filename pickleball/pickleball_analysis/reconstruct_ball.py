"""Fit ball flights from TrackNet observations and explicit event annotations."""
import argparse,json
from pathlib import Path
import numpy as np
from ball_geometry import fit_flights,sample_segment,project
from ball_exports import write_ball_csv

def run(args):
    path=args.data/'analysis'/f'rally-{args.start:g}.json';data=json.loads(path.read_text())
    annotations=args.data/'annotations'/f'ball-events-{args.start:g}.json'
    review=json.loads(annotations.read_text())
    for row in data['ball_track']:
        if row.get('rejected_detection'):row['ball']=row.pop('rejected_detection')
    segments=fit_flights(data['ball_track'],review['events'],data['calibration'])
    data['ball_segments']=segments;data['ball_events']=review
    K,R,C=[np.asarray(data['calibration'][k]) for k in ['K','R','C']]
    rejected=0
    for row in data['ball_track']:
        row['reconstruction']=sample_segment(segments,row['t'])
        if row['reconstruction']:
            pixel=project(row['reconstruction']['xyz'],K,R,C)
            row['reconstruction']['pixel']=pixel.round(2).tolist()
            if row['ball'] and np.linalg.norm(pixel-row['ball']['pixel'])>22:
                row['rejected_detection']=row['ball'];row['ball']=None;rejected+=1
    stamps=np.array([row['t'] for row in data['ball_track']])
    for f in data['frames']:
        f['ball3d']=sample_segment(segments,f['t'])
        f['ball']=data['ball_track'][int(np.argmin(abs(stamps-f['t'])))]['ball']
    data['ball_summary']['reconstruction']='estimated gravity/drag flights, with manually reviewed bounces and assumed contact heights'
    data['ball_summary']['reconstructed_frames']=sum(r['reconstruction'] is not None for r in data['ball_track'])
    data['ball_summary']['flight_outliers_rejected']=rejected
    data['ball_summary']['detected_frames']=sum(r['ball'] is not None for r in data['ball_track'])
    data['method']['ball3d']='approximate flight fit; bounce heights constrained, contact heights are priors; no extrapolation outside reviewed flights'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,separators=(',',':')));temporary.replace(path)
    write_ball_csv(data,path.parent)
    for s in segments:print(f"{s['start']:.3f}–{s['end']:.3f}: {s['observations']} observations, {s['reprojection_rmse_px']:.2f}px RMS, {s['p0']} → {s['p1']}")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis'))
    p.add_argument('--start',type=float,default=55);run(p.parse_args())
