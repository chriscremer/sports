"""Run learned, overlapping three-frame ball tracking on an analyzed excerpt."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from tracknet import TrackNet,preprocess,heatmap_detection,filter_track
from ball_exports import write_ball_csv

DEFAULT=Path('/Users/chriscremer/Downloads/pickleball_video_analysis')

def run(args):
    root=Path(args.data);path=root/'analysis'/f'rally-{args.start:g}.json'
    data=json.loads(path.read_text()); clip=data['clip']
    cap=cv2.VideoCapture(str(root/'source-720p.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC,clip['start']*1000)
    images=[];times=[]
    while True:
        ok,img=cap.read()
        if not ok:break
        t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if t>=clip['end']:break
        times.append(t);images.append(cv2.resize(img,(512,288),interpolation=cv2.INTER_CUBIC))
    cap.release()
    if len(images)<3:raise ValueError('Excerpt needs at least three frames')
    model=TrackNet(root/'models').eval().to(args.device)
    maps=np.zeros((len(images),288,512),dtype=np.float32);counts=np.zeros(len(images))
    began=time.monotonic()
    with torch.inference_mode():
        for i in range(len(images)-2):
            x=torch.from_numpy(preprocess(images[i:i+3])[None]).to(args.device)
            maps[i:i+3]+=model(x).cpu().numpy()[0];counts[i:i+3]+=1
            if i%100==0: print(f'{i}/{len(images)} temporal windows · {time.monotonic()-began:.1f}s',flush=True)
    maps/=counts[:,None,None]
    # Ensembles align each output with its actual source frame, not the centre frame.
    track=[dict(t=round(t,6),ball=heatmap_detection(h,clip['width'],clip['height'],args.threshold))
           for t,h in zip(times,maps)]
    for row in track:
        if row['ball']:
            x,y=row['ball']['pixel']
            if not (.15*clip['width']<x<.87*clip['width'] and .1*clip['height']<y<.9*clip['height']):row['ball']=None
    rejected=[]
    for i,row in enumerate(track):
        if not row['ball']:continue
        # A frame needs a nearby supporting detection; preserve velocity reversals.
        neighbors=[r for r in track[max(0,i-2):i]+track[i+1:i+3] if r['ball']]
        supported=any(np.linalg.norm(np.subtract(row['ball']['pixel'],r['ball']['pixel']))
                      <=clip['width']*.1*abs(row['t']-r['t'])*fps+8 for r in neighbors)
        if not supported:rejected.append(i)
    for i in rejected:track[i]['ball']=None
    filtered=filter_track(track,clip['width'])
    data['ball_track']=track
    # A new inference pass invalidates fits based on the old observations.
    data.pop('ball_segments',None);data.pop('ball_events',None)
    stamps=np.array(times)
    for frame in data['frames']:
        i=int(np.argmin(abs(stamps-frame['t'])))
        frame['ball']=track[i]['ball'] if abs(stamps[i]-frame['t'])<1/fps else None
        frame.pop('ball3d',None)
    detected=sum(r['ball'] is not None for r in track)
    data['ball_summary']={'model':'TrackNet-Pickleball','checkpoint_url':'https://github.com/AndrewDettor/TrackNet-Pickleball',
        'source_hz':fps,'threshold':args.threshold,'detected_frames':detected,'total_frames':len(track),
        'temporal_method':'mean of aligned predictions from up to three overlapping RGB triplets; isolated jumps rejected',
        'score_note':'heatmap peak, not a calibrated probability','reconstruction':'height not yet estimated'}
    data['ball_summary']['temporal_rejections']=len(rejected)+filtered
    data['method']['ball']='pretrained TrackNet-Pickleball, temporal heatmap ensemble; unknown on uncertain frames'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,separators=(',',':')));temporary.replace(path)
    write_ball_csv(data,path.parent)
    if args.heatmaps:np.savez_compressed(root/'analysis'/f'ball-heatmaps-{args.start:g}.npz',maps=maps,times=stamps)
    print(f'Saved {detected}/{len(track)} learned detections; {len(rejected)+filtered} temporal peaks rejected',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(DEFAULT));p.add_argument('--start',type=float,default=55)
    p.add_argument('--device',default='mps');p.add_argument('--threshold',type=float,default=.5);p.add_argument('--heatmaps',action='store_true')
    run(p.parse_args())
