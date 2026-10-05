"""Extract real YOLO poses and consistent local IDs. Ball inference is separate."""
import argparse,json,os,subprocess,time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from geometry import calibrate,ground,lift_pose
from tracking import bridge_short_gaps

DEFAULT=Path('/Users/chriscremer/Downloads/pickleball_video_analysis')

def overlap(a,b):
    low=np.maximum(a[:2],b[:2]);high=np.minimum(a[2:4],b[2:4]);intersection=np.prod(np.maximum(0,high-low))
    return intersection/(np.prod(a[2:4]-a[:2])+np.prod(b[2:4]-b[:2])-intersection+1e-6)

def run(args):
    data=Path(args.data); (data/'analysis').mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(data/'source-720p.mp4'))
    if not cap.isOpened(): raise RuntimeError('Download source-720p.mp4 first')
    width,height=int(cap.get(3)),int(cap.get(4)); fps=cap.get(5)
    # Manually verified court corners at 00:57 of this broadcast, scaled from 640x360.
    corners=np.array([[237,105],[409,105],[545,268],[103,268]],float)*[width/640,height/360]
    H,K,R,C,error=calibrate(width,height,corners)
    model=YOLO(str(data/'models/yolo11s-pose.pt'))
    frames=[]; active={}; next_id=1
    total=int(args.duration*args.hz); started=time.monotonic()
    for index in range(total):
        stamp=args.start+index/args.hz
        cap.set(cv2.CAP_PROP_POS_MSEC,stamp*1000); ok,img=cap.read()
        if not ok: break
        result=model.predict(img,imgsz=960,conf=.25,device=args.device,verbose=False)[0]
        detections=[]
        batches=[(result,0,0)]
        # Two overlapping tiles enlarge distant players without cutting off their feet.
        # Include run-off behind the baseline, not just the painted court.
        for bounds in [[.219,.09,.570,.472],[.484,.09,.836,.472]]:
            x0,y0,x1,y1=[round(v*s) for v,s in zip(bounds,[width,height,width,height])]
            far_result=model.predict(img[y0:y1,x0:x1],imgsz=960,conf=.25,device=args.device,verbose=False)[0]
            batches.append((far_result,x0,y0))
        for result,offset_x,offset_y in batches:
            if result.keypoints is None: continue
            for box,joints in zip(result.boxes.data.cpu().numpy(),result.keypoints.data.cpu().numpy()):
                joints[:,0]+=offset_x; joints[:,1]+=offset_y
                box[[0,2]]+=offset_x;box[[1,3]]+=offset_y
                # Prefer an existing full-frame detection over a second crop of it.
                # Crop ankle estimates are unreliable when feet meet the crop edge.
                if offset_x and any(overlap(box,d['box'])>.45 for d in detections):continue
                feet=[p[:2] for p in joints[15:17] if p[2]>.2]
                pixel=np.mean(feet,axis=0) if feet else np.array([(box[0]+box[2])/2,box[3]])
                position=ground(H,pixel)
                # Reject officials; include players just beyond a baseline.
                if not (-2<position[0]<8.3 and -6<position[1]<19.5): continue
                if not (width*.18<pixel[0]<width*.86): continue
                side='near' if position[1]>6.7056 else 'far'
                if offset_x and side!='far': continue
                candidate=dict(side=side,position=position,joints=joints,box=box)
                duplicate=next((i for i,d in enumerate(detections) if d['side']==side and np.linalg.norm(d['position']-position)<.6),None)
                if duplicate is None:detections.append(candidate)
                elif box[4]>detections[duplicate]['box'][4]:detections[duplicate]=candidate
        # At most two players per side; identity matching uses side and court continuity.
        detections=[d for side in ['far','near'] for d in sorted([d for d in detections if d['side']==side],key=lambda d:-d['box'][4])[:2]]
        old=[(pid,d) for pid,d in active.items() if stamp-d['time']<1.0]
        assigned={}
        if old and detections:
            costs=np.array([[np.linalg.norm(o['pixel_center']-(d['box'][:2]+d['box'][2:4])/2) if o['side']==d['side'] else 1e4 for d in detections] for _,o in old])
            a,b=linear_sum_assignment(costs)
            assigned={int(j):old[i][0] for i,j in zip(a,b) if costs[i,j]<width*.12}
        players=[]
        for j,d in enumerate(detections):
            pid=assigned.get(j)
            if pid is None: pid=next_id; next_id+=1
            active[pid]={'time':stamp,'position':d['position'],'side':d['side'],'pixel_center':(d['box'][:2]+d['box'][2:4])/2}
            players.append({'id':pid,'team':d['side'],'xy':np.round(d['position'],4).tolist(),
                'confidence':round(float(d['box'][4]),3),'keypoints':np.round(d['joints'],3).tolist(),
                'joints3d':lift_pose(K,R,C,d['position'],d['joints'])})
        frames.append({'t':round(stamp,4),'players':players,'ball':None})
        if index%30==0: print(f'{index}/{total} frames, {len(players)} players, {time.monotonic()-started:.1f}s',flush=True)
    filled=bridge_short_gaps(frames,K,R,C)
    observed=sum(sum(p['tracking_status']=='detected' for p in f['players']) for f in frames)
    name=f'rally-{args.start:g}'
    payload={'source':{'id':'tdGSHF5FseE','url':'https://www.youtube.com/watch?v=tdGSHF5FseE','title':'Johns / Tardio vs Alshon / Daescu · PPA Las Vegas Open'},
      'clip':{'start':args.start,'end':args.start+len(frames)/args.hz,'hz':args.hz,'width':width,'height':height,'video':f'/media/previews/{name}.mp4'},
      'calibration':{'corners':corners.tolist(),'H':H.tolist(),'K':K.tolist(),'R':R.tolist(),'C':C.tolist(),'reprojection_rmse_px':error},
      'method':{'pose':'YOLO11s-pose with overlapping far-court tiles','tracking':'image-box continuity; same-ID gaps <=0.4s interpolated without endpoint extrapolation','ground':'four manually calibrated court corners','depth':'upright plane approximation; hidden limb depth unobserved','ball':'run track_ball.py for learned ball inference','value':'not trained'},
      'tracking_summary':{'detected_player_samples':observed,'interpolated_player_samples':filled,'four_player_frames':sum(len(f['players'])==4 for f in frames)},'frames':frames}
    path=data/'analysis'/f'{name}.json'
    # Recomputing players need not discard independently inferred ball data.
    if path.exists():
        previous=json.loads(path.read_text())
        same_view=previous.get('calibration')==payload['calibration']
        same_interval=all(previous.get('clip',{}).get(k)==payload['clip'][k] for k in ['start','end','width','height'])
        if same_view and same_interval and previous.get('ball_track'):
            from ball_geometry import sample_segment
            for key in ['ball_track','ball_summary','ball_segments','ball_events']:
                if key in previous:payload[key]=previous[key]
            for key in ['ball','ball3d']:
                if key in previous['method']:payload['method'][key]=previous['method'][key]
            stamps=np.array([row['t'] for row in payload['ball_track']])
            for frame in frames:
                nearest=int(np.argmin(abs(stamps-frame['t'])))
                frame['ball']=payload['ball_track'][nearest]['ball'] if abs(stamps[nearest]-frame['t'])<.04 else None
                frame['ball3d']=sample_segment(payload.get('ball_segments',[]),frame['t'])
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload,separators=(',',':')));temporary.replace(path)
    print(f'Saved {path}: {len(frames)} frames; {observed} detected, {filled} interpolated player samples',flush=True)
    if args.clip:
        cap.set(cv2.CAP_PROP_POS_MSEC,args.start*1000)
        raw=data/'previews'/f'{name}-raw.mp4';raw.parent.mkdir(exist_ok=True)
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        for _ in range(round(args.duration*fps)):
            ok,img=cap.read()
            if not ok: break
            writer.write(img)
        writer.release()
        command=[args.ffmpeg,'-y','-hide_banner','-loglevel','error','-i',str(raw),'-ss',str(args.start),'-i',str(data/'source.mp4'),'-map','0:v:0','-map','1:a:0','-t',str(args.duration),'-c:v','libx264','-preset','fast','-crf','19','-c:a','aac','-movflags','+faststart',str(data/'previews'/f'{name}.mp4')]
        subprocess.run(command,check=True);raw.unlink()
    cap.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(DEFAULT));p.add_argument('--start',type=float,default=55);p.add_argument('--duration',type=float,default=11);p.add_argument('--hz',type=float,default=15);p.add_argument('--device',default='mps');p.add_argument('--clip',action='store_true');p.add_argument('--ffmpeg',default='/Users/chriscremer/miniconda3/bin/ffmpeg');run(p.parse_args())
