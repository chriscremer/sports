"""Approximate monocular flight reconstruction with reviewed event constraints.

Height at a bounce is constrained to the ball radius. Contact heights are
explicit priors, not observations. Linear drag is fitted per flight; this is an
approximation of a perforated ball's aerodynamics, not a measured 3D trajectory.
"""
import numpy as np
from scipy.optimize import least_squares

GRAVITY=np.array([0.,0.,-9.81])

def flight(p0,p1,duration,drag,dt):
    dt=np.asarray(dt,dtype=float)[...,None]
    drag=max(float(drag),1e-5)
    factor=-np.expm1(-drag*dt)/drag
    end_factor=-np.expm1(-drag*duration)/drag
    v0=(np.asarray(p1)-p0-GRAVITY*(duration-end_factor)/drag)/end_factor
    position=np.asarray(p0)+factor*v0+GRAVITY*(dt-factor)/drag
    velocity=np.exp(-drag*dt)*v0+GRAVITY*factor
    return position,velocity

def project(points,K,R,C):
    cam=(np.asarray(points)-C)@R.T
    if np.any(cam[...,2]<=0):raise ValueError('Ball is behind the camera')
    pixels=cam@K.T
    return pixels[...,:2]/pixels[...,2,None]

def fit_flights(track,events,calibration):
    K,R,C=[np.asarray(calibration[k]) for k in ['K','R','C']]
    observations=[];initial=[]
    detections=[r for r in track if r['ball']]
    for event in events:
        row=min(detections,key=lambda r:abs(r['t']-event['t']))
        u,v=row['ball']['pixel'];ray=R.T@np.array([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],1])
        h=event['height_prior_m'];initial.append(C+(h-C[2])/ray[2]*ray)
    for a,b in zip(events,events[1:]):
        rows=[r for r in detections if a['t']<=r['t']<=b['t']]
        if len(rows)<5:raise ValueError('A reviewed flight has fewer than five model detections')
        observations.append((np.array([r['t'] for r in rows]),np.array([r['ball']['pixel'] for r in rows])))
    n=len(events);initial=np.r_[np.clip(np.array(initial),[-2,-8,.01],[8,19,3]).ravel(),np.full(n-1,.5)]
    low=np.r_[np.tile([-3,-10,0],n),np.full(n-1,.005)]
    high=np.r_[np.tile([9,20,3],n),np.full(n-1,6.)]
    def residual(parameters):
        nodes=parameters[:n*3].reshape(n,3);drags=parameters[n*3:];parts=[]
        for i,(times,pixels) in enumerate(observations):
            duration=events[i+1]['t']-events[i]['t']
            xyz,_=flight(nodes[i],nodes[i+1],duration,drags[i],times-events[i]['t'])
            parts.extend((project(xyz,K,R,C)-pixels).ravel())
        for node,event in zip(nodes,events):
            # Bounce anchors are tight; shot heights are weak, reviewed assumptions.
            weight=1500 if event['kind']=='bounce' else 100
            parts.append((node[2]-event['height_prior_m'])*weight)
            if event.get('reviewed_pixel'):
                parts.extend((project(node,K,R,C)-event['reviewed_pixel'])*20)
        return np.asarray(parts)
    fit=least_squares(residual,initial,bounds=(low,high),loss='soft_l1',f_scale=3,max_nfev=800)
    # Remove learned heatmap outliers before the final joint, continuous fit.
    nodes=fit.x[:n*3].reshape(n,3)
    for i,(times,pixels) in enumerate(observations):
        xyz,_=flight(nodes[i],nodes[i+1],events[i+1]['t']-events[i]['t'],fit.x[n*3+i],times-events[i]['t'])
        keep=np.linalg.norm(project(xyz,K,R,C)-pixels,axis=1)<15
        observations[i]=(times[keep],pixels[keep])
    fit=least_squares(residual,fit.x,bounds=(low,high),loss='soft_l1',f_scale=3,max_nfev=800)
    nodes=fit.x[:n*3].reshape(n,3);segments=[]
    for i,(times,pixels) in enumerate(observations):
        start,end=events[i]['t'],events[i+1]['t'];drag=fit.x[n*3+i]
        xyz,vel=flight(nodes[i],nodes[i+1],end-start,drag,times-start)
        errors=np.linalg.norm(project(xyz,K,R,C)-pixels,axis=1)
        dense,speeds=flight(nodes[i],nodes[i+1],end-start,drag,np.linspace(0,end-start,100))
        rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else float('inf')
        minimum=3 if end-start<.35 and events[i+1].get('reviewed_pixel') else 5
        if len(times)<minimum or rmse>12 or dense[:,2].min()<-.05 or dense[:,2].max()>5 or np.linalg.norm(speeds,axis=1).max()>45:
            continue
        segments.append({'start':start,'end':end,'p0':nodes[i].round(5).tolist(),'p1':nodes[i+1].round(5).tolist(),
            'drag_per_second':round(float(drag),6),'reprojection_rmse_px':round(rmse,2),
            'support_times':sorted(set(times.round(6).tolist()+[e['t'] for e in events[i:i+2] if e.get('reviewed_pixel')])),'observations':len(times),
            'method':'reviewed event constraints + fitted gravity and linear drag; contact heights are priors'})
    return segments

def sample_segment(segments,t,max_gap=.35):
    for s in segments:
        if s['start']<=t<=s['end']:
            support=np.asarray(s['support_times'])
            before=support[support<=t];after=support[support>=t]
            if before.size and after.size and after[0]-before[-1]>max_gap:return None
            if np.min(abs(support-t))>.12:return None
            xyz,velocity=flight(np.array(s['p0']),np.array(s['p1']),s['end']-s['start'],s['drag_per_second'],t-s['start'])
            return {'xyz':xyz.round(5).tolist(),'velocity':velocity.round(5).tolist(),'height':round(float(xyz[2]),4),
                    'kind':'estimated 3D flight','reprojection_rmse_px':s['reprojection_rmse_px']}
    return None
