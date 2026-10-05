"""Bounded, explicitly labelled interpolation between observations of the same player."""
import numpy as np
from geometry import lift_pose

def bridge_short_gaps(frames,K,R,C,max_gap=.4):
    tracks={}
    for index,frame in enumerate(frames):
        for player in frame['players']:
            player['tracking_status']='detected'
            tracks.setdefault(player['id'],[]).append((index,player))
    filled=0
    for observations in tracks.values():
        for (a,left),(b,right) in zip(observations,observations[1:]):
            duration=frames[b]['t']-frames[a]['t']
            if b-a<=1 or duration>max_gap+1e-6 or left['team']!=right['team']:continue
            # Avoid bridging jumps / identity errors, and never extrapolate endpoints.
            if np.linalg.norm(np.array(right['xy'])-left['xy'])>6*duration+.25:continue
            for index in range(a+1,b):
                if any(p['id']==left['id'] for p in frames[index]['players']):continue
                if sum(p['team']==left['team'] for p in frames[index]['players'])>=2:continue
                alpha=(frames[index]['t']-frames[a]['t'])/duration
                xy=(1-alpha)*np.array(left['xy'])+alpha*np.array(right['xy'])
                kl,kr=np.array(left['keypoints']),np.array(right['keypoints'])
                joints=(1-alpha)*kl+alpha*kr
                # Only show joints reliable at both ends of the gap.
                joints[:,2]=np.minimum(kl[:,2],kr[:,2])
                player={'id':left['id'],'team':left['team'],'xy':np.round(xy,4).tolist(),
                    'confidence':None,'keypoints':np.round(joints,3).tolist(),
                    'joints3d':lift_pose(K,R,C,xy,joints),'tracking_status':'interpolated',
                    'support_times':[frames[a]['t'],frames[b]['t']]}
                frames[index]['players'].append(player);filled+=1
    return filled
