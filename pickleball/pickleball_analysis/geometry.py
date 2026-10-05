"""Court calibration and an explicit upright-plane approximation of 3D pose."""
import cv2
import numpy as np
from scipy.optimize import least_squares

WIDTH, LENGTH = 6.096, 13.4112

def calibrate(width, height, corners):
    # Coordinates: x across court, y from far baseline, z upwards; metres.
    world = np.array([[0,0,0],[WIDTH,0,0],[WIDTH,LENGTH,0],[0,LENGTH,0]], dtype=float)
    pixels = np.array(corners, dtype=float)
    H = cv2.getPerspectiveTransform(pixels.astype('float32'), world[:,:2].astype('float32'))
    # Focal length is an assumption: a single planar view does not fully calibrate it.
    K = np.array([[width*1.1,0,width/2],[0,width*1.1,height/2],[0,0,1]], dtype=float)
    ok, r, t = cv2.solvePnP(world,pixels,K,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise ValueError('Court calibration failed')
    def residual(p):
        return (cv2.projectPoints(world,p[:3],p[3:],K,np.zeros(5))[0].reshape(-1,2)-pixels).ravel()
    fit=least_squares(residual,np.r_[r.ravel(),t.ravel()])
    R=cv2.Rodrigues(fit.x[:3])[0]; t=fit.x[3:]; C=-R.T@t
    # x-right / y-towards-near-baseline is a left-handed chart when height is up.
    # solvePnP uses a right-handed chart, so flip its vertical axis explicitly.
    # R is then a world-to-camera basis transform, not a proper rotation matrix.
    if C[2]<0:
        R[:,2]*=-1
        C[2]*=-1
    return H,K,R,C,float(np.sqrt(np.mean(fit.fun**2)))

def ground(H, pixel):
    p=H@np.r_[pixel,1.0]
    return p[:2]/p[2]

def ray(K,R,C,pixel):
    return R.T@np.linalg.inv(K)@np.r_[pixel,1.0]

def lift_pose(K,R,C,foot,joints):
    # Every joint lies in an upright plane through the feet facing the camera.
    # This preserves visible posture, but does not infer hidden limb depth.
    anchor=np.r_[foot,0.0]; normal=C-anchor; normal[2]=0; normal/=np.linalg.norm(normal)
    out=[]
    for x,y,confidence in joints:
        if confidence<0.2: out.append(None); continue
        direction=ray(K,R,C,[x,y]); denom=normal@direction
        if abs(denom)<1e-8: out.append(None); continue
        point=C+direction*((normal@(anchor-C))/denom)
        point[2]=np.clip(point[2],0,2.5)
        out.append([round(float(v),4) for v in point])
    return out
