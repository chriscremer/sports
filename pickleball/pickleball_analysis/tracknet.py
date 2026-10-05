"""Inference for the public TrackNet-Pickleball checkpoint, using PyTorch/MPS.

The original checkpoint is a Keras NCHW graph. In particular, its batch norms
normalize width (axis 3), NOT channels. Preserve that learned behavior exactly.
No weights are retrained and no colour thresholds are used.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class TrackNet(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        root = Path(model_dir)
        graph = json.loads((root/'tracknet-config.json').read_text())['config']
        self.layers = graph['layers']
        self.output_name = graph['output_layers'][0][0]
        self.ops = nn.ModuleDict()
        with np.load(root/'tracknet-weights.npz') as weights:
            for layer in self.layers:
                name, kind, cfg = layer['name'], layer['class_name'], layer['config']
                if kind == 'Conv2D':
                    kernel = weights[f'{name}__0'].transpose(3,2,0,1).copy()
                    conv = nn.Conv2d(kernel.shape[1],kernel.shape[0],tuple(cfg['kernel_size']),
                                     stride=tuple(cfg['strides']),padding=cfg['padding'])
                    conv.weight.data.copy_(torch.from_numpy(kernel))
                    conv.bias.data.copy_(torch.from_numpy(weights[f'{name}__1']))
                    self.ops[name] = conv
                elif kind == 'BatchNormalization':
                    axis = cfg['axis'][0] if isinstance(cfg['axis'],list) else cfg['axis']
                    shape = [1]*4; shape[axis] = -1
                    gamma,beta,mean,var = [weights[f'{name}__{i}'] for i in range(4)]
                    scale = gamma/np.sqrt(var+cfg['epsilon'])
                    self.ops[name] = Affine(torch.from_numpy(scale.reshape(shape).copy()),
                                            torch.from_numpy((beta-mean*scale).reshape(shape).copy()))
                elif kind not in {'InputLayer','Activation','MaxPooling2D','UpSampling2D','Concatenate'}:
                    raise ValueError(f'Unsupported checkpoint layer: {kind}')
        # Release intermediates once their last graph consumer has run.
        self.uses = {}
        for layer in self.layers:
            for node in layer.get('inbound_nodes', [[]])[0] if layer.get('inbound_nodes') else []:
                self.uses[node[0]] = self.uses.get(node[0],0)+1

    def forward(self, x):
        values = {}; remaining = self.uses.copy()
        for layer in self.layers:
            name,kind,cfg = layer['name'],layer['class_name'],layer['config']
            sources = [node[0] for node in layer['inbound_nodes'][0]] if layer['inbound_nodes'] else []
            ins = [values[n] for n in sources]
            if kind == 'InputLayer': out = x
            elif kind in {'Conv2D','BatchNormalization'}: out = self.ops[name](ins[0])
            elif kind == 'Activation':
                if cfg['activation'] == 'relu': out = F.relu(ins[0])
                elif cfg['activation'] == 'sigmoid': out = torch.sigmoid(ins[0])
                else: raise ValueError(cfg['activation'])
            elif kind == 'MaxPooling2D': out = F.max_pool2d(ins[0],tuple(cfg['pool_size']),tuple(cfg['strides']))
            elif kind == 'UpSampling2D': out = F.interpolate(ins[0],scale_factor=tuple(cfg['size']),mode='nearest')
            elif kind == 'Concatenate': out = torch.cat(ins,dim=cfg['axis'])
            values[name] = out
            for source in sources:
                remaining[source] -= 1
                if remaining[source] == 0: del values[source]
        return values[self.output_name]

class Affine(nn.Module):
    def __init__(self,scale,bias):
        super().__init__(); self.register_buffer('scale',scale); self.register_buffer('bias',bias)
    def forward(self,x): return x*self.scale+self.bias

def preprocess(images):
    """Chronological RGB triplet, 9 × 288 × 512, matching upstream inference."""
    if len(images)!=3: raise ValueError('TrackNet needs exactly three consecutive frames')
    return np.concatenate([cv2.resize(im[:,:,::-1],(512,288),interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
                           for im in images],axis=0).astype(np.float32)/255

def heatmap_detection(heatmap,width,height,threshold=.5):
    """Return largest predicted component, or explicit unknown; score is not calibrated."""
    mask=(heatmap>=threshold).astype('uint8')
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count<=1: return None
    label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    ys,xs=np.nonzero(labels==label); confidence=heatmap[ys,xs]
    x=float(np.average(xs,weights=confidence)+.5)*width/heatmap.shape[1]
    y=float(np.average(ys,weights=confidence)+.5)*height/heatmap.shape[0]
    return {'pixel':[round(x,2),round(y,2)],'model_score':round(float(confidence.max()),4),
            'kind':'TrackNet-Pickleball','tracking_status':'detected','height':None}

def filter_track(track,width):
    """Conservative offline spike / stationary-object rejection, without colour."""
    rejected=set();observed=[i for i,row in enumerate(track) if row['ball']]
    for at,i in enumerate(observed[1:-1],1):
        prev,nxt=track[observed[at-1]],track[observed[at+1]];row=track[i]
        span=nxt['t']-prev['t']
        if span>.14:continue
        fraction=(row['t']-prev['t'])/span
        predicted=np.array(prev['ball']['pixel'])*(1-fraction)+np.array(nxt['ball']['pixel'])*fraction
        neighbors_consistent=np.linalg.norm(np.subtract(prev['ball']['pixel'],nxt['ball']['pixel']))<width*.06*span*30
        if neighbors_consistent and np.linalg.norm(predicted-row['ball']['pixel'])>width*.025:rejected.add(i)
    for i in observed:
        window=[j for j in observed if track[i]['t']<=track[j]['t']<=track[i]['t']+.65]
        if len(window)<8 or track[window[-1]]['t']-track[i]['t']<.5:continue
        pixels=np.array([track[j]['ball']['pixel'] for j in window])
        if np.linalg.norm(np.ptp(pixels,axis=0))<width*.022:
            rejected.update(window)
    for i in rejected:track[i]['ball']=None
    return len(rejected)
