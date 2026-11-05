"""Prediction-only speed/rounding check; not a DSE/PPA throughput result."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'output/inmin_v12_20261104/continuation_study_v1/engine'
sys.path[:0]=[str(APP),str(APP/'scripts')]
import numpy as np
import torch
from simple_rapidchiplet.hybrid.residual_cost_q import ResidualCostQ

path=ROOT/'output/inmin_v12_20261104/replayed_learning/normalized_37/policy.pt'
data=torch.load(path,map_location='cpu',weights_only=False)
q=ResidualCostQ(data['size'],data['after_indices'],data['local_indices']);q.load(path)
batch=q.replay[-32:]


@torch.no_grad()
def looped():
    result=[];choices=[]
    for _,r,nxt,done,p0,p1 in batch:
        future=0.;choice=None
        if not done and len(nxt):
            tensor=q._tensor(nxt)
            choice=int((q.model(tensor)[:,0].numpy()+q.residual.predict(nxt)).argmax())
            future=p1+float(q.target(tensor)[choice,0])+float(q.target_residual.predict(nxt)[choice])
        result.append(r-p0+q.gamma*future);choices.append(choice)
    return np.asarray(result),choices


@torch.no_grad()
def batched():
    active=[(i,row) for i,row in enumerate(batch) if not row[3] and len(row[2])]
    future=np.zeros(len(batch));choices=[None]*len(batch)
    if active:
        features=np.concatenate([row[2] for _,row in active],axis=0)
        tensor=q._tensor(features)
        online=q.model(tensor)[:,0].numpy()+q.residual.predict(features)
        target=q.target(tensor)[:,0].numpy()+q.target_residual.predict(features)
        offset=0
        for i,row in active:
            count=len(row[2]);pick=int(online[offset:offset+count].argmax())
            future[i]=row[5]+target[offset+pick];choices[i]=pick;offset+=count
    return np.asarray([row[1]-row[4] for row in batch])+q.gamma*future,choices


reference=looped();vector=batched()
times={}
for name,function in (('loop',looped),('batch',batched)):
    started=time.perf_counter();cpu=time.process_time()
    for _ in range(16):function()
    times[name]=dict(wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu,iterations=16)
result=dict(minibatch=len(batch),nonterminal=sum(not r[3] and len(r[2])>0 for r in batch),
    max_target_absolute_difference=float(np.max(np.abs(reference[0]-vector[0]))),
    successor_argmax_identical=reference[1]==vector[1],times=times,
    loop_over_batch_wall_ratio=times['loop']['wall_s']/times['batch']['wall_s'],new_PPA=0,new_SGD=0,
    scope='one saved minibatch target computation only; not total training or DSE acceleration; no code adopted')
out=ROOT/'output/inmin_v12_20261104/bellman_batch_benchmark.json'
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
