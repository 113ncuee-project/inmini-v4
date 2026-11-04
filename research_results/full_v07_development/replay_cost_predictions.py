"""Public inference-only reproduction of held-out after-cost diagnostics."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'simplified_rapidchiplet'),str(ROOT/'research_tools')]
from simple_rapidchiplet.hybrid.cost_value_q import CostValueQ
from diagnose_full_policy_v07 import signature,correlation


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);args=parser.parse_args()
    manifest=read(HERE/'manifest.json')['files']
    names=['development/cost_restart/7/screening.json.gz','development/cost_restart/7/rl/events.jsonl.gz',
           *[f'development/selected_controls/7/{method}/events.jsonl.gz' for method in ('sa','random')],
           *[f'inference/{stage}.pt' for stage in ('initial','after_shared','final')]]
    for name in names:
        if name not in manifest or hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=manifest[name]:
            raise ValueError('public evidence hash mismatch: '+name)
    with gzip.open(HERE/names[0],'rt',encoding='utf-8') as stream:shared=json.load(stream)['rows']
    known={signature(r['trial']['state']) for r in shared if r.get('cost') is not None}
    known|={signature(r['action']['state']) for r in lines(HERE/names[1]) if r.get('feasible') and r.get('new_full_ppa')}
    controls=[];seen=set()
    for name in names[2:4]:
        for row in lines(HERE/name):
            key=signature(row['action']['state'])
            if row.get('feasible') and row.get('cost') is not None and key not in known and key not in seen:
                controls.append(row);seen.add(key)
    x=np.asarray([r['features'] for r in controls],np.float32)
    y=np.asarray([10*math.log(r['cost']) for r in controls])
    result=dict(held_out=len(controls),new_PPA=0,SGD=0,stages={},
        scope='absolute after-cost head; no cross-state future-Q or causal search superiority claim')
    for stage in ('initial','after_shared','final'):
        data=torch.load(HERE/'inference'/(stage+'.pt'),map_location='cpu',weights_only=True)
        if data['source_identity']!='b6bf8e77f95b1e47fff56491b248fd9ac55a8884aa505d5e12dcac75bdda2f05':
            raise ValueError('unexpected model source')
        q=CostValueQ(data['size'],data['after_indices']);q.model.load_state_dict(data['model']);q.model.eval()
        prediction=-q.predict_reward(x)
        result['stages'][stage]=dict(saved_SGD=data['original_SGD_updates'],cost_spearman=correlation(prediction,y),
            log_potential_MAE=float(np.abs(prediction-y).mean()),
            best_control_rank=int((prediction<prediction[int(y.argmin())]).sum())+1)
    reference=read(HERE/'heldout_policy_diagnostic.json')
    if len(controls)!=reference['held_out']:raise ValueError('heldout exclusion mismatch')
    for stage,row in result['stages'].items():
        if not math.isclose(row['cost_spearman'],reference['stages'][stage]['cost_spearman'],abs_tol=1e-12):
            raise ValueError('public weights do not reproduce recorded predictions')
    if args.out:args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
