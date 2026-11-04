"""Reproduce reward-head ranking from public weights/traces; no PPA or SGD."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'simplified_rapidchiplet'),str(ROOT/'simplified_rapidchiplet/scripts')]
from simple_rapidchiplet.hybrid.multi_step_q import MultiStepQ
from analyze_replay_mapping import ranking_metrics


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def signature(state):
    mapping=json.loads(json.dumps(state['mapping']))
    for i,g in enumerate(mapping['groups']):g['id']='group'+str(i)
    return hashlib.sha256(json.dumps(dict(architecture=state['architecture_candidate_id'],mapping=mapping),sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=(7,17),default=17)
    parser.add_argument('--out',type=Path);args=parser.parse_args();case=HERE/str(args.seed)
    manifest=read(HERE/'manifest.json')['files']
    required={f'{args.seed}/inference/{stage}.pt' for stage in ('initial','after_shared','final')}
    required|={f'{args.seed}/inference/trained_physical_signatures.json',f'{args.seed}/sa_events.jsonl.gz',f'{args.seed}/random_events.jsonl.gz'}
    if not required.issubset(manifest):raise ValueError('incomplete public inference manifest')
    for name,digest in manifest.items():
        if name.startswith(str(args.seed)+'/') and ('inference/' in name or name.endswith('_events.jsonl.gz')):
            if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:raise ValueError('public evidence hash mismatch: '+name)
    known=set(read(case/'inference/trained_physical_signatures.json')['signatures'])
    controls=[];seen=set()
    for method in ('sa','random'):
        with gzip.open(case/(method+'_events.jsonl.gz'),'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);key=signature(row['action']['state'])
                if row.get('feasible') and row.get('reward') is not None and key not in known and key not in seen:
                    controls.append(row);seen.add(key)
    if len(controls)<3:raise ValueError('insufficient independent held-out physical states')
    x=np.asarray([r['features'] for r in controls],dtype=np.float32);y=np.asarray([r['reward'] for r in controls])
    result=dict(seed=args.seed,held_out_count=len(controls),new_PPA=0,SGD=0,
        scope='original v05 auxiliary immediate-reward head, stricter canonical physical-state exclusion; not future-Q, online policy causality, or a v06 outcome')
    for stage in ('initial','after_shared','final'):
        data=torch.load(case/'inference'/(stage+'.pt'),map_location='cpu',weights_only=True)
        q=MultiStepQ(data['size'],base_size=data['base_size']);q.model.load_state_dict(data['model']);q.model.eval()
        result[stage]=dict(original_SGD_updates=data['original_SGD_updates'],**ranking_metrics(q.predict_reward(x),y))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
