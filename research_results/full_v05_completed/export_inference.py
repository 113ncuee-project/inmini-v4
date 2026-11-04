"""Export original completed neural weights for prediction-only reproduction."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def signature(state):
    mapping=json.loads(json.dumps(state['mapping']))
    for i,g in enumerate(mapping['groups']):g['id']='group'+str(i)
    data=dict(architecture=state['architecture_candidate_id'],mapping=mapping)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('workspace',type=Path);args=parser.parse_args()
    target=Path(__file__).resolve().parent
    for seed in (7,17):
        case=args.workspace/'output/inmin_v05_20261103/night_edp/resnet18'/str(seed)
        if read(case/'status.json')['status']!='completed' or not read(case/'analysis/audit.json')['passed']:
            raise ValueError('only audited completed cases may be exported')
        source=read(case/'source_snapshot_manifest.json')['identity']
        checkpoint=read(case/'rl/checkpoint.json')
        out=target/str(seed)/'inference';out.mkdir(exist_ok=True)
        for label,name in [('initial','policy_initial.pt'),('after_shared','policy_after_shared.pt'),('final',checkpoint['policy_file'])]:
            path=case/'rl'/name;data=torch.load(path,map_location='cpu',weights_only=False)
            size=data.get('size',next(v.shape[1] for v in data['model'].values() if v.ndim==2))
            payload=dict(model=data['model'],size=size,base_size=data.get('base_size'),label=label,
                original_SGD_updates=data['updates'],original_source_identity=source,
                original_checkpoint_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='inference weights only; no optimizer, replay, target or resume state; no new PPA or SGD')
            torch.save(payload,out/(label+'.pt'))
        screen=read(case/'screening.json')
        known={signature(r['trial']['state']) for r in screen['rows'] if r.get('cost') is not None}
        for line in (case/'rl/events.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('feasible') and row.get('new_full_ppa'):known.add(signature(row['action']['state']))
        (out/'trained_physical_signatures.json').write_text(json.dumps(dict(signatures=sorted(known),
            scope='all feasible shared labels and valid new RL observations; canonical group names; excludes control labels from all scoring stages'),indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,original_source=source,exported_weights=3,known_physical_states=len(known),new_PPA=0,SGD=0)))


if __name__=='__main__':main()
