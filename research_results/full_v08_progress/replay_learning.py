"""Reproduce published held-out after-cost diagnostics; no PPA or SGD."""
import argparse
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'simplified_rapidchiplet'),str(ROOT/'research_tools')]
from simple_rapidchiplet.hybrid.cost_value_q import CostValueQ
from analyze_policy_learning import signature,group_key,concordance,correlation


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    manifest = read(HERE/'manifest.json')['files']
    locations = read(HERE/'payload_locations.json')
    original = read(HERE/'original_sha256.json')

    def verified(name):
        path = HERE/name
        if name not in manifest or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]:
            raise ValueError('Public payload hash mismatch: '+name)
        return path

    def payload(name):
        location = locations[name]
        path = verified(location['path'])
        data = gzip.decompress(path.read_bytes()) if location['gzip'] else path.read_bytes()
        if hashlib.sha256(data).hexdigest() != original[name]:
            raise ValueError('Original payload mismatch: '+name)
        return data

    def rows(name):
        return [json.loads(line) for line in payload(name).decode('utf-8').splitlines() if line.strip()]

    results = []
    for item in read(verified('learning_cases.json')):
        learned,control,seed = item['learned'],item['controls'],item['seed']
        reference = read(verified(item['reference']))
        shared = json.loads(payload(learned+'/screening.json'))['rows']
        common = json.loads(payload(learned+'/common.json'))
        own = rows(learned+'/rl/events.jsonl')
        known = {signature(r['trial']['state']) for r in shared if r.get('cost') is not None}
        known.add(signature(common['initial_state']))
        known.update(signature(r['action']['state']) for r in own if r.get('new_full_ppa'))
        held,seen = [],set()
        for method in ('sa','random'):
            for row in rows(control+'/'+method+'/events.jsonl'):
                key = signature(row['action']['state'])
                if row.get('feasible') and row.get('cost') is not None and key not in known and key not in seen:
                    held.append(row)
                    seen.add(key)
        x = np.asarray([r['features'] for r in held],np.float32)
        cost = np.asarray([r['cost'] for r in held])
        truth = -10*np.log(cost)
        strata = {}
        for i,row in enumerate(held):
            strata.setdefault(group_key(row['action']['state']),[]).append(i)
        pairs = [pair for ids in strata.values() for pair in combinations(ids,2)]
        result = dict(seed=seed,held_out=len(held),stages={})
        if result['held_out'] != reference['held_out_count']:
            raise ValueError('Held-out exclusion differs')
        for stage in ('initial','after_shared','final'):
            path = verified(f'inference/seed{seed}/{stage}.pt')
            data = torch.load(path,map_location='cpu',weights_only=True)
            if data['original_checkpoint_sha256'] != reference['models'][stage]['checkpoint_sha256']:
                raise ValueError('Weights provenance differs')
            q = CostValueQ(data['size'],data['after_indices'])
            q.model.load_state_dict(data['model'])
            q.model.eval()
            with torch.no_grad():
                prediction = q.model(torch.as_tensor(x))[:,1].numpy()
            values = dict(after_cost_spearman=correlation(prediction,truth),
                after_potential_MAE=float(np.abs(prediction-truth).mean()),
                same_arch_group_concordance=concordance(prediction,truth,pairs)['concordance'],
                original_SGD=data['original_SGD_updates'])
            expected = reference['models'][stage]
            for key in ('after_cost_spearman','after_potential_MAE'):
                if not math.isclose(values[key],expected[key],rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('Prediction does not reproduce: '+str(seed)+'/'+stage+'/'+key)
            if not math.isclose(values['same_arch_group_concordance'],expected['same_architecture_and_group']['concordance'],abs_tol=1e-12):
                raise ValueError('Fine-mapping pair diagnostic differs')
            result['stages'][stage] = values
        results.append(result)
    output = dict(passed=True,results=results,new_PPA=0,new_SGD=0,images=0,
        scope='safe inference-only weights and exact original public payloads; same corpus/exclusion/metrics reproduced; '
              'not future-Q, original candidate-pool, new search or independent statistical superiority')
    if args.out:
        args.out.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
