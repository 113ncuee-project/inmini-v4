"""Read-only held-out cost/ranking and architecture audit: zero PPA, zero SGD.

This reads local, self-generated checkpoints. Never use a control's labels to
train a model. Absolute after-cost scores can be compared across before-states;
full continuation-Q scores from different before-states cannot.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(state):
    state = json.loads(json.dumps(state))
    for i, group in enumerate(state['mapping']['groups']):
        group['id'] = 'group' + str(i)
    return hashlib.sha256(json.dumps(dict(architecture=state['architecture_candidate_id'],
        mapping=state['mapping']), sort_keys=True).encode()).hexdigest()


def ranks(values):
    values = np.asarray(values)
    return np.asarray([(float((values < value).sum()) + .5 * float((values == value).sum()-1))
                       for value in values])


def correlation(x, y):
    a, b = ranks(x), ranks(y)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.engine.resolve()))
    from simple_rapidchiplet.hybrid.cost_value_q import CostValueQ
    from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
    architectures = {c.id: c for c in enumerate_single_array_architectures_v3()}
    case = args.case
    rl = events(case / 'rl/events.jsonl')
    paid = [r for r in read(case / 'screening.json')['rows'] if r.get('cost') is not None]
    known = {signature(r['trial']['state']) for r in paid}
    known |= {signature(r['action']['state']) for r in rl if r.get('feasible') and r.get('new_full_ppa')}
    controls = []
    paths = []
    seen = set()
    for method in ('sa', 'random'):
        path = args.controls / method / 'events.jsonl'
        paths.append(path)
        for row in events(path):
            key = signature(row['action']['state'])
            if row.get('feasible') and row.get('cost') is not None and key not in known and key not in seen:
                controls.append(row)
                seen.add(key)
    if len(controls) < 3:
        raise ValueError('insufficient independent control states')
    x = np.asarray([r['features'] for r in controls], dtype=np.float32)
    y = np.asarray([10*math.log(r['cost']) for r in controls])
    checkpoint = read(case / 'rl/checkpoint.json')
    files = dict(initial=case / 'rl/policy_initial.pt', after_shared=case / 'rl/policy_after_shared.pt',
                 final=case / 'rl' / checkpoint['policy_file'])
    result = dict(scope='held-out absolute after-cost model; not cross-state continuation-Q ranking, '
                  'new search, causal policy benefit, or original-paper comparison', new_PPA=0, SGD=0,
                  source_identity=read(case / 'source_snapshot_manifest.json')['identity'],
                  held_out=len(controls), excluded_trained_physical_states=len(known),
                  controls={str(path.resolve()): digest(path) for path in paths}, stages={}, coverage={})
    counts = np.asarray([architectures[r['action']['state']['architecture_candidate_id']].hardware.compute_count
                         for r in controls])
    for stage, path in files.items():
        data = torch.load(path, map_location='cpu', weights_only=False)
        q = CostValueQ(data['size'], data['after_indices'])
        q.model.load_state_dict(data['model']); q.model.eval()
        with torch.no_grad():
            prediction = -q.model(torch.as_tensor(x))[:, 1].numpy()
        errors = prediction-y
        groups = {}
        for count in sorted(set(counts)):
            mask = counts == count
            groups[str(int(count))] = dict(n=int(mask.sum()),
                log_potential_MAE=float(np.abs(errors[mask]).mean()),
                predicted_over_true_cost_geomean=float(math.exp(float(errors[mask].mean())/10)))
        top = np.argsort(y)[:5]
        result['stages'][stage] = dict(checkpoint_sha256=digest(path), saved_SGD_updates=data['updates'],
            log_potential_MAE=float(np.abs(errors).mean()), cost_spearman=correlation(prediction, y),
            architectures=groups, top_true_cost_states=[dict(method=controls[i]['method'], step=controls[i]['step'],
                architecture=controls[i]['action']['state']['architecture_candidate_id'],
                cost=controls[i]['cost'], predicted_cost=math.exp(float(prediction[i])/10),
                predicted_cost_rank=int((prediction < prediction[i]).sum())+1,
                actual_cost_rank=int((y < y[i]).sum())+1) for i in top])
    for method, rows in [('rl', rl), *[(m, events(args.controls / m / 'events.jsonl')) for m in ('sa', 'random')]]:
        by_arch = defaultdict(list)
        for row in rows:
            by_arch[row['action']['state']['architecture_candidate_id']].append(row)
        result['coverage'][method] = dict(
            selected_families=dict(Counter(r['action']['family'] for r in rows)),
            novel_architectures=len(by_arch),
            architectures={name: dict(attempts=len(rs), full=sum(r.get('new_full_ppa', 0) for r in rs),
                feasible=sum(bool(r.get('feasible')) for r in rs),
                neural_high_probability=sum(r.get('behavior_probability', 0) > .5 for r in rs),
                best_cost=min((r['cost'] for r in rs if r.get('cost') is not None), default=None))
                for name, rs in sorted(by_arch.items())})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(held_out=result['held_out'], new_PPA=0, SGD=0,
        stages={name: {k: value for k, value in data.items() if k != 'architectures'}
                for name, data in result['stages'].items()}, coverage=result['coverage']), indent=2))


if __name__ == '__main__':
    main()
