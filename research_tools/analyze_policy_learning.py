"""Measure learned after-cost ranking on untouched control designs.

The exact measured before-cost term is removed. Otherwise a correlation of
reward predictions can look good just because both contain the same known
before cost. This is a held-out diagnostic, not a new search or proof about
the continuation Q head. No PPA or training is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'simplified_rapidchiplet'), str(Path(__file__).resolve().parent)]
from simple_rapidchiplet.hybrid.cost_value_q import CostValueQ
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.full_space import decision_hash
from diagnose_full_policy_v07 import correlation


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def signature(data):
    return decision_hash(DesignState.from_dict(data))


def group_key(data):
    return (data['architecture_candidate_id'], tuple((g['mode'], tuple(g['layer_ids'])) for g in data['mapping']['groups']))


def concordance(prediction, truth, pairs):
    concordant = 0.
    count = 0
    for i,j in pairs:
        difference = truth[i]-truth[j]
        if abs(difference) <= 1e-10:
            continue
        score = prediction[i]-prediction[j]
        concordant += .5 if abs(score) <= 1e-10 else float(score*difference > 0)
        count += 1
    return dict(non_tied_pairs=count, concordance=concordant/count if count else None)


def analyze(learned, control, output):
    learned, control, output = learned.resolve(), control.resolve(), output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Diagnostic output must remain in workspace')
    config = read(learned/'effective_config.json')
    other = read(control/'effective_config.json')
    lm, cm = read(learned/'source_snapshot_manifest.json'), read(control/'source_snapshot_manifest.json')
    if lm['identity'] != cm['identity'] or config['objective'] != other['objective'] or read(learned/'common.json')['reference'] != read(control/'common.json')['reference']:
        raise ValueError('Control source, objective or physical reference differs')
    shared = read(learned/'screening.json')['rows']
    trained = {signature(row['trial']['state']) for row in shared if row.get('cost') is not None}
    trained.add(signature(read(learned/'common.json')['initial_state']))
    trained.update(signature(row['action']['state']) for row in events(learned/'rl/events.jsonl') if row.get('new_full_ppa'))
    held, seen = [], set()
    for method in ('sa','random'):
        for row in events(control/method/'events.jsonl'):
            key = signature(row['action']['state'])
            if row.get('feasible') and row.get('cost') is not None and key not in trained and key not in seen:
                held.append(row)
                seen.add(key)
    if len(held) < 3:
        raise ValueError('Too few untouched control designs')
    x = np.asarray([r['features'] for r in held], dtype=np.float32)
    cost = np.asarray([r['cost'] for r in held])
    truth = -10*np.log(cost)
    before = np.asarray([r['before_cost'] for r in held])
    actual_reward = 10*np.log(before/cost)
    strata = {}
    for i,row in enumerate(held):
        strata.setdefault(group_key(row['action']['state']), []).append(i)
    same_arch_group_pairs = [pair for indices in strata.values() for pair in combinations(indices,2)]
    checkpoint = read(learned/'rl/checkpoint.json')
    models = {}
    for label,name in (('initial','policy_initial.pt'),('after_shared','policy_after_shared.pt'),('final',checkpoint['policy_file'])):
        path = learned/'rl'/name
        header = torch.load(path, map_location='cpu', weights_only=False)
        if header.get('representation') != CostValueQ.representation:
            raise ValueError('This diagnostic explicitly requires a cost-factored Q')
        policy = CostValueQ(x.shape[1], header['after_indices'])
        policy.load(path)
        with torch.no_grad():
            outputs = policy.model(torch.as_tensor(x)).numpy()
        prediction = outputs[:,1]
        rank = np.argsort(-prediction, kind='stable')
        best = int(cost.argmin())
        k = max(1, math.ceil(len(held)/4))
        models[label] = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            SGD=policy.updates, after_cost_spearman=correlation(prediction, truth),
            after_potential_MAE=float(np.abs(prediction-truth).mean()),
            after_logcost_MSE=float(np.square(prediction-truth).mean()),
            best_actual_control_rank=int((prediction > prediction[best]).sum())+1,
            selected_one_actual_cost=float(cost[rank[0]]),
            selected_quartile_mean_actual_cost=float(cost[rank[:k]].mean()),
            all_mean_actual_cost=float(cost.mean()),
            same_architecture_and_group=concordance(prediction, truth, same_arch_group_pairs),
            reward_spearman_with_exact_before_term=correlation(prediction+10*np.log(before), actual_reward),
            continuation_component_mean=float((outputs[:,0]-outputs[:,1]).mean()),
            continuation_component_std=float((outputs[:,0]-outputs[:,1]).std()))
    learning_path = learned/'rl/learning.jsonl'
    learning = events(learning_path) if learning_path.exists() else []
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_identity=lm['identity'],
        learned_case=str(learned), control_case=str(control), objective=config['objective'],
        random_seed=config['seed'], held_out_count=len(held), observed_RL_transitions=len(learning),
        nonterminal_RL_transitions=sum(not r['terminal'] for r in learning),
        physical_shared_samples=sum(r.get('cost') is not None for r in shared),
        models=models, actual_best_heldout_cost=float(cost.min()),
        heldout_state_hashes=[signature(r['action']['state']) for r in held],
        new_PPA=0, new_SGD=0, images=0,
        scope='fixed identical unseen control corpus for initial/shared/final comparison; canonical group-label-invariant '
              'exclusion of all RL-observed physical designs and shared costs; removes exact known before cost; '
              'same-architecture/group pair check limits gross architecture confounding. '
              'Not an original candidate-pool counterfactual, independent RNG statistic, proof of future-Q accuracy or a search-speed result.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('heldout_state_hashes','scope')},indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learned-case',type=Path,required=True)
    parser.add_argument('--control-case',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    analyze(args.learned_case,args.control_case,args.out)
