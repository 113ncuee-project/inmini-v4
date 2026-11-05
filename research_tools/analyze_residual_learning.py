"""Predict-only fixed-corpus diagnostics for frozen-prior residual Double Q.

Full-Q is not scored against immediate cost: its continuation has a different
target. A changed ranking demonstrates an active branch, not useful foresight.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def events(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def context_pairs(rows, *, same_after=False):
    """Retrospective, equal-before-state comparisons; not original action pools."""
    strata={};costs={}
    from analyze_policy_learning import group_key
    for i,row in enumerate(rows):
        before=row.get('before_state_hash')
        if not before:raise ValueError('Missing before-state identity in control evidence')
        cost=float(row['before_cost'])
        if before in costs and not np.isclose(cost,costs[before],rtol=1e-10,atol=1e-12):
            raise ValueError('One before-state hash has different measured physical costs')
        costs[before]=cost
        key=(before,group_key(row['action']['state'])) if same_after else before
        strata.setdefault(key,[]).append(i)
    groups=[indices for indices in strata.values() if len(indices)>1]
    return [pair for indices in groups for pair in combinations(indices,2)],groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--development-legacy-controls',action='store_true')
    args=parser.parse_args()
    case,control=args.case.resolve(),args.controls.resolve()
    source=case/'source_snapshot'
    sys.path[:0]=[str(source),str(source/'scripts'),str(ROOT/'research_tools')]
    from run_local_rl_diagnostics import sources
    from simple_rapidchiplet.hybrid.residual_cost_q import ResidualCostQ
    from simple_rapidchiplet.hybrid.shared_evidence import verify_physical_sources
    from analyze_policy_learning import signature,group_key,concordance,correlation
    manifest,other_manifest=read(case/'source_snapshot_manifest.json'),read(control/'source_snapshot_manifest.json')
    if sources(source)!=manifest['files']:raise ValueError('Learner frozen source changed')
    legacy=manifest['identity']!=other_manifest['identity']
    if legacy:
        if not args.development_legacy_controls:raise ValueError('Different sources require an explicit DEVELOPMENT-only diagnostic')
        verify_physical_sources(other_manifest['files'],manifest['files'])
        def functions(path):
            return {f.name:ast.dump(f,include_attributes=False) for c in ast.parse(path.read_text(encoding='utf-8-sig')).body
                    if isinstance(c,ast.ClassDef) and c.name=='FullDesignSpace' for f in c.body if isinstance(f,ast.FunctionDef)}
        before=functions(control/'source_snapshot/simple_rapidchiplet/hybrid/full_space.py')
        after=functions(source/'simple_rapidchiplet/hybrid/full_space.py')
        if any(value!=after.get(name) for name,value in before.items()):raise ValueError('Legacy decision or feature code differs')
    config,other=read(case/'effective_config.json'),read(control/'effective_config.json')
    for key in ('objective','guidance','no_boundary_features','service_features','operators','max_fusion_depth'):
        if config.get(key)!=other.get(key):raise ValueError('Feature/domain/reference mismatch: '+key)
    if read(case/'common.json')['reference']!=read(control/'common.json')['reference']:raise ValueError('Physical objective reference differs')
    shared=read(case/'screening.json')['rows']
    known={signature(row['trial']['state']) for row in shared if row.get('cost') is not None}
    known.add(signature(read(case/'common.json')['initial_state']))
    known.update(signature(row['action']['state']) for row in events(case/'rl/events.jsonl') if row.get('new_full_ppa'))
    held=[];seen=set()
    for method in ('sa','random'):
        for row in events(control/method/'events.jsonl'):
            key=signature(row['action']['state'])
            if row.get('feasible') and row.get('cost') is not None and key not in known and key not in seen:
                held.append(row);seen.add(key)
    if len(held)<3:raise ValueError('Too few untouched control designs')
    x=np.asarray([r['features'] for r in held],np.float32)
    cost=np.asarray([r['cost'] for r in held]);truth=-10*np.log(cost)
    strata={}
    for i,row in enumerate(held):strata.setdefault(group_key(row['action']['state']),[]).append(i)
    pairs=[pair for indices in strata.values() for pair in combinations(indices,2)]
    before_pairs,before_groups=context_pairs(held)
    fine_pairs,fine_groups=context_pairs(held,same_after=True)
    checkpoint=read(case/'rl/checkpoint.json');stages={}
    final=None
    for stage,name in (('initial','policy_initial.pt'),('after_shared','policy_after_shared.pt'),('final',checkpoint['policy_file'])):
        path=case/'rl'/name;header=torch.load(path,map_location='cpu',weights_only=False)
        if header['representation']!=ResidualCostQ.representation:raise ValueError('Wrong frozen residual representation')
        policy=ResidualCostQ(header['size'],header['after_indices'],header['local_indices']);policy.load(path)
        prediction=policy.predict_reward(x,1.)
        with torch.no_grad():h=policy.model.future(torch.as_tensor(x))[:,0].numpy()
        stages[stage]=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),SGD=policy.updates,
            after_cost_spearman=correlation(prediction,truth),after_potential_MAE=float(np.abs(prediction-truth).mean()),
            best_actual_control_rank=int((prediction>prediction[cost.argmin()]).sum())+1,
            same_architecture_and_group=concordance(prediction,truth,pairs),
            same_before_state=dict(**concordance(prediction,truth,before_pairs),groups=len(before_groups)),
            same_before_and_after_architecture_group=dict(**concordance(prediction,truth,fine_pairs),groups=len(fine_groups)),
            continuation_std=float(h.std()),continuation_range=float(np.ptp(h)),residual=policy.residual.diagnostics())
        final=policy
    changed=0;pool_count=0
    for _,_,successors,done,_,_ in final.replay:
        if done or not len(successors):continue
        pool_count+=1
        changed+=int(final.predict(successors,1.).argmax()!=final.predict_reward(successors,1.).argmax())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_identity=manifest['identity'],control_source=other_manifest['identity'],
        random_seed=config['seed'],development_only=legacy,held_out_count=len(held),models=stages,
        continuation_changes_argmax_pools=changed,recorded_nonterminal_pools=pool_count,
        heldout_state_hashes=[signature(r['action']['state']) for r in held],new_PPA=0,new_SGD=0,images=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='same unseen canonical control corpus at3model stages; after-cost prediction includes frozenV+learnedR and excludes known before term. Same-before comparisons also hold measured starting design/cost fixed, optionally holding after-architecture/groups fixed; these are retrospective subsets, not original full candidate-pool counterfactuals or independent RNG samples. H ranking changes are not scored as accurate future returns. Legacy control labels may have guided earlier algorithm development; explicit development-only. Not new DSE or causal search efficacy.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('heldout_state_hashes','scope')},indent=2))


if __name__=='__main__':main()
