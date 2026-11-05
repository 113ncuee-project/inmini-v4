"""Same saved real transitions, no new PPA: development-only learning check."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--app',type=Path,required=True)
parser.add_argument('--case',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
sys.path[:0]=[str(args.app),str(args.app/'scripts')]
import numpy as np
import torch
from run_local_rl_diagnostics import sources,sha
from simple_rapidchiplet.hybrid.residual_cost_q import ResidualCostQ

out=args.out.resolve()
if out.exists():raise ValueError('Fresh offline diagnostic required')
out.mkdir(parents=True)
started=time.perf_counter()
warm_path=args.case/'rl/policy_after_shared.pt'
warm=torch.load(warm_path,map_location='cpu',weights_only=False)
checkpoint=json.loads((args.case/'rl/checkpoint.json').read_text(encoding='utf-8'))
actual_path=args.case/'rl'/checkpoint['policy_file']
actual=torch.load(actual_path,map_location='cpu',weights_only=False)
config=json.loads((args.case/'effective_config.json').read_text(encoding='utf-8'))
# All three fixed feature bases use the same full-space input; copy its
# additive index definition without borrowing control labels or policies.
from simple_rapidchiplet.hybrid.full_space import FullDesignSpace
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from profile_mapping_oracle import load_workload
workload=load_workload(args.case/'seed_input_snapshot/workload.json')
space=FullDesignSpace(workload,enumerate_single_array_architectures_v3())
q=ResidualCostQ(warm['size'],warm['after_indices'],space.residual_local_indices,seed=config['seed'],gamma=config['gamma'])
for feature,label in warm['cost_replay']:
    q.pretrain(feature,math.exp(-label/10),updates=24)
warm_exact=all(torch.equal(value,q.model.state_dict()[key]) for key,value in warm['model'].items() if key.startswith('value.'))
if not warm_exact:raise AssertionError('Offline shared prior differs from the originally paid warmup')
optimizer_future_aged=sum(parameter in q.optimizer.state for parameter in q.model.future.parameters())
for feature,reward,successors,done,p0,p1 in actual['replay']:
    q.observe(feature,reward,successors,done,24,before_cost=math.exp(p0/10),next_cost=math.exp(p1/10))
x=torch.as_tensor(np.stack([r[0] for r in actual['replay']]),dtype=torch.float32)
with torch.no_grad():
    h=q.model.future(x)[:,0].numpy();v=q.model(x)[:,1].numpy()
    targets=[]
    for feature,reward,successors,done,p0,p1 in actual['replay']:
        future=0.
        if not done and len(successors):
            nx=torch.as_tensor(successors)
            chosen=int((q.model(nx)[:,0].numpy()+q.residual.predict(successors)).argmax())
            future=p1+float(q.target(nx)[chosen,0])+q.target_residual.predict(successors)[chosen]
        targets.append(reward-p0+q.gamma*future)
    corrected_h_target=np.asarray(targets)-v-q.residual.predict(x.numpy())
result=dict(source_identity=sha(sources(args.app)),representation=q.representation,shared_prior_bitwise_equal=warm_exact,
    shared_observations=len(warm['cost_replay']),own_real_transitions=len(actual['replay']),
    SGD=q.updates,new_PPA=0,residual_fits=q.residual.fit_count,linear_candidates_fitted=q.residual.linear_candidates_fitted,
    future_parameters_with_aged_optimizer_after_warmup=optimizer_future_aged,
    continuation_std=float(h.std()),continuation_range=float(np.ptp(h)),
    continuation_target_std=float(corrected_h_target.std()),continuation_target_MAE=float(np.abs(h-corrected_h_target).mean()),
    elapsed_s=time.perf_counter()-started,original_checkpoint_sha256=hashlib.sha256(actual_path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='development on already observed real PPA transitions and their original successor features; new SGD/linear fits but zero new PPA; targets bootstrap from each learner, not known optimal returns, not DSE efficacy or fresh confirmation')
q.save(out/'policy.pt')
shutil.copyfile(__file__,out/'replay_continuation.py')
(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
