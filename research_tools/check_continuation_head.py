"""Read-only activation/target diagnostic; no SGD, PPA or policy mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simplified_rapidchiplet'))
from simple_rapidchiplet.hybrid.cost_value_q import CostValueQ


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    checkpoint=json.loads((args.case/'rl/checkpoint.json').read_text(encoding='utf-8'))
    path=args.case/'rl'/checkpoint['policy_file']
    header=torch.load(path,map_location='cpu',weights_only=False)
    if header['representation']!=CostValueQ.representation:raise ValueError('Original CostValueQ only')
    model=CostValueQ(header['size'],header['after_indices']);model.load(path)
    warm=CostValueQ(header['size'],header['after_indices']);warm.load(args.case/'rl/policy_after_shared.pt')
    optimizer_steps=[int(warm.optimizer.state.get(p,{}).get('step',0)) for p in warm.model.future.parameters()]
    rows=header['replay'];x=torch.as_tensor(np.stack([r[0] for r in rows]),dtype=torch.float32)
    activations=[]
    with torch.no_grad():
        z=x
        for module in model.model.future:
            z=module(z)
            activations.append(dict(layer=type(module).__name__,mean_column_std=float(z.std(dim=0,unbiased=False).mean()),
                almost_constant_columns=int((z.std(dim=0,unbiased=False)<1e-6).sum()),columns=z.shape[1],
                abs_over_099_fraction=float((z.abs()>.99).float().mean()) if isinstance(module,torch.nn.Tanh) else None))
        h=model.model.future(x)[:,0].numpy();v=model.model(x)[:,1].numpy()
        targets=[]
        for feature,reward,successors,done,p0,p1 in rows:
            future=0.
            if not done and len(successors):
                nx=torch.as_tensor(successors)
                pick=int(model.model(nx)[:,0].argmax())
                future=p1+float(model.target(nx)[pick,0])
            targets.append(reward-p0+model.gamma*future)
        target_h=np.asarray(targets)-v
    result=dict(representation=header['representation'],source_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        training_rows=len(rows),nonterminal_rows=sum(not r[3] for r in rows),
        input_max_abs=float(x.abs().max()),input_mean_column_std=float(x.std(dim=0,unbiased=False).mean()),
        continuation_std=float(h.std()),continuation_range=float(np.ptp(h)),cost_head_std=float(v.std()),
        future_Adam_steps_after_V_only_warmup=optimizer_steps,
        bootstrapped_continuation_target_std=float(target_h.std()),continuation_target_MAE=float(np.abs(h-target_h).mean()),
        activations=activations,new_PPA=0,new_SGD=0,
        scope='training activation and current Double-Q bootstrap diagnostic; targets are not measured optimal returns or a held-out search result')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
