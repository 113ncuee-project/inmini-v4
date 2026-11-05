"""Replay published, inference-only residual-Q diagnostics; no PPA or training."""
import argparse
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'simplified_rapidchiplet'),str(ROOT/'simplified_rapidchiplet/scripts'),str(ROOT/'research_tools')]
from run_local_rl_diagnostics import sources,sha
from simple_rapidchiplet.hybrid.residual_cost_q import ResidualCostQ
from analyze_policy_learning import concordance,correlation


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(folder):
    folder=folder.resolve()
    manifest=read(folder/'manifest.json')
    for name,expected in manifest['files'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Evidence bytes differ: '+name)
    for name,entry in read(folder/'payload_locations.json').items():
        value=(folder/entry['path']).read_bytes()
        if entry['gzip']:value=gzip.decompress(value)
        if hashlib.sha256(value).hexdigest()!=entry['original_sha256']:
            raise ValueError('Original evidence bytes differ: '+name)


def pairs(rows,keys):
    groups={}
    for i,row in enumerate(rows):groups.setdefault(tuple(row[k] for k in keys),[]).append(i)
    selected=[g for g in groups.values() if len(g)>1]
    return [p for g in selected for p in combinations(g,2)],len(selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=ROOT/'research_results/full_v09_development')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();folder=args.evidence.resolve();verify(folder)
    torch.set_num_threads(1)
    receipt=read(folder/'inference_receipt.json')
    if sha(sources(ROOT/'simplified_rapidchiplet'))!=receipt['source_identity']:
        raise ValueError('Released learner/source differs from inference evidence')
    corpus=read(folder/'inference_corpus.json');rows=corpus['rows']
    if set(r['state_hash'] for r in rows)&set(corpus['excluded_observed_state_hashes']):
        raise ValueError('Corpus contains an RL-observed/shared design')
    x=np.asarray([r['features'] for r in rows],dtype=np.float32)
    costs=np.asarray([r['cost'] for r in rows]);truth=-10*np.log(costs)
    expected=read(folder/'learning_diagnostic.json');models={};last=None
    for label in ('initial','after_shared','final'):
        header=torch.load(folder/f'inference_{label}.pt',map_location='cpu',weights_only=True)
        if set(header)!={'format','source_identity','representation','size','after_indices','local_indices','SGD','source_checkpoint_sha256','model','residual_model'}:
            raise ValueError('Inference model has an unexpected field')
        if header['representation']!=ResidualCostQ.representation or header['source_identity']!=receipt['source_identity']:
            raise ValueError('Wrong published representation')
        learner=ResidualCostQ(header['size'],header['after_indices'],header['local_indices'])
        learner.model.load_state_dict(header['model']);learner.model.eval()
        residual=header['residual_model']
        if residual is not None:
            learner.residual.model=(tuple(residual['indices']),residual['scale'].numpy(),residual['coefficient'].numpy())
        prediction=learner.predict_reward(x,1.)
        with torch.inference_mode():h=learner.model.future(torch.as_tensor(x))[:,0].numpy()
        result=dict(after_cost_spearman=correlation(prediction,truth),
            after_potential_MAE=float(np.abs(prediction-truth).mean()),
            best_actual_control_rank=int((prediction>prediction[costs.argmin()]).sum())+1,
            continuation_std=float(h.std()),continuation_range=float(np.ptp(h)))
        for key,fields in (('same_architecture_and_group',('after_group',)),
                           ('same_before_state',('before_state_hash',)),
                           ('same_before_and_after_architecture_group',('before_state_hash','after_group'))):
            indices,count=pairs(rows,fields)
            result[key]=concordance(prediction,truth,indices)
            if key!='same_architecture_and_group':result[key]['groups']=count
        if header['source_checkpoint_sha256']!=expected['models'][label]['checkpoint_sha256']:
            raise ValueError('Original model provenance differs')
        for key,value in result.items():
            target=expected['models'][label][key]
            if isinstance(value,dict):
                for name,number in value.items():
                    if not np.isclose(number,target[name],rtol=1e-6,atol=1e-7):raise ValueError('Ranking replay differs: '+label+'/'+key+'/'+name)
            elif not np.isclose(value,target,rtol=1e-6,atol=1e-7):raise ValueError('Model replay differs: '+label+'/'+key)
        result['SGD_previously_paid']=header['SGD'];models[label]=result;last=learner
    pools=torch.load(folder/'successor_pool_inputs.pt',map_location='cpu',weights_only=True)
    changed=sum(int(last.predict(x.numpy(),1.).argmax()!=last.predict_reward(x.numpy(),1.).argmax()) for x in pools)
    if len(pools)!=expected['recorded_nonterminal_pools'] or changed!=expected['continuation_changes_argmax_pools']:
        raise ValueError('Recorded successor-pool ranking differs')
    result=dict(passed=True,source_identity=receipt['source_identity'],untouched_development_designs=len(rows),
        models=models,continuation_changed_argmax=changed,recorded_nonterminal_pools=len(pools),
        new_PPA=0,new_SGD=0,images=0,
        scope='Inference-only replay on development controls; not fresh generalization, useful-foresight proof or new DSE. Exported models contain no optimizer, training replay or RNG; successor inputs have no reward labels.')
    if args.out:args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
