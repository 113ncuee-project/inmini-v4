"""Equal-budget main-method comparison on completed audited studies; read only."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--all-controls',action='store_true')
    args=parser.parse_args()
    study=args.study
    groups={7:{'rl':'development/stratified/7','sa':'development/controls/7','random':'development/controls/7'},
            37:{k:'confirmation/stratified/37' for k in ('rl','sa','random')},
            47:{k:'confirmation/stratified/47' for k in ('rl','sa','random')}}
    if args.all_controls:
        for seed in groups:
            phase='development' if seed==7 else 'confirmation'
            for label in ('frozen','uniform'):groups[seed][label]=f'{phase}/{label}/{seed}'
    rows=[]
    for seed,cases in groups.items():
        curves={}
        for method,relative in cases.items():
            case=study/relative
            if read(case/'status.json')['status']!='completed' or not read(case/'analysis/audit.json')['passed']:
                raise ValueError('Only completed audited main cases')
            actual='rl' if method in ('frozen','uniform') else method
            shared=read(case/'shared_start.json');summary=read(case/'summary.json')[actual]
            events=[json.loads(line) for line in (case/actual/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            curves[method]=[dict(q=shared['full_ppa'],seconds=shared['wall_s'],cost=shared['cost'])]+[
                dict(q=r['charged_full_ppa'],seconds=shared['wall_s']+r['search_wall_s'],cost=r['best_cost']) for r in events]
            rows.append(dict(seed=seed,scope='completed endpoint; unequal actual queries/time',method=method,budget=None,
                EDP=summary['best_cost'],new_PPA=summary['search_full_ppa'],search_seconds=summary['search_wall_s']))
        for axis in ('q','seconds'):
            budget=min(max(row[axis] for row in curve) for curve in curves.values())
            for method,curve in curves.items():
                rows.append(dict(seed=seed,scope='same '+axis+' budget across all'+str(len(curves))+' methods',method=method,budget=budget,
                    EDP=min(row['cost'] for row in curve if row[axis]<=budget)))
    aggregates=[]
    for scope in dict.fromkeys(r['scope'] for r in rows):
        for method in groups[7]:
            values=[r['EDP'] for r in rows if r['scope']==scope and r['method']==method]
            aggregates.append(dict(scope=scope,method=method,n=len(values),mean_EDP=statistics.mean(values),
                median_EDP=statistics.median(values),sample_std_EDP=statistics.stdev(values),min_EDP=min(values),max_EDP=max(values)))
    result=dict(rows=rows,aggregates=aggregates,all_controls=args.all_controls,new_PPA=0,new_SGD=0,
        scope='3 fixed shared-start RNG trajectories; development7 is not fresh confirmation. Floors differ across RNG but match all3 methods within each RNG. No independence or statistical-significance claim')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
