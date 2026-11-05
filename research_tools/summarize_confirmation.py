"""Matched-query/time comparisons from completed audited cohort cases only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def load(case,method):
    status=read(case/'status.json') if (case/'status.json').exists() else {}
    if status.get('status')!='completed':return None,status.get('status','not_started')
    if not read(case/'analysis/audit.json')['passed']:raise ValueError('Failed completed-case audit: '+str(case))
    shared=read(case/'shared_start.json');common=read(case/'common.json')
    summary=read(case/'summary.json')[method];rec=read(case/method/'recommendation.json')
    if not rec['metrics']['feasible'] or not math.isfinite(rec['cost']):raise ValueError('No strictly feasible endpoint')
    rows=[json.loads(line) for line in (case/method/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    curve=[dict(q=shared['full_ppa'],seconds=shared['wall_s'],cost=shared['cost'])]
    for row in rows:
        point=dict(q=row['charged_full_ppa'],seconds=shared['wall_s']+row['search_wall_s'],cost=row['best_cost'])
        if point['q']<curve[-1]['q'] or point['seconds']<curve[-1]['seconds'] or point['cost']>curve[-1]['cost']+1e-12:
            raise ValueError('Nonmonotone charged budget/best curve: '+str(case))
        curve.append(point)
    if curve[-1]['q']!=summary['full_ppa'] or not math.isclose(curve[-1]['cost'],rec['cost'],rel_tol=1e-10):
        raise ValueError('Endpoint differs from durable search curve')
    metrics=rec['metrics']
    return dict(source=read(case/'source_snapshot_manifest.json')['identity'],objective=read(case/'effective_config.json')['objective'],
        reference=common['reference'],shared_state=shared['state'],shared_cost=shared['cost'],
        shared_charges=(shared['full_ppa'],shared['wall_s']),model=rec['metrics']['model'],curve=curve,
        endpoint=dict(EDP=rec['cost'],new_PPA=summary['search_full_ppa'],charged_PPA=summary['full_ppa'],
            search_seconds=summary['search_wall_s'],architecture=metrics['architecture_candidate_id'],
            latency_ms=1000*metrics['latency_s'],energy_mJ=1000*metrics['energy_j'],
            area_mm2=metrics['area_mm2'],power_W=metrics['power_w'])), 'completed'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True);parser.add_argument('--family',default='residual')
    parser.add_argument('--seeds',default='197,207,217');parser.add_argument('--controls',default='myopic,frozen')
    parser.add_argument('--path-template',default='confirmation/{family}/{seed}')
    parser.add_argument('--supplemental',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scope',default='Fresh RNG confirmation; shared-host per-case wall, not isolated single-search acceleration. No significance/native-paper/silicon claim.')
    args=parser.parse_args();seeds=[int(v) for v in args.seeds.split(',')];controls=[v for v in args.controls.split(',') if v]
    if len(seeds)!=len(set(seeds)):raise ValueError('Distinct RNG seeds required')
    data={};pending=[];endpoints=[];matched=[]
    for seed in seeds:
        cases={method:(args.cohort/args.path_template.format(family=args.family,seed=seed),method) for method in ('rl','sa','random')}
        cases.update({label:(args.cohort/args.path_template.format(family=label,seed=seed),'rl') for label in controls})
        if args.supplemental:
            cases['no_residual']=(args.supplemental/args.path_template.format(family='no_residual',seed=seed),'rl')
        methods={}
        for label,(case,actual) in cases.items():
            row,status=load(case,actual)
            if row is None:pending.append(dict(seed=seed,method=label,status=status));continue
            methods[label]=row;endpoints.append(dict(seed=seed,method=label,**row['endpoint']))
        if methods:
            ref=next(iter(methods.values()))
            if ref['objective']!='edp':raise ValueError('This EDP report requires an EDP objective')
            for row in methods.values():
                if any(row[key]!=ref[key] for key in ('source','reference','shared_state','shared_cost','shared_charges','model')):
                    raise ValueError('Compared sources/initial state/objective reference differ')
        data[seed]=methods
        sets={'main':['rl','sa','random'],'all':list(cases)}
        for label in controls+(['no_residual'] if args.supplemental else []):sets['rl_vs_'+label]=['rl',label]
        for name,labels in sets.items():
            if not set(labels)<=set(methods):continue
            for axis in ('q','seconds'):
                floor=min(max(p[axis] for p in methods[label]['curve']) for label in labels)
                for label in labels:
                    value=min(p['cost'] for p in methods[label]['curve'] if p[axis]<=floor)
                    matched.append(dict(seed=seed,comparison=name,axis=axis,budget=floor,method=label,EDP=value))
    completed=[row for methods in data.values() for row in methods.values()]
    if completed:
        ref=completed[0]
        if any(any(row[key]!=ref[key] for key in ('source','reference','shared_state','shared_cost','shared_charges','model')) for row in completed):
            raise ValueError('RNG aggregation requires the same source, model, fixed initial mapping and paid start')
    aggregates=[]
    for key in dict.fromkeys((r['comparison'],r['axis'],r['method']) for r in matched):
        values=[r['EDP'] for r in matched if (r['comparison'],r['axis'],r['method'])==key]
        aggregates.append(dict(comparison=key[0],axis=key[1],method=key[2],n=len(values),mean_EDP=statistics.mean(values),
            median_EDP=statistics.median(values),sample_std_EDP=statistics.stdev(values) if len(values)>1 else None,
            min_EDP=min(values),max_EDP=max(values)))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=args.scope,
        endpoints=endpoints,matched=matched,aggregates=aggregates,pending=pending,
        all_requested_cases_completed=not pending,new_PPA=0,new_SGD=0,images=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        budget_definition='All pre-search query/time charges included per arm. Floors match every method within each comparison set/seed. Endpoints may have unequal budgets. Report all planned seeds including pending, no selective wins.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# 已審計確認案例：相同查詢與時間','',args.scope,'',
        '|RNG|方法|累計PPA|新PPA|搜尋秒|EDP|架構|T ms|E mJ|面積mm²|平均W|',
        '|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in endpoints:
        lines.append(f"|{r['seed']}|{r['method']}|{r['charged_PPA']}|{r['new_PPA']}|{r['search_seconds']:.3f}|{r['EDP']:.9f}|{r['architecture']}|{r['latency_ms']:.6f}|{r['energy_mJ']:.6f}|{r['area_mm2']:.3f}|{r['power_W']:.6f}|")
    lines += ['', '|比較集合|匹配方式|方法|已完成RNG數|平均EDP|中位數|樣本標準差|最小|最大|',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        sd='—' if r['sample_std_EDP'] is None else f"{r['sample_std_EDP']:.9f}"
        lines.append(f"|{r['comparison']}|{r['axis']}|{r['method']}|{r['n']}|{r['mean_EDP']:.9f}|{r['median_EDP']:.9f}|{sd}|{r['min_EDP']:.9f}|{r['max_EDP']:.9f}|")
    lines += ['', f"尚未完成：{json.dumps(pending,ensure_ascii=False)}",'',
        '只納入已完成且audit通過的案例；未完成的主案例不先挑出某個已跑完方法宣稱勝利。比較集合不同會有不同floor，不能混成同一預算。端點PPA／時間不一定相同。PPA仍是未校準解析模型，本報告新增0次PPA／SGD、0張圖片。','']
    args.out.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(reported_endpoints=len(endpoints),matched_rows=len(matched),pending=len(pending),
        complete=not pending,new_PPA=0,new_SGD=0)))


if __name__=='__main__':main()
