"""INMIN v0.4 mapping-only replay-Q experiments on an existing Tangram seed.

Examples:
  python simplified_rapidchiplet/replay_mapping_dse.py --out output/v04_r18
  python simplified_rapidchiplet/replay_mapping_dse.py --resume --out output/v04_r18
All accepted results use the complete resource-calendar oracle. There is no
surrogate PPA counted as a full evaluation, and no cloud/LLM dependency.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np

APP=Path(__file__).resolve().parent
sys.path.insert(0,str(APP/'scripts'))
from run_local_rl_diagnostics import read,write,sources,sha,lock,environment
from profile_mapping_oracle import load_workload
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.rl_search import JointObjective
from simple_rapidchiplet.hybrid.search import _rng_tuple,plan_from_dict
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.single_array_runtime import SingleArrayOraclePool
from simple_rapidchiplet.hybrid.mapping_proposals_v4 import StructuredMappings,MappingTrial
from simple_rapidchiplet.hybrid.replay_q import ReplayQ
from simple_rapidchiplet.hybrid.search_space import check_transition

METRICS=('feasible','latency_s','energy_j','power_w','area_mm2','dram_read_bytes','dram_write_bytes',
    'noc_hop_bytes','nop_hop_bytes','executed_macs','recompute_macs','unit_count','calendar_backend','oracle_phase_wall_s','oracle_process_cpu_s')

def compact(m):return {k:m.get(k) for k in METRICS}

def cost(m,reference,objective='balanced'):
    from ppa_preferences import scalar_cost
    return scalar_cost(m,reference,objective)

def append(path,row):
    with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def run_child(args):
    out=Path(args.out);case=Path(args.seed_case)
    manifest=read(out/'source_snapshot_manifest.json')
    if sources(APP)!=manifest['files']:raise ValueError('frozen source hash mismatch')
    input_manifest=read(out/'seed_input_manifest.json')
    for name,digest in input_manifest['files'].items():
        if hashlib.sha256((case/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen seed input hash mismatch: '+name)
    workload=load_workload(case/'workload.json');old=read(case/'seed_attempts.json')
    if not old or not old.get('bootstrap'):raise ValueError('a completed feasible seed archive is required')
    source=read(case/'source_identity.json')
    if source['workload']!=workload.fingerprint:raise ValueError('workload source hash mismatch')
    old_reference=old['bootstrap'][0]['metrics']
    selected=min(old['bootstrap'],key=lambda r:cost(r['metrics'],old_reference,args.objective))
    initial=DesignState.from_dict(selected['state']);candidate=next(c for c in enumerate_single_array_architectures_v3() if c.id==initial.architecture_candidate_id)
    if initial.mapping.fusion_stacks:raise ValueError('v04 first-stage experiments require Fusion Off')
    control=read(out/'status.json',{});control.update(status='running',pid=os.getpid(),source_identity=manifest['identity'])
    write(out/'status.json',control)
    common=read(out/'common.json')
    if not common:
        pool=SingleArrayOraclePool(workload,out/'common_worker',deadline=time.monotonic()+max(60,args.eval_timeout*2+30),evaluation_timeout_s=args.eval_timeout)
        began=time.perf_counter()
        try:
            reference_state=DesignState.from_dict(old['bootstrap'][0]['state'])
            reference_result=pool.evaluate(candidate,reference_state.mapping,phase='v04_common_reference')['metrics']
            initial_result=pool.evaluate(candidate,initial.mapping,phase='v04_common_best_seed')['metrics']
            for actual,expected in ((reference_result,old_reference),(initial_result,selected['metrics'])):
                for key in ('latency_s','energy_j','power_w','area_mm2','executed_macs','dram_read_bytes','dram_write_bytes','noc_hop_bytes','nop_hop_bytes'):
                    if not math.isclose(actual[key],expected[key],rel_tol=1e-12,abs_tol=1e-18):
                        raise ValueError('oracle semantic equivalence failed at '+key)
            common=dict(initial_state=initial.to_dict(),metrics=initial_result,reference=reference_result,
                full_ppa=pool.full_evaluations,wall_s=time.perf_counter()-began,
                seed_source=str(case),workload_hash=workload.fingerprint,original_best_seed_cost=cost(initial_result,reference_result,args.objective),
                equivalence='all original physical metrics match rel_tol=1e-12; every common evaluation charged once per method')
            write(out/'common.json',common)
        finally:pool.close()
    initial=DesignState.from_dict(common['initial_state']);reference=common['reference'];initial_metrics=common['metrics']
    seed_cost=cost(initial_metrics,reference,args.objective)
    began=time.perf_counter()
    bank=StructuredMappings(workload,candidate,initial,inner=not args.no_inner)
    globals=bank.global_pool()
    preparation_wall=time.perf_counter()-began
    if not (out/'proposal_domain.json').exists():
        write(out/'proposal_domain.json',dict(count=len(globals),generation_wall_s=preparation_wall,
        version=bank.version,domain=bank.domain,inner=not args.no_inner,hardware=candidate.hardware.to_dict(),
        scope='same bank and legal controls for all methods; inner ranking is a proposal operation, never full PPA'))
    # Replay pays the measured rebuilding work too, never hiding proposal work
    # in an oracle count or claiming only the fastest phase as DSE wall time.
    preparation_record=read(out/'preparation_accounting.json',[])
    preparation_record.append(dict(wall_s=preparation_wall,resume=bool(preparation_record)))
    write(out/'preparation_accounting.json',preparation_record)
    preparation_total=sum(r['wall_s'] for r in preparation_record)
    summaries=read(out/'summary.json',{})
    for method in args.methods.split(','):
        if method in summaries:continue
        folder=out/method;folder.mkdir(exist_ok=True)
        saved=read(folder/'checkpoint.json',{})
        rng=random.Random(args.seed)
        if saved:rng.setstate(_rng_tuple(saved['rng']))
        acceptance_rng=random.Random(args.seed^0x5A17)
        if saved.get('acceptance_rng'):acceptance_rng.setstate(_rng_tuple(saved['acceptance_rng']))
        elite=DesignState.from_dict(saved['elite']) if saved else initial
        elite_metrics=saved.get('elite_metrics',initial_metrics);elite_cost=saved.get('elite_cost',seed_cost)
        current=DesignState.from_dict(saved['current']) if saved else initial
        current_metrics=saved.get('current_metrics',initial_metrics)
        visited=set(saved.get('visited',[]));steps=saved.get('steps',0);elapsed=saved.get('elapsed_s',0.)
        warmup_best=saved.get('warmup_best_cost',seed_cost);best_step=saved.get('best_step',0)
        greedy_saved=saved.get('greedy_lookahead',{})
        greedy_plans=[MappingTrial(DesignState.from_dict(t['state']),tuple(t['parameters']),t['family'],tuple(t['targets']))
            for t in greedy_saved.get('remaining',[])]
        greedy_base=DesignState.from_dict(greedy_saved['base']) if greedy_saved else initial
        greedy_metrics=greedy_saved.get('metrics',initial_metrics)
        size=len(bank.features(globals[0],initial_metrics,reference,seed_cost))
        policy=ReplayQ(size,args.seed)
        if saved.get('policy_file'):policy.load(folder/saved['policy_file'])
        elif method=='replay_rl':policy.save(folder/'policy_initial.pt')
        start=time.perf_counter();deadline=time.monotonic()+max(.01,args.seconds-elapsed)
        pool=SingleArrayOraclePool(workload,folder/'worker',deadline=deadline,evaluation_timeout_s=args.eval_timeout)
        control.update(method=method,status='running');write(out/'status.json',control)
        stop_reason='full_query_budget_exhausted'
        # A completed orphan response recovers verified PPA but never invents a
        # missing policy update. Its exact mapping may still enter the archive.
        for record in pool.completed_evidence():
            if not record.get('mapping'):continue
            state=replace(initial,mapping=plan_from_dict(record['mapping']))
            m=record['metrics'];value=cost(m,reference,args.objective)
            if value is not None and value<elite_cost:
                elite,elite_metrics,elite_cost=state,m,value
                pending=read(folder/'pending_trial.json',{})
                best_step=pending.get('step') if pending.get('state_hash')==state.fingerprint else None
            visited.add(state.fingerprint)

        def save(status):
            policy_file=None
            if method=='replay_rl':
                policy_file=f'policy_{steps:06d}.pt';tmp=folder/(policy_file+'.tmp');policy.save(tmp);tmp.replace(folder/policy_file)
            payload=dict(version='replay_mapping_v04_3',status=status,elite=elite.to_dict(),elite_metrics=elite_metrics,elite_cost=elite_cost,
                current=current.to_dict(),current_metrics=current_metrics,visited=sorted(visited),steps=steps,rng=rng.getstate(),
                elapsed_s=elapsed+time.perf_counter()-start,policy_file=policy_file,policy_updates=policy.updates,
                warmup_best_cost=warmup_best,best_step=best_step,source_identity=manifest['identity'],
                acceptance_rng=acceptance_rng.getstate())
            if method=='greedy':payload['greedy_lookahead']=dict(base=greedy_base.to_dict(),metrics=greedy_metrics,
                remaining=[t.to_dict() for t in greedy_plans])
            write(folder/'checkpoint.json',payload)
            write(folder/'recommendation.json',dict(state=elite.to_dict(),metrics=elite_metrics,cost=elite_cost))

        try:
            while common['full_ppa']+pool.full_evaluations<args.queries and steps<args.queries*8:
                if (out/'STOP_REQUESTED').exists():stop_reason='interrupted';break
                if time.monotonic()>=deadline:stop_reason='wall_time_budget_exhausted';break
                base,base_metrics=(current,current_metrics) if method=='sa' and steps>=args.warmup else (elite,elite_metrics)
                greedy_plans=[t for t in greedy_plans if t.fingerprint not in visited]
                if method=='greedy' and greedy_plans:
                    base,base_metrics=greedy_base,greedy_metrics;plans=greedy_plans
                else:
                    remaining=[t for t in globals if t.fingerprint not in visited]
                    count=min(max(1,args.shortlist-args.local_candidates),len(remaining))
                    plans=rng.sample(remaining,count) if count else []
                    local=bank.local_pool(base,base_metrics,rng,args.local_candidates,guidance=args.generator=='guided')
                    plans.extend(t for t in local if t.fingerprint not in visited)
                    plans=list({t.fingerprint:t for t in plans}.values())
                    if method=='greedy':greedy_base,greedy_metrics,greedy_plans=base,base_metrics,list(plans)
                before=cost(base_metrics,reference,args.objective)
                if not plans:stop_reason='candidate_domain_exhausted';break
                features=np.stack([bank.features(t,base_metrics,reference,before) for t in plans])
                if args.bottleneck_features=='off':features[:,20:28]=0
                epsilon=max(.08,.35*(1-pool.full_evaluations/max(1,args.queries-common['full_ppa'])))
                phase='uniform_warmup' if pool.full_evaluations<args.warmup else 'learned_selection'
                if method=='replay_rl' and phase=='learned_selection':probabilities,q=policy.probabilities(features,epsilon)
                else:
                    probabilities=np.ones(len(plans))/len(plans)
                    q=policy.predict(features) if method=='replay_rl' else np.zeros(len(plans))
                if method=='greedy':
                    # Complete every predeclared lookahead candidate, charge
                    # each full PPA, retain its minimum, then build next round.
                    phase='full_oracle_lookahead';probabilities=np.zeros(len(plans));probabilities[0]=1.;chosen=0
                    greedy_plans=greedy_plans[1:]
                else:chosen=rng.choices(range(len(plans)),weights=probabilities,k=1)[0]
                trial=plans[chosen];check_transition(initial,trial.state,'mapping_only')
                request=dict(schema_version=1,step=steps+1,phase=phase,method=method,source_identity=manifest['identity'],
                    utc_unix=time.time(),before_state_hash=base.fingerprint,state_hash=trial.fingerprint,action=trial.to_dict(),
                    features=features[chosen].tolist(),candidate_count=len(plans),candidate_hashes=[t.fingerprint for t in plans],
                    predicted_utilities=q.tolist(),probabilities=probabilities.tolist(),behavior_probability=float(probabilities[chosen]),
                    selected_candidate_index=chosen,selected_q_before=float(q[chosen]) if method=='replay_rl' else None,
                    entropy=float(-np.dot(probabilities,np.log(np.maximum(probabilities,1e-300)))/max(math.log(len(plans)),1e-30)),
                    policy_updates_before=policy.updates,before_cost=before)
                write(folder/'pending_trial.json',request)
                tick=time.perf_counter();completed_before=pool.full_evaluations;visited.add(trial.fingerprint);steps+=1
                try:
                    metrics=pool.evaluate(candidate,trial.state.mapping,phase='v04_'+phase)['metrics']
                    value=cost(metrics,reference,args.objective);valid=value is not None
                    improved=valid and value<elite_cost
                    if improved:elite,elite_metrics,elite_cost,best_step=trial.state,metrics,value,steps
                    if phase=='uniform_warmup':warmup_best=min(warmup_best,value) if valid else warmup_best
                    if valid:
                        # Scaled logarithmic utility is monotone in the declared
                        # objective. No bottleneck/byte/handwritten reward bonus.
                        utility=-100*math.log(value/seed_cost)
                        if method=='replay_rl' and pool.full_evaluations>completed_before:
                            policy.observe(features[chosen],utility,args.replay_updates)
                        if method=='sa' and phase!='uniform_warmup':
                            temperature=max(.003,.15*(1-pool.full_evaluations/max(1,args.queries-common['full_ppa'])))
                            if value<=before or acceptance_rng.random()<math.exp(min(0,(before-value)/(temperature*seed_cost))):
                                current,current_metrics=trial.state,metrics
                        else:current,current_metrics=elite,elite_metrics
                    else:utility=None
                    write(folder/'evaluations'/ (trial.fingerprint+'.json'),dict(state=trial.state.to_dict(),metrics=metrics,
                        cost=value,request_step=steps,full_completed=pool.full_evaluations>completed_before))
                    outcome=dict(status='full_completed',new_full_ppa=pool.full_evaluations-completed_before,feasible=valid,
                        cost=value,utility=utility,physical_cost_delta=before-value if valid else None,
                        best_cost=elite_cost,improved_global_best=bool(improved),metrics=compact(metrics))
                except (ValueError,RuntimeError,TimeoutError) as error:
                    outcome=dict(status='timeout' if isinstance(error,TimeoutError) else 'compiler_or_worker_reject',
                        reason=str(error),new_full_ppa=pool.full_evaluations-completed_before,feasible=None,cost=None,utility=None,
                        improved_global_best=False,best_cost=elite_cost,policy_update=False)
                updated=policy.updates>request['policy_updates_before']
                after_predictions=policy.predict(features) if method=='replay_rl' else None
                event={**request,**outcome,'elapsed_evaluation_s':time.perf_counter()-tick,
                    'search_wall_s':elapsed+time.perf_counter()-start,'charged_full_ppa':common['full_ppa']+pool.full_evaluations,
                    'policy_updates_after':policy.updates,'policy_update':updated,
                    'selected_q_after':float(after_predictions[chosen]) if after_predictions is not None else None,
                    'candidate_prediction_drift_l2':float(np.linalg.norm(after_predictions-q)) if after_predictions is not None else None,
                    'gradient_norm':policy.last_gradient_norm if updated else None,
                    'gradient_norm_definition':'last minibatch L2 norm before clipping to 10; null without a new update',
                    'training_loss':float(np.mean(policy.losses[-args.replay_updates:])) if updated else None}
                append(folder/'events.jsonl',event)
                # Preserve the entire pre-decision set only when requested;
                # complete selected mappings are always persisted.
                if args.save_candidate_sets:
                    import gzip
                    raw=dict(before_state=base.to_dict(),plans=[t.to_dict() for t in plans],features=features.tolist())
                    target=folder/'candidates'/f'{steps:06d}.json.gz';target.parent.mkdir(exist_ok=True)
                    target.write_bytes(gzip.compress(json.dumps(raw).encode(),compresslevel=1,mtime=0))
                save('running')
                print(json.dumps(dict(method=method,full_ppa=common['full_ppa']+pool.full_evaluations,step=steps,
                    best_gain_pct=100*(1-elite_cost/seed_cost),policy_updates=policy.updates,entropy=event['entropy'],
                    last_full_s=event['elapsed_evaluation_s'])),flush=True)
            if stop_reason=='full_query_budget_exhausted' and common['full_ppa']+pool.full_evaluations<args.queries:
                stop_reason='proposal_attempt_budget_exhausted'
            save(stop_reason)
        finally:pool.close()
        if stop_reason=='interrupted':
            control.update(status='interrupted',method=method);write(out/'status.json',control);return 130
        summary=dict(method=method,status=stop_reason,common_full_ppa=common['full_ppa'],search_full_ppa=pool.full_evaluations,
            full_ppa=common['full_ppa']+pool.full_evaluations,seed_cost=seed_cost,best_cost=elite_cost,
            gain_vs_seed_pct=100*(1-elite_cost/seed_cost),gain_after_uniform_warmup_pct=100*(1-elite_cost/warmup_best),
            warmup_best_cost=warmup_best,best_step=best_step,policy_updates=policy.updates,
            charged_wall_s=common['wall_s']+preparation_total+elapsed+time.perf_counter()-start,best=compact(elite_metrics),
            preparation_wall_s=preparation_total,
            initial=compact(initial_metrics),objective=args.objective,
            generator=args.generator,bottleneck_features=args.bottleneck_features,
            latency_improvement_pct=100*(1-elite_metrics['latency_s']/initial_metrics['latency_s']),
            energy_improvement_pct=100*(1-elite_metrics['energy_j']/initial_metrics['energy_j']),
            edp_improvement_pct=100*(1-(elite_metrics['latency_s']*elite_metrics['energy_j'])/(initial_metrics['latency_s']*initial_metrics['energy_j'])),
            limitations='analytical PPA; Tangram-style adapter seed, not original Tangram binary; contextual one-step RL gamma=0; warmup and inner optimizer gains separately identified')
        write(folder/'summary.json',summary);summaries[method]=summary;write(out/'summary.json',summaries)
        report(out,summaries,common)
    control.update(status='completed',source_stable=sources(APP)==manifest['files']);write(out/'status.json',control)
    report(out,summaries,common)
    return 0

def report(out,summaries,common):
    lines=['# INMIN v0.4 固定硬體 Mapping 實驗','',
        '全部推薦均經完整資源日曆／流量／容量／能量 oracle 驗證。Neural replay-Q 是 gamma=0 的單步情境式 RL，內層 optimizer 與 warmup 收益不能等同於已學會的 policy 收益。',
        '', '|Method|Full PPA|Cost 改善 %|Latency 改善 %|Energy 改善 %|Warmup 後改善 %|SGD updates|Charged s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries.values():lines.append('|'+ '|'.join(str(r[k]) for k in ('method','full_ppa','gain_vs_seed_pct','latency_improvement_pct','energy_improvement_pct','gain_after_uniform_warmup_pct','policy_updates','charged_wall_s'))+'|')
    lines+=['','SGD updates 不等於新的 PPA；所有 compiler reject／timeout 留在 events.jsonl 與 worker/accounting.json。',
        '固定架構與 Fusion Off；不能把結果當成架構搜尋收益、原版 Tangram/Gemini 實作的直接勝負或 silicon 量測。',
        '同一個 full query ceiling 不代表每方法實際完成數相同；同時讀取查詢與牆鐘。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    default_seed=APP.parent/'inputs/replay_v04/resnet18'
    if not default_seed.exists():default_seed=APP.parent/'output/local_rl_v03_r18/cases/resnet18/7/methods'
    p.add_argument('--seed-case',default=str(default_seed))
    p.add_argument('--out',required=True);p.add_argument('--methods',default='replay_rl,random,sa',help='replay_rl,random,sa,greedy')
    p.add_argument('--queries',type=int,default=512);p.add_argument('--seconds',type=float,default=7200)
    p.add_argument('--eval-timeout',type=float,default=300);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--shortlist',type=int,default=64);p.add_argument('--local-candidates',type=int,default=16)
    p.add_argument('--warmup',type=int,default=4);p.add_argument('--replay-updates',type=int,default=32)
    from ppa_preferences import OBJECTIVES
    p.add_argument('--objective',choices=OBJECTIVES,default='balanced')
    p.add_argument('--generator',choices=('guided','generic'),default='guided')
    p.add_argument('--bottleneck-features',choices=('on','off'),default='on')
    p.add_argument('--no-inner',action='store_true');p.add_argument('--save-candidate-sets',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--frozen-child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();args.out=str(Path(args.out).resolve());args.seed_case=str(Path(args.seed_case).resolve())
    if args.queries<4 or args.seconds<=0 or args.eval_timeout<=0 or args.shortlist<2 or not 0<=args.local_candidates<args.shortlist:
        p.error('positive budgets, queries>=4, shortlist>=2, local_candidates<shortlist required')
    if args.warmup<0 or args.replay_updates<=0:p.error('warmup>=0 and replay_updates>0 required')
    if any(m not in ('replay_rl','random','sa','greedy') for m in args.methods.split(',')):p.error('unknown method')
    if args.frozen_child:return run_child(args)
    out=Path(args.out)
    with lock(APP.parent/'output/.local_rl_diagnostics.lock'),lock(out/'run.lock'):
        config=out/'effective_config.json'
        if args.resume:
            saved=read(config)
            if not saved:raise ValueError('missing existing effective_config')
            for k,v in saved.items():setattr(args,k,v)
            args.resume=True
        else:
            if any(f.name!='run.lock' for f in out.iterdir()):raise ValueError('output not empty; use a new output or --resume')
            original_case=Path(args.seed_case)
            frozen_inputs=out/'seed_input_snapshot';input_hashes={}
            for name in ('workload.json','seed_attempts.json','source_identity.json'):
                source=original_case/name
                target=frozen_inputs/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
                input_hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
            write(out/'seed_input_manifest.json',dict(original_case=str(original_case),files=input_hashes,
                scope='original completed seed evidence copied unchanged; no search or surrogate score introduced'))
            args.seed_case=str(frozen_inputs)
            files=sources(APP);snapshot=out/'source_snapshot'
            for rel in files:
                target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(APP/rel,target)
            write(out/'source_snapshot_manifest.json',dict(files=files,identity=sha(files)))
            write(out/'environment.json',environment());write(config,{k:v for k,v in vars(args).items() if k not in ('resume','frozen_child')})
        manifest=read(out/'source_snapshot_manifest.json')
        if sources(out/'source_snapshot')!=manifest['files']:raise ValueError('source snapshot changed')
        if environment()!=read(out/'environment.json'):raise ValueError('environment differs; do not mix source cohorts')
        input_manifest=read(out/'seed_input_manifest.json')
        if input_manifest:
            for name,digest in input_manifest['files'].items():
                if hashlib.sha256((Path(args.seed_case)/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('frozen seed input hash mismatch: '+name)
        saved_summaries=read(out/'summary.json',{})
        if args.resume and read(out/'status.json',{}).get('status')=='completed' and all(m in saved_summaries for m in args.methods.split(',')):
            print(json.dumps(dict(status='already_completed',out=str(out),additional_ppa=0)),flush=True)
            return 0
        starting_status=read(out/'status.json',{})
        starting_status.update(status='starting',return_code=None)
        write(out/'status.json',starting_status)
        (out/'STOP_REQUESTED').unlink(missing_ok=True)
        command=[sys.executable,'-u','-X','utf8',str(out/'source_snapshot/replay_mapping_dse.py'),'--frozen-child']
        for k,v in vars(args).items():
            if k in ('resume','frozen_child'):continue
            flag='--'+k.replace('_','-')
            if isinstance(v,bool):
                if v:command.append(flag)
            else:command.extend((flag,str(v)))
        with (out/'runner.log').open('ab') as log:
            parent_started=time.perf_counter();parent_started_unix=time.time()
            child=subprocess.Popen(command,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            print(json.dumps(dict(pid=child.pid,out=str(out),log=str(out/'runner.log'))),flush=True)
            try:
                code=child.wait()
            except KeyboardInterrupt:
                (out/'STOP_REQUESTED').touch();code=child.wait()
            child_status=read(out/'status.json',{})
            child_status['return_code']=code
            if code and child_status.get('status')!='interrupted':
                child_status.update(status='failed',return_code=code,log=str(out/'runner.log'))
            write(out/'status.json',child_status)
            invocation=dict(return_code=code,latest_invocation_wall_s=time.perf_counter()-parent_started,
                started_unix=parent_started_unix,finished_unix=time.time(),
                note='Child startup, preparation, sequential arms and cleanup; source-copy/import overhead and historical seed generation are separate.')
            write(out/'parent_execution.json',invocation)
            append(out/'parent_invocations.jsonl',invocation)
            return code

if __name__=='__main__':raise SystemExit(main())
