"""Frozen, bounded concurrent full-PPA experiments with one shared start/model.

Runs independent RNG cases, not parallel surrogate estimates or extra RL agents
inside one case. All physical calls, initialization, wall/CPU time and policies
retain their normal ledgers. Cohort shared locks block unmanaged experiments.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

APP = Path(__file__).resolve().parents[1]
ROOT = next((p for p in APP.parents if (p/'inputs/replay_v04').is_dir() and (p/'.venv').is_dir()),APP.parent)
sys.path.insert(0,str(APP/'scripts'))
from run_local_rl_diagnostics import sources,sha,lock,read,write,environment
from full_cohort import shared_lock


def default_jobs(args):
    models = [v.strip() for v in args.models.split(',') if v.strip()]
    seeds = [int(v) for v in args.seeds.split(',')]
    if len(set(models)) != len(models) or len(set(seeds)) != len(seeds) or not models or not seeds:
        raise ValueError('Distinct models and RNG seeds required')
    if (args.seed_case or args.shared_case) and len(models) != 1:
        raise ValueError('Explicit seed/shared case is for one model at a time')
    jobs = []
    for model in models:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',model):
            raise ValueError('Unsafe model output identifier')
        first = None
        for seed in seeds:
            identifier = model+'_'+str(seed)
            flags = ['--queries',str(args.queries),'--seconds',str(args.seconds),'--eval-timeout',str(args.eval_timeout),
                     '--objective',args.objective,'--architecture-screen',args.architecture_screen,
                     '--screen-seconds',str(args.screen_seconds),'--seed-refinement',args.seed_refinement,
                     '--seed-queries',str(args.seed_queries),'--seed-seconds',str(args.seed_seconds),
                     '--seed-dp-seconds',str(args.seed_dp_seconds),'--max-fusion-depth',str(args.max_fusion_depth),
                     '--architecture-refresh-interval',str(args.architecture_refresh_interval),
                     '--episode-cost-rise',str(args.episode_cost_rise),'--episode-restart-probability','.9']
            if args.policy in ('cost','residual'):
                flags += ['--cost-value-q']
            if args.policy == 'residual':
                flags += ['--residual-cost-q']
            for key in ('max_area','max_power','max_latency_ms'):
                value = getattr(args,key)
                if value is not None:
                    flags += ['--'+key.replace('_','-'),str(value)]
            job = dict(id=identifier,out=f'cases/{model}/{seed}',seed=seed,methods=args.methods,
                       seed_case=str(Path(args.seed_case).resolve() if args.seed_case else ROOT/'inputs/replay_v04'/model),
                       arguments=flags)
            if args.shared_case:
                job['shared_case']=str(Path(args.shared_case).resolve())
            elif first:
                job['share_from']=first
            jobs.append(job)
            first = first or identifier
    return jobs


def prepare(out,args):
    if args.resume:
        protocol = read(out/'cohort_protocol.json')
        if not protocol:
            raise ValueError('No cohort to resume')
        return protocol
    if (out/'cohort_protocol.json').exists():
        raise ValueError('Cohort exists; use --resume or new output')
    plan = read(Path(args.plan)) if args.plan else dict(jobs=default_jobs(args))
    jobs = plan.get('jobs',[])
    if not jobs or not 1 <= args.jobs <= 4:
        raise ValueError('Nonempty plan and1..4 concurrent jobs required')
    ids,outputs = set(),set()
    reserved = {'--out','--seed','--seed-case','--shared-case','--methods','--resume','--frozen-child','--ppa-cohort','--ppa-slot'}
    normalized = []
    for raw in jobs:
        job = dict(raw)
        if not re.fullmatch(r'[A-Za-z0-9_-]+',job['id']) or job['id'] in ids:
            raise ValueError('Job IDs must be unique safe identifiers')
        ids.add(job['id'])
        target = (out/job['out']).resolve()
        if (not target.is_relative_to(out) or target == out or str(target) in outputs
                or any(target.is_relative_to(out/name) for name in ('engine','seed_inputs','slots'))):
            raise ValueError('Case output escapes its cohort')
        outputs.add(str(target))
        job['out']=str(target)
        if not isinstance(job.get('arguments',[]),list) or any(not isinstance(v,str) or v in reserved or any(v.startswith(r+'=') for r in reserved) for v in job.get('arguments',[])):
            raise ValueError('Reserved or malformed case arguments')
        if job.get('methods','rl,sa,random') not in ('rl','sa','random','rl,sa,random','sa,random'):
            raise ValueError('Use declared full-system method groups')
        job['methods']=job.get('methods','rl,sa,random')
        job['seed']=int(job['seed'])
        seed = Path(job['seed_case']).resolve()
        hashes = {name:hashlib.sha256((seed/name).read_bytes()).hexdigest() for name in
                  ('workload.json','seed_attempts.json','source_identity.json')}
        identifier = sha(hashes)[:20]
        frozen_seed = out/'seed_inputs'/identifier
        for name in hashes:
            target = frozen_seed/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != hashes[name]:
                raise ValueError('Seed snapshot changed')
            shutil.copyfile(seed/name,target)
        job['original_seed_case']=str(seed)
        job['seed_case']=str(frozen_seed)
        job['seed_files']=hashes
        if job.get('shared_case'):
            job['shared_case']=str(Path(job['shared_case']).resolve())
        normalized.append(job)
    for job in normalized:
        if job.get('share_from') and job['share_from'] not in ids:
            raise ValueError('Unknown shared-start dependency')
        visited = {job['id']}
        current = job
        while current.get('share_from'):
            name=current['share_from']
            if name in visited:
                raise ValueError('Cyclic shared-start dependencies')
            visited.add(name)
            current=next(j for j in normalized if j['id']==name)
    files = sources(APP)
    for name in files:
        target = out/'engine'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(APP/name,target)
    protocol = dict(version=1,created_utc=datetime.now(timezone.utc).isoformat(),source_identity=sha(files),source_files=files,
        environment=environment(),jobs_limit=args.jobs,jobs=normalized,
        scope='independent RNG cases with same paid shared starts; no search-label/policy sharing; full physical evaluator unchanged',
        timing='concurrent host wall time and per-case CPU/wall ledgers; not isolated single-case speedup; query-matched quality remains separate',images=0)
    write(out/'cohort_protocol.json',protocol)
    return protocol


def shared_ready(case):
    state = read(Path(case)/'status.json',{})
    return (state.get('status')=='completed' or state.get('method') in ('rl','sa','random')) and all(
        (Path(case)/name).exists() for name in ('common.json','screening.json','shared_start.json'))


def execute(out,protocol):
    if sources(APP) != protocol['source_files'] or environment() != protocol['environment']:
        raise ValueError('Frozen cohort source/environment changed')
    if read(out/'cohort_status.json',{}).get('status') == 'completed':
        print(json.dumps(dict(status='already_completed',additional_PPA=0)),flush=True)
        return 0
    jobs = {job['id']:job for job in protocol['jobs']}
    pending = [name for name,job in jobs.items() if read(Path(job['out'])/'status.json',{}).get('status')!='completed']
    active,finished = {},[]
    failures = []
    interrupted = False
    started = time.perf_counter()
    lease_path = out/'cohort_lease.json'
    lease = dict(status='prepared',workspace=str(ROOT.resolve()),source_identity=protocol['source_identity'],
        jobs=protocol['jobs_limit'],output_root=str(out),allowed_outputs=[job['out'] for job in jobs.values()],
        coordinator_pid=os.getpid(),token=uuid.uuid4().hex)
    global_path = ROOT/'output/.local_rl_diagnostics.lock'
    # Admit one cohort only when no legacy serial experiment or other cohort
    # exists. Shared locks held by each case preserve exclusion on parent exit.
    with ExitStack() as admission_stack:
        # Serialize cohort coordinators across the exclusive-to-shared handoff.
        # If one coordinator dies, its surviving case parents still hold the
        # original shared byte and prevent another coordinator's admission.
        admission_stack.enter_context(lock(ROOT/'output/.local_rl_cohort.lock'))
        with lock(global_path):
            write(lease_path,lease)
        admission_stack.enter_context(shared_lock(global_path))
        lease['status']='active'
        write(lease_path,lease)
        try:
            while pending or active:
                try:
                    if (out/'STOP_REQUESTED').exists():
                        interrupted=True
                    if interrupted or failures:
                        for job,_process,_stream in active.values():
                            (Path(job['out'])/'STOP_REQUESTED').touch()
                    else:
                        slots = [i for i in range(protocol['jobs_limit']) if i not in active]
                        for name in list(pending):
                            if not slots:
                                break
                            job = jobs[name]
                            shared = job.get('shared_case')
                            if job.get('share_from'):
                                donor=Path(jobs[job['share_from']]['out'])
                                if not shared_ready(donor):
                                    continue
                                shared=str(donor)
                            slot=slots.pop(0)
                            case=Path(job['out'])
                            case.mkdir(parents=True,exist_ok=True)
                            arguments=[sys.executable,'-u','-X','utf8',str(APP/'full_system_dse.py'),'--out',str(case)]
                            if (case/'effective_config.json').exists():
                                arguments += ['--resume']
                            else:
                                arguments += ['--seed-case',job['seed_case'],'--seed',str(job['seed']),'--methods',job['methods'],*job.get('arguments',[])]
                                if shared:
                                    arguments += ['--shared-case',shared]
                            arguments += ['--ppa-cohort',str(lease_path),'--ppa-slot',str(slot)]
                            # A fresh case must contain only run.lock when its
                            # own admission check runs. Coordinator logs belong
                            # outside the case, including failed fresh starts.
                            log_path=out/'case_logs'/(job['id']+'.log')
                            log_path.parent.mkdir(parents=True,exist_ok=True)
                            stream=log_path.open('ab')
                            process=subprocess.Popen(arguments,cwd=ROOT,stdout=stream,stderr=stream,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            active[slot]=(job,process,stream)
                            pending.remove(name)
                            print(json.dumps(dict(event='started',job=name,pid=process.pid,slot=slot)),flush=True)
                    for slot,(job,process,stream) in list(active.items()):
                        code=process.poll()
                        if code is None:
                            continue
                        stream.close()
                        active.pop(slot)
                        finished.append(job['id'])
                        if code==130:
                            interrupted=True
                        elif code:
                            failures.append(dict(job=job['id'],return_code=code))
                        print(json.dumps(dict(event='finished',job=job['id'],return_code=code)),flush=True)
                    write(out/'cohort_status.json',dict(status='stopping' if interrupted or failures else 'running',
                        active=[v[0]['id'] for v in active.values()],pending=pending,finished=finished,failures=failures,
                        coordinator_pid=os.getpid(),source_identity=protocol['source_identity']))
                    if (interrupted or failures) and not active:
                        break
                    if pending and not active and not any(not j.get('share_from') or shared_ready(jobs[j['share_from']]['out']) for j in (jobs[n] for n in pending)):
                        raise ValueError('Shared initialization dependency never became available')
                    time.sleep(.1)
                except KeyboardInterrupt:
                    interrupted=True
        except Exception:
            for job,process,stream in active.values():
                (Path(job['out'])/'STOP_REQUESTED').touch()
            for job,process,stream in active.values():
                process.wait()
                stream.close()
            raise
        finally:
            lease['status']='inactive'
            write(lease_path,lease)
    if failures or interrupted:
        status='failed' if failures else 'interrupted'
        write(out/'cohort_status.json',dict(status=status,source_identity=protocol['source_identity'],failures=failures,
            pending=pending,finished=finished,wall_s=time.perf_counter()-started))
        return 1 if failures else 130
    rows=[]
    new_initialization=0
    for job in jobs.values():
        case=Path(job['out'])
        if read(case/'status.json',{}).get('status')!='completed':
            raise ValueError('Cohort case incomplete')
        for script in ('analyze_full_system.py','evaluate_full_policy.py'):
            subprocess.run([sys.executable,'-X','utf8',str(APP/'scripts'/script),'--out',str(case)],cwd=ROOT,check=True)
        summaries=read(case/'summary.json')
        if not (case/'imported_shared.json').exists():
            new_initialization+=read(case/'shared_start.json')['full_ppa']
        for method,summary in summaries.items():
            recommendation=read(case/method/'recommendation.json',{})
            metrics=recommendation.get('metrics') or {}
            rows.append(dict(job=job['id'],method=method,new_search_full_PPA=summary['search_full_ppa'],
                charged_full_PPA=summary['full_ppa'],best_cost=recommendation.get('cost'),
                strict_feasible=recommendation.get('strict_feasible',False),search_elite_cost=summary['best_cost'],
                latency_ms=metrics.get('latency_s',0)*1000 if metrics else None,
                energy_mJ=metrics.get('energy_j',0)*1000 if metrics else None,
                area_mm2=metrics.get('area_mm2'),average_power_W=metrics.get('power_w'),search_s=summary['search_wall_s']))
    result=dict(status='completed',source_identity=protocol['source_identity'],jobs_limit=protocol['jobs_limit'],
        rows=rows,new_initialization_full_PPA=new_initialization,new_search_full_PPA=sum(r['new_search_full_PPA'] for r in rows),
        cohort_wall_s=time.perf_counter()-started,images=0,source_stable=sources(APP)==protocol['source_files'],
        scope=protocol['scope'],timing=protocol['timing'])
    write(out/'cohort_summary.json',result)
    write(out/'cohort_status.json',dict(status='completed',source_identity=protocol['source_identity'],
        finished_utc=datetime.now(timezone.utc).isoformat()))
    lines=['# Concurrent full-PPA cohort','',protocol['timing'],'',
           '|Case|Method|New search PPA|Charged PPA|Strict cost|T ms|E mJ|Area mm²|Average W|Search seconds|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def number(value):return '-' if value is None else f'{value:.6f}'
    for row in rows:
        lines.append(f"|{row['job']}|{row['method']}|{row['new_search_full_PPA']}|{row['charged_full_PPA']}|{number(row['best_cost'])}|{number(row['latency_ms'])}|{number(row['energy_mJ'])}|{number(row['area_mm2'])}|{number(row['average_power_W'])}|{row['search_s']:.3f}|")
    (out/'COHORT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='completed',jobs=len(jobs),new_search_full_PPA=result['new_search_full_PPA'])),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--frozen-cohort',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--models',default='resnet18')
    parser.add_argument('--seeds',default='7,37,47')
    parser.add_argument('--methods',default='rl,sa,random')
    parser.add_argument('--queries',type=int,default=512)
    parser.add_argument('--seconds',type=float,default=7200)
    parser.add_argument('--eval-timeout',type=float,default=600)
    parser.add_argument('--objective',default='edp',choices=('edp','energy','latency','balanced'))
    parser.add_argument('--policy',default='cost',choices=('cost','residual','transition'))
    parser.add_argument('--architecture-screen',default='all',choices=('all','off'))
    parser.add_argument('--screen-seconds',type=float,default=3600)
    parser.add_argument('--seed-refinement',default='wide',choices=('wide','off'))
    parser.add_argument('--seed-queries',type=int,default=48)
    parser.add_argument('--seed-seconds',type=float,default=3600)
    parser.add_argument('--seed-dp-seconds',type=float,default=1800)
    parser.add_argument('--max-fusion-depth',type=int,default=4)
    parser.add_argument('--architecture-refresh-interval',type=int,default=4)
    parser.add_argument('--episode-cost-rise',type=float,default=.25)
    parser.add_argument('--max-area',type=float)
    parser.add_argument('--max-power',type=float)
    parser.add_argument('--max-latency-ms',type=float)
    parser.add_argument('--seed-case',type=Path)
    parser.add_argument('--shared-case',type=Path)
    args=parser.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/'output'):
        raise ValueError('Cohort must remain in project output')
    out.mkdir(parents=True,exist_ok=True)
    if args.frozen_cohort:
        protocol=read(out/'cohort_protocol.json')
        # Failure to acquire ownership must not overwrite the active owner's
        # status or lease (e.g. an accidental second --resume invocation).
        with lock(out/'cohort.lock'):
            try:
                if args.resume:
                    (out/'STOP_REQUESTED').unlink(missing_ok=True)
                return execute(out,protocol)
            except Exception as error:
                write(out/'cohort_status.json',dict(status='failed',error=str(error),source_identity=protocol['source_identity']))
                raise
    if not args.resume and any(out.iterdir()):
        raise ValueError('Fresh cohort output must be empty')
    protocol=prepare(out,args)
    script=out/'engine/scripts/run_parallel_full_suite.py'
    command=[sys.executable,'-u','-X','utf8',str(script),'--out',str(out),'--frozen-cohort']
    if args.resume:command.append('--resume')
    started=time.perf_counter()
    process=subprocess.Popen(command,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        code=process.wait()
    except KeyboardInterrupt:
        (out/'STOP_REQUESTED').touch()
        code=process.wait()
    with (out/'cohort_invocations.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(return_code=code,wall_s=time.perf_counter()-started,
            scope='frozen coordinator+admission+cases+audit; source/input preparation precedes this timer'))+'\n')
    return code


if __name__=='__main__':
    raise SystemExit(main())
