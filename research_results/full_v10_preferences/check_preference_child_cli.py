"""Small actual CHILD-CLI correctness cases; not parent/cohort DSE timing.

This fixture invokes the engine's internal frozen child with a tiny Toy input,
as a unit integration check. It does not run any DNN or claim to exercise the
outer parent's global admission. All actual Toy PPA/SGD stays in case ledgers.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'inputs/replay_v04').is_dir() and (p/'.venv').is_dir())
STUDY=ROOT/'output/inmin_v17_20261105/preference_readiness_v1'
ENGINE=STUDY/'engine'
REFERENCE=ROOT/'output/inmin_v14_20261105/readiness_study_v1/parallel_workflow/serial_7'
sys.path[:0]=[str(ENGINE),str(ENGINE/'scripts')]
from run_local_rl_diagnostics import sources,sha,environment
from analyze_full_system import analyze
from ppa_preferences import PPA_WEIGHTS,run_objective_context
from types import SimpleNamespace


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def prepare(name,preference,*,methods='rl,sa,random',queries=12):
    out=STUDY/'child_cli'/name
    if out.exists():raise ValueError('Preserve prior child CLI attempts')
    out.mkdir(parents=True)
    files=sources(ENGINE)
    for n in files:
        t=out/'source_snapshot'/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ENGINE/n,t)
    hashes={}
    for n in ('workload.json','seed_attempts.json','source_identity.json'):
        t=out/'seed_input_snapshot'/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REFERENCE/'seed_input_snapshot'/n,t)
        hashes[n]=hashlib.sha256(t.read_bytes()).hexdigest()
    config=read(REFERENCE/'effective_config.json')
    config.update(out=str(out),seed_case=str(out/'seed_input_snapshot'),shared_case=None,methods=methods,queries=queries,
        seconds=240.,eval_timeout=30.,objective=preference,preference=preference,warmup=2,shortlist=24,replay_updates=8,
        seed_refinement='off',seed_queries=0,architecture_screen='off',architecture_refresh_interval=2)
    write(out/'effective_config.json',config);write(out/'source_snapshot_manifest.json',dict(identity=sha(files),files=files))
    write(out/'seed_input_manifest.json',dict(original_case=str(REFERENCE/'seed_input_snapshot'),files=hashes))
    write(out/'environment.json',environment())
    command=[sys.executable,'-u','-X','utf8',str(out/'source_snapshot/full_system_dse.py'),'--frozen-child']
    for key,value in config.items():
        if value is None:continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
        else:command.extend((flag,str(value)))
    return out,command


def run(out,command,expected=0):
    p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    with (STUDY/'child_cli_commands.log').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(command)+'\n'+p.stdout+p.stderr+'\n')
    if p.returncode!=expected:raise AssertionError('Child CLI failed: '+p.stderr[-4000:])


def verify(out):
    result=analyze(out)
    if not result['passed']:raise AssertionError(result)
    cfg=read(out/'effective_config.json');common=read(out/'common.json')
    context=run_objective_context(SimpleNamespace(**cfg),common['reference'])
    if read(out/'objective.json')!=context:raise ValueError('Fixed-reference context not persisted')
    for method in cfg['methods'].split(','):
        for name in ('checkpoint.json','recommendation.json','summary.json'):
            if read(out/method/name)['objective_context']!=context:raise ValueError('Preference not propagated: '+method+'/'+name)
    return dict(preference=cfg['objective'],methods=cfg['methods'],audit=result,
        query_counts={method:read(out/method/'summary.json')['full_ppa'] for method in cfg['methods'].split(',')},
        SGD=read(out/'rl/summary.json')['policy_updates'])


def main():
    if (STUDY/'child_cli_result.json').exists():raise ValueError('Keep original child-CLI evidence')
    started=time.perf_counter();rows=[]
    for name in PPA_WEIGHTS:
        case,command=prepare(name,name);run(case,command);rows.append(verify(case))
    full,fullcmd=prepare('power_resume_reference','low_power',methods='rl',queries=20)
    partial,partialcmd=prepare('power_resumed','low_power',methods='rl',queries=20)
    run(full,fullcmd)
    with (STUDY/'child_cli_interrupted.log').open('wb') as log:
        process=subprocess.Popen(partialcmd,cwd=ROOT,stdout=log,stderr=log)
        deadline=time.monotonic()+180
        while process.poll() is None and time.monotonic()<deadline:
            path=partial/'rl/checkpoint.json'
            if path.exists() and read(path).get('steps',0)>=3:
                (partial/'STOP_REQUESTED').touch();break
            time.sleep(.01)
        if process.wait(timeout=60)!=130:raise ValueError('Cooperative child stop not recorded')
    (partial/'STOP_REQUESTED').unlink()
    run(partial,partialcmd)
    a=[json.loads(x) for x in (full/'rl/events.jsonl').read_text().splitlines()]
    b=[json.loads(x) for x in (partial/'rl/events.jsonl').read_text().splitlines()]
    if [(x['state_hash'],x.get('cost'),x.get('reward')) for x in a]!=[(x['state_hash'],x.get('cost'),x.get('reward')) for x in b]:
        raise ValueError('Preference child stop/resume changed exact actions/cost/reward')
    if read(full/'rl/checkpoint.json')['policy_updates']!=read(partial/'rl/checkpoint.json')['policy_updates']:
        raise ValueError('Preference resume SGD differs')
    rows.extend((verify(full),verify(partial)))
    ledger=(partial/'rl/worker/accounting.json').read_bytes()
    wrong=partialcmd.copy()
    wrong[wrong.index('--objective')+1]='small_area';wrong[wrong.index('--preference')+1]='small_area'
    run(partial,wrong,expected=1)
    if (partial/'rl/worker/accounting.json').read_bytes()!=ledger:raise ValueError('Wrong preference consumed PPA before rejection')
    result=dict(passed=True,actual_child_CLI=True,elapsed_s=time.perf_counter()-started,rows=rows,
        same_preference_stop_resume_exact=True,different_preference_resume_rejected_before_PPA=True,
        source_identity=sha(sources(ENGINE)),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        finished_utc=datetime.now(timezone.utc).isoformat(),new_DNN_PPA=0,images=0,
        scope='Tiny correctness fixture of actual internal frozen-child CLI with all four PPA preferences and real resource-calendar PPA/reward/learner. Actual Toy PPA and SGD are counted in each case, not zero. Outer-parent admission and full GUI job/cohort workflow are separate; existing b771 parent checks cannot be labeled rerun on this source. Same-host DNN confirmation runs concurrently; no timing/quality claim.')
    write(STUDY/'child_cli_result.json',result)
    print(json.dumps(dict(passed=True,cases=len(rows),same_preference_stop_resume_exact=True,
        different_preference_resume_rejected_before_PPA=True,elapsed_s=result['elapsed_s'],new_DNN_PPA=0)))


if __name__=='__main__':main()
