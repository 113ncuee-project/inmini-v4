"""Small Toy-only acceptance: sequential run, controlled interruption/resume, read-only analysis.

Never launches P1-P6 or a pretrained model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from run_local_rl_diagnostics import APP,read,write,sources,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('use a new validation output')
    out.mkdir(parents=True); runner=APP/'scripts/run_local_rl_diagnostics.py'
    before=sources(); started=time.monotonic()
    def invoke(*arguments):
        result=subprocess.run([sys.executable,'-X','utf8',str(runner),*map(str,arguments)],capture_output=True,text=True,encoding='utf-8')
        with (out/'commands.log').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(list(map(str,arguments)))+'\n'+result.stdout+result.stderr+'\n')
        if result.returncode:raise AssertionError(f'CLI exit {result.returncode}: {arguments}: {result.stderr}')
        return result
    full=out/'full'; partial=out/'resumed'
    invoke('run','--profile','smoke','--phase','p0','--out',full)
    args_run=['run','--profile','smoke','--phase','p0','--arms','bottleneck_rl','--out',str(partial)]
    with (out/'interrupted_runner.log').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-X','utf8',str(runner),*args_run],stdout=log,stderr=log)
        checkpoint=partial/'cases/toy/7/methods/bottleneck_rl/checkpoint.json'
        deadline=time.monotonic()+90
        while time.monotonic()<deadline and process.poll() is None:
            if read(checkpoint,{}).get('steps',0)>=2:
                invoke('stop','--out',partial)
                break
            time.sleep(.05)
        else:
            if process.poll() is None:
                (partial/'STOP_REQUESTED').touch()
            raise AssertionError('did not reach intended Toy checkpoint')
        code=process.wait(timeout=60)
        if code==0:raise AssertionError('interrupted run unexpectedly marked completed')
    stopped=read(partial/'suite_status.json')
    if stopped['status']!='interrupted':raise AssertionError(stopped)
    count_before=sum(r.get('full_evaluations',0) for r in read(partial/'cases/toy/7/methods/bottleneck_rl/worker/accounting.json',[]))
    invoke('resume','--out',partial)
    reference=read(full/'cases/toy/7/methods/bottleneck_rl/joint_search.json')
    resumed=read(partial/'cases/toy/7/methods/bottleneck_rl/joint_search.json')
    if reference['best_state']!=resumed['best_state']:
        # Report exact decision hash, excluding evaluation_report/timing metadata.
        def decisions(state):return {k:v for k,v in state.items() if k not in ('evaluation_report','remaining_budget')}
        if decisions(reference['best_state'])!=decisions(resumed['best_state']):raise AssertionError('resume best mapping differs')
    if [r['action_plan']['id'] for r in reference['trace']] != [r['action_plan']['id'] for r in resumed['trace']]:
        raise AssertionError('resume action sequence differs')
    ledger_file=partial/'cases/toy/7/methods/bottleneck_rl/worker/accounting.json'
    rows=read(ledger_file); full_rows=[r for r in rows if r.get('full_evaluations')]
    if len({r['durable_key'] for r in full_rows})!=len(full_rows):raise AssertionError('recomputed completed PPA')
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in partial.rglob('*') if p.is_file() and p.name in ('checkpoint.json','accounting.json')}
    invoke('status','--out',partial);invoke('summarize','--out',partial);invoke('audit','--out',partial)
    invoke('replay','--out',partial);invoke('resume','--out',partial);invoke('export','--out',partial)
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in partial.rglob('*') if p.is_file() and p.name in ('checkpoint.json','accounting.json')}
    if protected!=after:raise AssertionError('read-only analysis/completed resume changed oracle/checkpoint evidence')
    result=dict(passed=True,source_stable=before==sources(),elapsed_s=time.monotonic()-started,
        tests=['four Toy methods + common Seed','stop after >=2 steps','resume same action sequence and best mapping',
            'no duplicate completed durable PPA','status/analysis/audit/replay/completed-resume no PPA/checkpoint writes','export CRC'],
        interrupted_search_full_ppa=count_before,final_search_full_ppa=resumed['summary']['search_full_ppa'],
        best_architecture=resumed['best_state'].get('architecture_candidate_id'),
        best_decisions_sha256=sha({k:v for k,v in resumed['best_state'].items() if k not in ('evaluation_report','remaining_budget')}),source_identity=sha(before))
    write(out/'result.json',result);print(json.dumps(result,indent=2))
    if not result['source_stable']:raise AssertionError('sources changed during validation')

if __name__=='__main__':main()
