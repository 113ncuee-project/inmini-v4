"""Actual bounded parallel Toy PPA, serial equivalence and cooperative resume."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from run_local_rl_diagnostics import APP,read,write,sources


def events(case,method):
    return [json.loads(line) for line in (case/method/'events.jsonl').read_text(encoding='utf-8').splitlines()]


def compare(left,right):
    for method in ('rl','sa','random'):
        a,b=events(left,method),events(right,method)
        if [r['state_hash'] for r in a] != [r['state_hash'] for r in b]:
            raise AssertionError('Parallel/serial or resumed action sequence differs: '+method)
        la,lb=read(left/method/'checkpoint.json'),read(right/method/'checkpoint.json')
        if la['elite']!=lb['elite'] or la['policy_updates']!=lb['policy_updates']:
            raise AssertionError('Best design/SGD differs: '+method)
        for x,y in zip(a,b):
            if x.get('cost')!=y.get('cost') or x.get('reward')!=y.get('reward'):
                raise AssertionError('Physical outcome or reward differs: '+method)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-case',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():raise ValueError('Fresh validation output required')
    out.mkdir(parents=True)
    started=time.perf_counter()
    runner=APP/'scripts/run_parallel_full_suite.py'
    common=['--models','toy','--seeds','7,37','--methods','rl,sa,random','--queries','32','--seconds','300',
            '--eval-timeout','30','--seed-case',str(args.seed_case.resolve()),'--architecture-screen','off',
            '--seed-refinement','off','--architecture-refresh-interval','2','--policy','residual']
    def run(arguments):
        with (out/'commands.log').open('ab') as stream:
            stream.write((json.dumps(list(map(str,arguments)))+'\n').encode());stream.flush()
            result=subprocess.run([sys.executable,'-u','-X','utf8',*map(str,arguments)],stdout=stream,stderr=stream)
        if result.returncode:raise AssertionError('Actual cohort failed: '+str(result.returncode))
    parallel=out/'parallel'
    run([runner,'--out',parallel,'--jobs','2',*common])
    if read(parallel/'cohort_status.json')['status']!='completed':raise AssertionError('Parallel cohort incomplete')
    # Verify at least two actual worker-evaluation intervals overlap, rather
    # than calling simultaneous parent startup a physical throughput test.
    intervals=[]
    for seed in (7,37):
        for path in (parallel/f'cases/toy/{seed}').glob('*/worker/completed/*.json'):
            row=read(path)
            if row.get('full_evaluations'):
                intervals.append((seed,row['completed_unix']-row['evaluation_elapsed_s'],row['completed_unix']))
    overlap=any(a[0]!=b[0] and min(a[2],b[2])-max(a[1],b[1])>1e-5
                for i,a in enumerate(intervals) for b in intervals[i+1:])
    if not overlap:raise AssertionError('No actual concurrent complete PPA intervals were observed')
    donor=parallel/'cases/toy/7'
    for seed in (7,37):
        case=out/f'serial_{seed}'
        run([APP/'full_system_dse.py','--seed-case',args.seed_case.resolve(),'--shared-case',donor,'--out',case,
             '--methods','rl,sa,random','--queries','32','--seconds','300','--eval-timeout','30','--seed',seed,
             '--cost-value-q','--residual-cost-q','--architecture-screen','off','--seed-refinement','off',
             '--architecture-refresh-interval','2','--episode-cost-rise','.25','--episode-restart-probability','.9'])
        compare(parallel/f'cases/toy/{seed}',case)
    # Same shared inputs for the interrupted cohort prevent spending common
    # initialization again. Checkpoint and model RNG must survive host slots.
    partial=out/'resumed'
    command=[sys.executable,'-u','-X','utf8',str(runner),'--out',str(partial),'--jobs','2',
             *common,'--shared-case',str(donor)]
    with (out/'interrupted.log').open('ab') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=stream)
        deadline=time.monotonic()+180
        requested=False
        while process.poll() is None and time.monotonic()<deadline:
            if any(read(partial/f'cases/toy/{seed}/rl/checkpoint.json',{}).get('steps',0)>=5 for seed in (7,37)):
                duplicate=subprocess.run([sys.executable,'-u','-X','utf8',str(runner),'--out',str(partial),'--resume'],
                    stdout=stream,stderr=stream)
                if duplicate.returncode==0 or read(partial/'cohort_lease.json')['status']!='active' or read(partial/'cohort_status.json')['status']!='running':
                    raise AssertionError('Duplicate invocation changed the active cohort owner')
                (partial/'STOP_REQUESTED').touch();requested=True;break
            time.sleep(.02)
        code=process.wait(timeout=120)
    if not requested or code!=130 or read(partial/'cohort_status.json')['status']!='interrupted':
        raise AssertionError('Cohort cooperative stop failed')
    run([runner,'--out',partial,'--resume'])
    for seed in (7,37):compare(parallel/f'cases/toy/{seed}',partial/f'cases/toy/{seed}')
    before={str(p.relative_to(partial)):p.read_bytes() for p in partial.glob('cases/*/*/*/worker/accounting.json')}
    run([runner,'--out',partial,'--resume'])
    if any((partial/name).read_bytes()!=data for name,data in before.items()):
        raise AssertionError('Completed cohort resume added physical queries')
    if not before:raise AssertionError('No physical ledger checked')
    result=dict(passed=True,source_stable=sources(APP)==read(parallel/'cohort_protocol.json')['source_files'],
        actual_parallel_complete_PPA_intervals=True,observed_worker_completions=len(intervals),
        serial_trajectory_cost_reward_exact=True,resume_trajectory_exact=True,
        duplicate_invocation_preserves_active_owner=True,
        completed_resume_new_PPA=0,elapsed_s=time.perf_counter()-started,images=0,
        scope='actual Toy PPA only; not DNN speedup or policy efficacy',created_utc=datetime.now(timezone.utc).isoformat())
    write(out/'result.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
