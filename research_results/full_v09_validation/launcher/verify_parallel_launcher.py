"""Actual PowerShell CLI preflight: saved protocol, caps, paths and stop."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=Path(__file__).with_name('run_inmin_full_parallel.ps1')
ENGINE=ROOT/'output/inmin_v12_20261104/continuation_study_v1/engine'
OUT=ROOT/'output/inmin_v12_20261104/parallel_launcher_validation'
if OUT.exists():raise ValueError('Fresh launcher validation required')
OUT.mkdir(parents=True)
started=time.perf_counter()
records=[]


def run(arguments,expected=0):
    result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(SCRIPT),*map(str,arguments)],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    records.append(dict(arguments=list(map(str,arguments)),return_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if (expected==0 and result.returncode) or (expected!=0 and not result.returncode):raise AssertionError('Unexpected actual PowerShell return code')
    return json.loads(result.stdout) if expected==0 and result.stdout.strip().startswith('{') else None


fresh=OUT/'not_created'
value=run(['-Out',fresh,'-EngineApp',ENGINE,'-Models','resnet18,resnet50,mobilenet_v2','-Seeds','7,37',
    '-Jobs',3,'-Policy','residual','-Queries',612,'-Seconds',14400,'-EvalTimeout',900,'-MaxArea',200,'-MaxPower',8.5,
    '-MaxLatencyMs',4.2,'-EpisodeCostRise',.25,'-DryRun'])
arguments=value['arguments']
for flag,expected in {'--jobs':'3','--queries':'612','--seconds':'14400','--eval-timeout':'900','--policy':'residual',
    '--max-area':'200','--max-power':'8.5','--max-latency-ms':'4.2','--episode-cost-rise':'0.25'}.items():
    if arguments[arguments.index(flag)+1]!=expected:raise AssertionError('Dropped or culture-dependent flag: '+flag)
if fresh.exists():raise AssertionError('DryRun created an experiment')
fixture=OUT/'saved_protocol_fixture'
runner=fixture/'engine/scripts/run_parallel_full_suite.py';runner.parent.mkdir(parents=True)
shutil.copyfile(ENGINE/'scripts/run_parallel_full_suite.py',runner)
protocol=dict(source_identity='SYNTHETIC_LAUNCHER_FIXTURE_NOT_A_SCIENTIFIC_RESULT',jobs_limit=2,jobs=[dict(id='saved',
    arguments=['--cost-value-q','--residual-cost-q','--gamma','0','--queries','292','--max-area','200'])])
(fixture/'cohort_protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
resumed=run(['-Out',fixture,'-Resume','-Policy','transition','-Jobs',4,'-Queries',7,'-DryRun'])
if resumed['jobs']!=2 or resumed['cases']!=protocol['jobs'] or resumed['arguments'][-1]!='--resume':
    raise AssertionError('Resume preflight overwrote the saved algorithm')
run(['-Out',fixture,'-Stop','-DryRun'])
if (fixture/'STOP_REQUESTED').exists():raise AssertionError('Stop DryRun mutated output')
run(['-Out',fixture,'-Stop'])
if not (fixture/'STOP_REQUESTED').exists():raise AssertionError('Cooperative stop was not written')
run(['-Out',ROOT/'tmp/forbidden_parallel_output','-EngineApp',ENGINE,'-DryRun'],expected=1)
run(['-Out',fixture,'-Resume','-Stop','-DryRun'],expected=1)
result=dict(passed=True,actual_PowerShell_invocations=len(records),new_PPA=0,new_SGD=0,
    checks=['numeric caps and invariant formatting','all3 models','resume reads saved protocol despite conflicting defaults',
            'DryRun never launches Python or creates experiment','cooperative stop path','output containment','incompatible invocation rejected'],
    saved_resume_fixture='synthetic shell-only fixture, not an executed cohort',
    script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),elapsed_s=time.perf_counter()-started)
(OUT/'commands.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
shutil.copyfile(SCRIPT,OUT/SCRIPT.name)
shutil.copyfile(__file__,OUT/Path(__file__).name)
print(json.dumps(result,indent=2))
