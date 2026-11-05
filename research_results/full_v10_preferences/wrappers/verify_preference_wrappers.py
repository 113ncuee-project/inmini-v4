"""Actual PowerShell preflight for every PPA preset, without PPA or SGD."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/inmin_v17_20261105/preference_readiness_v1/wrappers'


def main():
    if OUT.exists():raise ValueError('Keep earlier wrapper evidence')
    OUT.mkdir();shell=shutil.which('pwsh') or shutil.which('powershell');rows=[]
    if not shell:raise ValueError('PowerShell unavailable')
    for name in ('balanced','low_latency','low_power','small_area','edp','latency','energy'):
        command=[shell,'-NoProfile','-File',str(ROOT/'run_inmin_full_parallel.ps1'),'-Out','output/preference_preflight_'+name,
            '-Objective',name,'-Models','resnet18','-Seeds','7','-Policy','residual','-MaxPower','3.5','-MaxArea','120','-DryRun']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8-sig')
        (OUT/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise ValueError(name+' preflight failed: '+result.stderr)
        value=json.loads(result.stdout);args=value['arguments']
        if args[args.index('--objective')+1]!=name:raise ValueError('PowerShell objective lost')
        if args[args.index('--max-power')+1]!='3.5' or args[args.index('--max-area')+1]!='120':raise ValueError('PowerShell limits lost')
        if (ROOT/('output/preference_preflight_'+name)).exists():raise ValueError('DryRun created a real case')
        rows.append(dict(objective=name,actual_PS_returncode=0,new_PPA=0,new_SGD=0))
    command=[shell,'-NoProfile','-File',str(ROOT/'run_inmin_full_overnight.ps1'),'-Objective','low_power',
        '-Models','resnet18','-Seeds','7','-OutputRoot','output/preference_serial_preflight','-DryRun']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8-sig')
    (OUT/'serial_low_power.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode or '--objective low_power' not in result.stdout:raise ValueError('Serial wrapper preset lost')
    if (ROOT/'output/preference_serial_preflight').exists():raise ValueError('Serial DryRun created output')
    value=dict(passed=True,parallel_presets=rows,serial_low_power_passed=True,
        hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('run_inmin_full_parallel.ps1','run_inmin_full_overnight.ps1')},
        new_PPA=0,new_SGD=0,images=0)
    (OUT/'result.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value))


if __name__=='__main__':main()
