"""Actual localhost preference routing and zero-PPA command-preview checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--app',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();app=args.app.resolve();sys.path.insert(0,str(app))
    from ppa_preferences import OBJECTIVES,objective_contract
    child=subprocess.Popen([sys.executable,'-u','-X','utf8',str(app/'full_system_gui.py'),'--port','0','--no-browser'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
    try:
        url=child.stdout.readline().strip();page=urllib.request.urlopen(url).read().decode('utf-8')
        token=json.loads(re.search(r"'X-INMIN-Token':(\"[^\"]+\")",page).group(1))
        for name in OBJECTIVES:
            if f'value="{name}"' not in page:raise ValueError('GUI preference missing: '+name)
        original=json.load(urllib.request.urlopen(url+'/api/status'))
        records=[]
        base=dict(model='resnet18',methods='rl',architecture_screen='all',queries='512',seconds='7200',eval_timeout='600',
            seed='7',max_fusion_depth='4',cost_value_q='residual',max_power='3.5',max_area='120',max_latency_ms='10')
        def request(data,code=200,*,with_token=True):
            headers={'Content-Type':'application/json'}
            if with_token:headers['X-INMIN-Token']=token
            req=urllib.request.Request(url+'/api/preview',json.dumps(data).encode(),headers)
            try:
                response=urllib.request.urlopen(req);actual=response.status;result=json.load(response)
            except urllib.error.HTTPError as error:actual=error.code;result=json.load(error)
            if actual!=code:raise ValueError('Unexpected preview result: '+str(actual)+'/'+str(result))
            return result
        for name in OBJECTIVES:
            result=request(dict(base,objective=name));command=result['command']
            if result['objective']!=objective_contract(name):raise ValueError('Wrong UI objective descriptor')
            if command[command.index('--objective')+1]!=name:raise ValueError('Objective not forwarded to actual CLI')
            if '--residual-cost-q' not in command:raise ValueError('Policy choice lost')
            if result['hard_limits']!=dict(max_power=3.5,max_area=120.,max_latency_ms=10.):raise ValueError('Hard constraints lost')
            if result['new_PPA'] or result['new_SGD']:raise ValueError('Preview performed search')
            records.append(dict(objective=name,weights=result['objective']['weights'],hard_limits=result['hard_limits']))
        alias=request(dict(base,preference='low_power'))
        if alias['objective']['name']!='low_power':raise ValueError('V3 preference API alias lost')
        request(dict(base,objective='edp',preference='small_area'),400)
        request(dict(base,objective='low_power',max_power='nan'),400)
        request(dict(base,objective='low_power',max_area='inf'),400)
        request(dict(base,objective='missing'),400)
        request(dict(base,objective='balanced'),403,with_token=False)
        after=json.load(urllib.request.urlopen(url+'/api/status'))
        if original!=after:raise ValueError('Preview changed saved job/status or started a worker')
        result=dict(passed=True,actual_HTTP=True,objectives=records,preference_alias=True,conflicting_target_rejected=True,
            nonfinite_limits_rejected=True,token_required=True,preview_preserves_saved_job=True,new_PPA=0,new_SGD=0,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Actual GUI HTTP forms/normalization and generated CLI flags, no worker or PPA execution. True CLI search/stop/resume remains a separate required check.')
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
    finally:child.terminate();child.wait(timeout=10)


if __name__=='__main__':main()
