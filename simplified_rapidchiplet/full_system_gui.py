"""Local full-system control panel. Standard library only; same frozen CLI engine."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import html
import subprocess
import sys
import time
import webbrowser

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP / 'scripts'))
from run_local_rl_diagnostics import read, write

HTML = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>INMIN Full System</title>
<style>body{font:16px system-ui;max-width:1050px;margin:35px auto;background:#101720;color:#edf2f9}input,select,button{padding:8px;margin:5px;background:#233245;color:inherit;border:1px solid #526880;border-radius:5px}label{display:inline-block}pre{white-space:pre-wrap;background:#192535;padding:16px}a{color:#8fc8ff}table{border-collapse:collapse;width:100%}td,th{border:1px solid #526880;padding:7px}</style>
<h1>INMIN 完整設計空間搜尋</h1><p>沿用 v3 硬體：1–16 Compute／50 矩形、16 PE／Compute、共享 L2、Batch=1。完整 Mapping、分組、Fusion／Tile／Cache、Dataflow 與 DRAM。所有結果須完成完整 PPA。</p>
CUSTOM_WORKLOAD
<form id="form"><label>模型 <select name="model"><option>resnet18</option><option>resnet50</option><option>mobilenet_v2</option></select></label>
<label>目標 <select name="objective"><option>edp</option><option>balanced</option><option>latency</option><option>energy</option></select></label>
<label>方法 <select name="methods"><option value="rl,sa,random">RL + SA + Random</option><option value="rl,sa">RL + SA</option><option value="rl">RL</option></select></label><br>
<label>每方法完整 PPA 上限 <input name="queries" type="number" min="8" value="512"></label>
<label>每方法搜尋秒數 <input name="seconds" type="number" min="1" value="3600"></label>
<label>單次評估秒數 <input name="eval_timeout" type="number" min="1" value="300"></label><br>
<label>RNG Seed <input name="seed" type="number" value="7"></label>
<label>Fusion 最大深度 <input name="max_fusion_depth" type="number" min="2" value="4"></label>
<label>架構共同篩選 <select name="architecture_screen"><option value="all">全部 50 種</option><option value="off">僅在搜尋時切換架構</option></select></label><br>
<label>DAG 資料邊界特徵 <select name="boundary_features"><option value="on">開啟</option><option value="off">關閉（消融）</option></select></label>
<label>整組 PE 配額投影 <select name="resource_projection"><option value="on">開啟</option><option value="off">關閉（消融）</option></select></label><br>
<label>RL 成本表示 <select name="cost_value_q"><option value="on">起點校正多步Q</option><option value="residual">固定先驗＋線上修正Q（實驗）</option><option value="off">原多步Q（對照）</option></select></label>
<label>架構切換配額 <select name="architecture_scaling"><option value="proportional">按原使用比例配置</option><option value="legacy">舊版全部PE配置</option></select></label><br>
<label>Episode 容許 Cost 惡化比例（空白停用） <input name="episode_cost_rise" type="number" step="0.05" min="0" value="0.25"></label>
<label>惡化後重啟機率 <input name="episode_restart_probability" type="number" step="0.05" min="0" max="1" value="0.9"></label><br>
<label>架構重新探索間隔（0停用） <input name="architecture_refresh_interval" type="number" min="0" value="0"></label><br>
<label>起點強化 <select name="seed_refinement"><option value="wide">完整長度 DP + inner refinements</option><option value="off">略過額外強化</option></select></label>
<label>額外起點 PPA 上限 <input name="seed_queries" type="number" min="0" value="48"></label>
<label>起點評估秒數 <input name="seed_seconds" type="number" min="1" value="1200"></label><br>
<label>Power 上限 W（可空） <input name="max_power" type="number" step="any" min="0"></label>
<label>Area 上限 mm²（可空） <input name="max_area" type="number" step="any" min="0"></label>
<label>Latency 上限 ms（可空） <input name="max_latency_ms" type="number" step="any" min="0"></label><br>
<button>開始新實驗</button><button type="button" onclick="act('resume')">續跑上次實驗</button><button type="button" onclick="act('stop')">儲存後停止</button></form>
<p>可鎖定螢幕，請維持電腦開機。數值屬解析 PPA。RL／SA 勝負以相同查詢與時間下的結果為準。</p>
<h2>執行狀態</h2><pre id="status">載入中</pre><div id="results"></div><h2>最新記錄</h2><pre id="log"></pre>
<script>
async function act(action,params={}){let r=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json','X-INMIN-Token':TOKEN},body:JSON.stringify(params)});let d=await r.json();if(!r.ok)alert(d.error);await refresh()}
document.querySelector('#form').onsubmit=e=>{e.preventDefault();act('start',Object.fromEntries(new FormData(e.target)))};
async function refresh(){try{
let d=await(await fetch('/api/status')).json();document.querySelector('#status').textContent=JSON.stringify(d.status,null,2)+'\\n'+(d.out||'');document.querySelector('#log').textContent=d.log||'';
let box=document.querySelector('#results');box.replaceChildren();
for(let [k,v] of Object.entries(d.summary||{})){
let p=document.createElement('p'),r=(d.recommendations||{})[k];
if(!r||!r.strict_feasible){p.textContent=k+': 完整 PPA '+v.full_ppa+'；目前沒有符合所有硬限制的推薦設計。'}
else{let m=r.metrics,shared=d.shared_start||{};p.textContent=k+': 完整 PPA '+v.full_ppa+' | 總Cost改善 '+(100*(1-r.cost/v.seed_cost)).toFixed(3)+'% | 起點後再改善 '+(100*(1-r.cost/(shared.cost||v.seed_cost))).toFixed(3)+'% | Latency改善 '+(100*(1-m.latency_s/v.initial.latency_s)).toFixed(3)+'% | Energy改善 '+(100*(1-m.energy_j/v.initial.energy_j)).toFixed(3)+'% | Area '+m.area_mm2.toFixed(3)+' mm² | 平均Power '+m.power_w.toFixed(3)+' W | '+r.architecture+' | Fusion深度 '+r.fusion_depth;}
box.append(p)}}catch(e){document.querySelector('#status').textContent=String(e)}}
setInterval(refresh,4000);refresh();</script></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--no-browser', action='store_true')
    p.add_argument('--seed-case', help='Optional custom archived workload/seed directory; overrides model dropdown')
    args = p.parse_args()
    custom = Path(args.seed_case).resolve() if args.seed_case else None
    if custom and any(not (custom / name).is_file() for name in ('workload.json', 'seed_attempts.json', 'source_identity.json')):
        raise ValueError('custom seed archive is incomplete')
    token = secrets.token_urlsafe(24)
    pool = ThreadPoolExecutor(max_workers=1)
    registry = ROOT / 'output/full_system_gui/last_job.json'
    job = read(registry, {})
    process = None
    task = None
    def launch(command, out):
        nonlocal process
        out.mkdir(exist_ok=True, parents=True)
        import os
        awake = None
        if os.name == 'nt':
            import ctypes
            awake = ctypes.windll.kernel32.SetThreadExecutionState
            awake(0x80000001)
        try:
            logs = out.parent / 'launcher_logs'
            logs.mkdir(exist_ok=True)
            with (logs / (out.name + '.log')).open('ab') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=stream,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            code = process.wait()
            if code == 0:
                subprocess.run([sys.executable, '-X', 'utf8', str(APP / 'scripts/analyze_full_system.py'), '--out', str(out)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif read(out / 'status.json', {}).get('status') != 'interrupted':
                write(out / 'status.json', dict(status='failed', return_code=code,
                    error='runner exited before completion; inspect launcher_logs and runner.log (another active experiment may hold the project lock)'))
        finally:
            if awake:
                awake(0x80000000)

    def recommendations(out, summaries):
        result = {}
        for method in summaries:
            rec = read(out / method / 'recommendation.json', {})
            state = rec.get('state') or {}
            result[method] = dict(strict_feasible=bool(rec.get('strict_feasible')), cost=rec.get('cost'),
                architecture=state.get('architecture_candidate_id'),
                fusion_depth=max((len(s['layer_ids']) for s in state.get('mapping', {}).get('fusion_stacks', ())), default=0),
                metrics={k: rec.get('metrics', {}).get(k) for k in ('latency_s', 'energy_j', 'area_mm2', 'power_w')} if rec.get('metrics') else None)
        return result

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                label = '<p>目前使用自訂工作量：' + html.escape(read(custom / 'workload.json')['name']) + '。模型下拉選單不會變更此輸入。</p>' if custom else ''
                body = HTML.replace('TOKEN', json.dumps(token)).replace('CUSTOM_WORKLOAD', label).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/api/status':
                out = Path(job['out']) if job else None
                log = ''
                if out and (out / 'runner.log').exists():
                    with (out / 'runner.log').open('rb') as stream:
                        stream.seek(max(0, stream.seek(0, 2) - 10000))
                        log = stream.read().decode('utf-8', errors='replace')
                summaries = read(out / 'summary.json', {}) if out else {}
                self.respond(dict(out=str(out) if out else None, status=read(out / 'status.json', {}) if out else {},
                    summary=summaries, recommendations=recommendations(out, summaries) if out else {},
                    shared_start={k:v for k,v in read(out / 'shared_start.json', {}).items() if k in ('cost', 'full_ppa', 'wall_s', 'gain_before_rl_pct', 'initialization')} if out else {}, log=log))
            else:
                self.respond(dict(error='not found'), 404)

        def do_POST(self):
            nonlocal job, task
            if self.headers.get('X-INMIN-Token') != token:
                return self.respond(dict(error='invalid local request token'), 403)
            try:
                data = json.loads(self.rfile.read(min(int(self.headers.get('Content-Length', '0')), 16384)))
                active = task is not None and not task.done()
                if self.path == '/api/stop':
                    if job:
                        (Path(job['out']) / 'STOP_REQUESTED').touch()
                    return self.respond(dict(status='stop requested; current bounded oracle completes first'))
                if active:
                    raise ValueError('An experiment is already running; stop it before starting another.')
                if self.path == '/api/resume':
                    if not job:
                        raise ValueError('No previous GUI job')
                    out = Path(job['out']).resolve()
                    if not out.is_relative_to((ROOT / 'output/full_system_gui').resolve()):
                        raise ValueError('job output escapes GUI directory')
                    command = [sys.executable, '-u', '-X', 'utf8', str(APP / 'full_system_dse.py'), '--out', str(out), '--resume']
                elif self.path == '/api/start':
                    if data['model'] not in ('resnet18', 'resnet50', 'mobilenet_v2') or data['objective'] not in ('balanced', 'edp', 'latency', 'energy'):
                        raise ValueError('unknown model/objective')
                    if data['methods'] not in ('rl,sa,random', 'rl,sa', 'rl') or data['architecture_screen'] not in ('all', 'off'):
                        raise ValueError('unknown methods/screen')
                    numeric = {k: int(data[k]) for k in ('queries', 'seconds', 'eval_timeout', 'seed', 'max_fusion_depth')}
                    numeric.update(seed_queries=int(data.get('seed_queries', 48)), seed_seconds=int(data.get('seed_seconds', 1200)))
                    if numeric['queries'] < 8 or min(numeric['seconds'], numeric['eval_timeout']) <= 0 or numeric['max_fusion_depth'] < 2:
                        raise ValueError('invalid positive budget/depth')
                    if numeric['seed_queries'] < 0 or numeric['seed_seconds'] <= 0:
                        raise ValueError('invalid initialization query/time ceiling')
                    out = ROOT / 'output/full_system_gui' / (time.strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))
                    command = [sys.executable, '-u', '-X', 'utf8', str(APP / 'full_system_dse.py'), '--seed-case',
                        str(custom or ROOT / 'inputs/replay_v04' / data['model']), '--out', str(out)]
                    for key in ('methods', 'objective', 'architecture_screen'):
                        command += ['--' + key.replace('_', '-'), data[key]]
                    refinement = data.get('seed_refinement', 'wide')
                    if refinement not in ('wide', 'off'):
                        raise ValueError('unknown seed refinement')
                    command += ['--seed-refinement', refinement]
                    cost_value_q = data.get('cost_value_q', 'off')
                    scaling = data.get('architecture_scaling', 'proportional')
                    if cost_value_q not in ('on', 'off','residual') or scaling not in ('legacy', 'proportional'):
                        raise ValueError('unknown RL cost representation/architecture quota setting')
                    if cost_value_q in ('on','residual'):
                        command += ['--cost-value-q']
                    if cost_value_q == 'residual':
                        command += ['--residual-cost-q']
                    command += ['--architecture-scaling', scaling]
                    refresh_interval=int(data.get('architecture_refresh_interval','0'))
                    if refresh_interval < 0:
                        raise ValueError('architecture refresh interval must be nonnegative')
                    command += ['--architecture-refresh-interval',str(refresh_interval)]
                    restart_probability=float(data.get('episode_restart_probability','.9'))
                    if not math.isfinite(restart_probability) or not 0 <= restart_probability <= 1:
                        raise ValueError('restart probability must be in [0,1]')
                    command += ['--episode-restart-probability',str(restart_probability)]
                    if data.get('episode_cost_rise'):
                        rise=float(data['episode_cost_rise'])
                        if not math.isfinite(rise) or rise < 0:
                            raise ValueError('cost rise must be a nonnegative finite fraction')
                        command += ['--episode-cost-rise',str(rise)]
                    for key in ('boundary_features', 'resource_projection'):
                        option = data.get(key, 'on')
                        if option not in ('on', 'off'):
                            raise ValueError('unknown feature/projection setting')
                        if option == 'off':
                            command += ['--no-' + key.replace('_', '-')]
                    for key, value in numeric.items():
                        command += ['--' + key.replace('_', '-'), str(value)]
                    for key in ('max_power', 'max_area', 'max_latency_ms'):
                        if data.get(key):
                            value = float(data[key])
                            if value <= 0:
                                raise ValueError('limits must be positive')
                            command += ['--' + key.replace('_', '-'), str(value)]
                    job = dict(out=str(out), command=command)
                    write(registry, job)
                else:
                    return self.respond(dict(error='not found'), 404)
                task = pool.submit(launch, command, out)
                self.respond(dict(status='starting', out=str(out)))
            except (ValueError, KeyError, TypeError) as error:
                self.respond(dict(error=str(error)), 400)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}'
    print(url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
