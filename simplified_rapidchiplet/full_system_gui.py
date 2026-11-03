"""Local full-system control panel. Standard library only; same frozen CLI engine."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
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
<label>Power 上限 W（可空） <input name="max_power" type="number" step="any" min="0"></label>
<label>Area 上限 mm²（可空） <input name="max_area" type="number" step="any" min="0"></label>
<label>Latency 上限 ms（可空） <input name="max_latency_ms" type="number" step="any" min="0"></label><br>
<button>開始新實驗</button><button type="button" onclick="act('resume')">續跑上次實驗</button><button type="button" onclick="act('stop')">儲存後停止</button></form>
<p>可鎖定螢幕，請維持電腦開機。數值屬解析 PPA。RL／SA 勝負以相同查詢與時間下的結果為準。</p>
<h2>執行狀態</h2><pre id="status">載入中</pre><div id="results"></div><h2>最新記錄</h2><pre id="log"></pre>
<script>
async function act(action,params={}){let r=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json','X-INMIN-Token':TOKEN},body:JSON.stringify(params)});let d=await r.json();if(!r.ok)alert(d.error);await refresh()}
document.querySelector('#form').onsubmit=e=>{e.preventDefault();act('start',Object.fromEntries(new FormData(e.target)))};
async function refresh(){try{let d=await(await fetch('/api/status')).json();document.querySelector('#status').textContent=JSON.stringify(d.status,null,2)+'\\n'+(d.out||'');document.querySelector('#log').textContent=d.log||'';let box=document.querySelector('#results');box.replaceChildren();for(let [k,v] of Object.entries(d.summary||{})){let p=document.createElement('p');p.textContent=k+': Full PPA '+v.full_ppa+' | Cost改善 '+v.gain_vs_seed_pct.toFixed(3)+'% | Latency改善 '+v.latency_improvement_pct.toFixed(3)+'% | Energy改善 '+v.energy_improvement_pct.toFixed(3)+'% | '+v.best_architecture+' | Fusion深度 '+v.best_fusion_depth;box.append(p)}}catch(e){document.querySelector('#status').textContent=String(e)}}
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
        finally:
            if awake:
                awake(0x80000000)

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
                self.respond(dict(out=str(out) if out else None, status=read(out / 'status.json', {}) if out else {},
                    summary=read(out / 'summary.json', {}) if out else {}, log=log))
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
                    if numeric['queries'] < 8 or min(numeric['seconds'], numeric['eval_timeout']) <= 0 or numeric['max_fusion_depth'] < 2:
                        raise ValueError('invalid positive budget/depth')
                    out = ROOT / 'output/full_system_gui' / (time.strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))
                    command = [sys.executable, '-u', '-X', 'utf8', str(APP / 'full_system_dse.py'), '--seed-case',
                        str(custom or ROOT / 'inputs/replay_v04' / data['model']), '--out', str(out)]
                    for key in ('methods', 'objective', 'architecture_screen'):
                        command += ['--' + key.replace('_', '-'), data[key]]
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
