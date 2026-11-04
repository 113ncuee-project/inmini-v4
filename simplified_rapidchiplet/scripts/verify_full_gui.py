"""Local HTTP checks with an optional real eight-query Toy job; no browser."""
import json
import argparse
import re
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-case', help='Optional Toy archive to run a real eight-query GUI job')
    args = p.parse_args()
    app = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-u', '-X', 'utf8', str(app / 'full_system_gui.py'), '--port', '0', '--no-browser']
    if args.seed_case:
        command += ['--seed-case', str(Path(args.seed_case).resolve())]
    child = subprocess.Popen(command,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    try:
        url = child.stdout.readline().strip()
        if not url.startswith('http://127.0.0.1:'):
            raise AssertionError('local GUI failed to bind: ' + child.stderr.read())
        html = urllib.request.urlopen(url).read().decode('utf-8')
        for field in ('queries', 'max_fusion_depth', 'architecture_screen', 'max_power', 'max_area', 'max_latency_ms',
                      'boundary_features', 'resource_projection', 'cost_value_q', 'architecture_scaling',
                      'episode_cost_rise','episode_restart_probability'):
            if 'name="' + field + '"' not in html:
                raise AssertionError('missing UI field: ' + field)
        for field in ('seed_refinement', 'seed_queries', 'seed_seconds'):
            if 'name="' + field + '"' not in html:
                raise AssertionError('missing initialization field: ' + field)
        status = json.load(urllib.request.urlopen(url + '/api/status'))
        if not {'out', 'status', 'summary', 'recommendations', 'log'}.issubset(status):
            raise AssertionError('status API missing fields')
        request = urllib.request.Request(url + '/api/start', b'{}', {'Content-Type': 'application/json'})
        try:
            urllib.request.urlopen(request)
            raise AssertionError('missing token was accepted')
        except urllib.error.HTTPError as error:
            if error.code != 403:
                raise
        full = 0
        job_out = None
        if args.seed_case:
            token = json.loads(re.search(r"'X-INMIN-Token':(\"[^\"]+\")", html).group(1))
            data = dict(model='resnet18', objective='edp', methods='rl', architecture_screen='off', queries='8', seconds='90',
                        eval_timeout='20', seed='7', max_fusion_depth='4', cost_value_q='on', architecture_scaling='proportional',
                        episode_cost_rise='.25',episode_restart_probability='.9')
            request = urllib.request.Request(url + '/api/start', json.dumps(data).encode(),
                {'Content-Type': 'application/json', 'X-INMIN-Token': token})
            job = json.load(urllib.request.urlopen(request))
            job_out = Path(job['out'])
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                current = json.load(urllib.request.urlopen(url + '/api/status'))
                if current['status'].get('status') in ('completed', 'failed'):
                    if current['status']['status'] == 'failed':
                        raise AssertionError(current['log'])
                    full = current['summary']['rl']['full_ppa']
                    effective=json.loads((job_out/'effective_config.json').read_text(encoding='utf-8'))
                    if effective.get('episode_cost_rise')!=.25 or effective.get('episode_restart_probability')!=.9:
                        raise AssertionError('GUI did not forward measured episode controls')
                    if not current['recommendations']['rl']['strict_feasible'] or not current['recommendations']['rl']['metrics']:
                        raise AssertionError('GUI failed to expose strict-feasible recommendation')
                    break
                time.sleep(.25)
            if full != 8:
                raise AssertionError('GUI did not complete eight actual full PPA')
        print(json.dumps(dict(passed=True, fields=15, cost_value_q=True, episode_controls=True, status_api=True, strict_recommendations=True, token_required=True, new_ppa=full, out=str(job_out))))
    finally:
        child.terminate()
        child.wait(timeout=10)


if __name__ == '__main__':
    main()
