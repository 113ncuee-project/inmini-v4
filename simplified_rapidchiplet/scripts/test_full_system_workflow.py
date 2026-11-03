"""Actual CLI stop/resume and full-domain screening integration checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_local_rl_diagnostics import APP, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-case', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError('new validation folder required')
    out.mkdir(parents=True)
    script = APP / 'full_system_dse.py'
    start = time.perf_counter()
    def run(arguments, expected=0):
        result = subprocess.run([sys.executable, '-u', '-X', 'utf8', str(script), *map(str, arguments)], capture_output=True, text=True, encoding='utf-8')
        with (out / 'commands.log').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(list(map(str, arguments))) + '\n' + result.stdout + result.stderr + '\n')
        if result.returncode != expected:
            raise AssertionError(result.stdout + result.stderr)

    def command(target):
        return ['--seed-case', str(Path(args.seed_case).resolve()), '--out', str(target), '--queries', '14', '--seconds', '240',
                '--architecture-screen', 'off', '--methods', 'rl,sa,random', '--eval-timeout', '20', '--shortlist', '24',
                '--horizon', '4', '--warmup', '3', '--replay-updates', '8']
    full, partial = out / 'full', out / 'resumed'
    run(command(full))
    with (out / 'interrupted.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', str(script), *command(partial)], stdout=log, stderr=log)
        deadline = time.monotonic() + 120
        while process.poll() is None and time.monotonic() < deadline:
            if read(partial / 'rl/checkpoint.json', {}).get('steps', 0) >= 3:
                (partial / 'STOP_REQUESTED').touch()
                break
            time.sleep(.05)
        code = process.wait(timeout=60)
        if code != 130 or read(partial / 'status.json')['status'] != 'interrupted':
            raise AssertionError('cooperative stop was not recorded')
    run(['--out', partial, '--resume'])
    for method in ('rl', 'sa', 'random'):
        a, b = read(full / method / 'checkpoint.json'), read(partial / method / 'checkpoint.json')
        def events(root):
            return [json.loads(line) for line in (root / method / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        if [r['state_hash'] for r in events(full)] != [r['state_hash'] for r in events(partial)]:
            raise AssertionError(method + ': resumed action sequence differs')
        if a['elite'] != b['elite'] or a['policy_updates'] != b['policy_updates']:
            raise AssertionError(method + ': resumed best state/updates differ')
        ledger = read(partial / method / 'worker/accounting.json')
        keys = [r['durable_key'] for r in ledger if r.get('full_evaluations')]
        if len(set(keys)) != len(keys):
            raise AssertionError(method + ': duplicate PPA charged')
    before = (partial / 'rl/worker/accounting.json').read_bytes()
    run(['--out', partial, '--resume'])
    if before != (partial / 'rl/worker/accounting.json').read_bytes():
        raise AssertionError('completed resume changed PPA ledger')
    from analyze_full_system import analyze
    for folder in (full, partial):
        audit = analyze(folder)
        if not audit['passed']:
            raise AssertionError(audit)
    # Impossible strict area must produce no fake recommendation.
    constrained = out / 'infeasible'
    run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', constrained, '--queries', '6', '--seconds', '60',
         '--architecture-screen', 'off', '--methods', 'rl', '--max-area', '.01', '--replay-updates', '4'])
    rec = read(constrained / 'rl/recommendation.json')
    if rec['strict_feasible'] or rec['state'] is not None:
        raise AssertionError('impossible constraint fabricated a feasible recommendation')
    screened = out / 'shared_screen'
    run(['--seed-case', str(Path(args.seed_case).resolve()), '--out', screened, '--queries', '58', '--seconds', '120',
         '--screen-seconds', '120', '--methods', 'rl,sa,random', '--shortlist', '24', '--replay-updates', '4'])
    screen = read(screened / 'screening.json')
    if not screen['complete_domain_attempted'] or len(screen['rows']) != 50 or screen['full_ppa'] != 50:
        raise AssertionError('complete 50-layout physical screen not verified')
    total = read(screened / 'summary.json')
    if any(s['common_full_ppa'] != 52 or s['full_ppa'] != 58 for s in total.values()):
        raise AssertionError('shared 50-layout charge differs across methods')
    if not analyze(screened)['passed']:
        raise AssertionError('shared-screen audit failed')
    write(out / 'result.json', dict(passed=True, elapsed_s=time.perf_counter() - start,
        checks=['3 methods exact-query CLI', 'same trajectory after cooperative stop/resume', 'same Q update count',
                'no duplicate durable PPA', 'completed resume no new PPA', 'physical/reward/source audits', 'strict infeasible archive',
                '50 actual architecture PPA', 'identical shared-screen charges for all 3 methods']))
    print(json.dumps(read(out / 'result.json')))


if __name__ == '__main__':
    main()
