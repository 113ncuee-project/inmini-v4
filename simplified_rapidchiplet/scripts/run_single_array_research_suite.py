"""Run the authorized representative research suite against an immutable copy.

Every model uses Seed7 and balanced PPA with no invented user hard limits.
The four algorithms share completed seeds, reference, query and wall ceilings.
An additional fusion-off bottleneck-RL run provides the paired search ablation.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--queries', type=int, default=128)
    parser.add_argument('--seconds', type=float, default=7200.)
    parser.add_argument('--eval-timeout', type=float, default=1800.)
    parser.add_argument('--toy-queries', type=int, default=64)
    parser.add_argument('--toy-seconds', type=float, default=600.)
    parser.add_argument('--seed-seconds', type=float, default=7200.)
    parser.add_argument('--seed-queries', type=int, default=16)
    parser.add_argument('--seed-dp-seconds', type=float, default=120.)
    parser.add_argument('--models', default='resnet18,resnet50,mobilenet_v2')
    parser.add_argument('--skip-fusion-off', action='store_true')
    args = parser.parse_args()
    snapshot, output = Path(args.snapshot).resolve(), Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(snapshot))
    from simple_rapidchiplet.hybrid.auto_runtime import atomic_json
    from single_array_dse import exclusive_run
    manifest = json.loads((snapshot / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    def verify():
        for relative, expected in manifest['files'].items():
            path = (snapshot / relative).resolve()
            if not path.is_relative_to(snapshot) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('frozen source manifest mismatch: ' + relative)
    with exclusive_run(output):
        verify()
        status_path = output / 'suite_status.json'
        saved = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        configuration = {key: value for key, value in vars(args).items() if key not in ('snapshot', 'out')}
        if saved and saved['configuration'] != configuration:
            raise ValueError('suite configuration differs; use a new output directory')
        rows = saved.get('cases', [])
        state = {'status': 'running', 'owner_pid': os.getpid(), 'snapshot': str(snapshot),
                 'snapshot_identity': manifest['identity'], 'configuration': configuration, 'cases': rows}
        power = None
        if os.name == 'nt':
            power = ctypes.windll.kernel32.SetThreadExecutionState
            power.argtypes, power.restype = [ctypes.c_uint], ctypes.c_uint
            state['temporary_system_awake_request'] = bool(power(0x80000001))
        cases = [('toy', 'fusion_on')]
        for model in args.models.split(','):
            cases.append((model, 'fusion_on'))
            if not args.skip_fusion_off:
                cases.append((model, 'fusion_off'))
        try:
            for model, mode in cases:
                verify()
                key = model + '/' + mode
                previous = next((row for row in rows if row['case'] == key), None)
                case_output = output / model / mode
                run_state = case_output / 'run_state.json'
                if previous and previous['status'] == 'completed' and run_state.exists() and json.loads(run_state.read_text())['source_stable']:
                    continue
                queries = args.toy_queries if model == 'toy' else args.queries
                seconds = args.toy_seconds if model == 'toy' else args.seconds
                arms = 'random,greedy,atomic_rl,bottleneck_rl' if mode == 'fusion_on' else 'bottleneck_rl'
                command = [sys.executable, '-X', 'utf8', str(snapshot / 'single_array_dse.py'),
                    '--model', model, '--engine', 'v03', '--seed-seconds', str(min(args.seed_seconds,120.) if model=='toy' else args.seed_seconds),
                    '--seed-queries', str(args.seed_queries), '--seed-dp-seconds', str(args.seed_dp_seconds), '--shortlist', '64', '--seed', '7', '--preference', 'balanced', '--seed-method', 'tangram_style_v02',
                    '--queries', str(queries), '--seconds', str(seconds), '--eval-timeout', str(args.eval_timeout),
                    '--fusion', 'on' if mode == 'fusion_on' else 'off', '--arms', arms, '--out', str(case_output)]
                if (case_output / 'source_identity.json').exists():
                    command.append('--resume')
                row = {'case': key, 'model': model, 'mode': mode, 'status': 'running', 'command': command,
                       'started_unix': time.time(), 'query_ceiling_per_arm': queries, 'wall_ceiling_per_arm_s': seconds}
                if previous:
                    rows.remove(previous)
                rows.append(row)
                with (output / (model + '_' + mode + '.log')).open('ab') as log:
                    process = subprocess.Popen(command, cwd=snapshot, stdout=log, stderr=log,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    row['pid'] = process.pid
                    atomic_json(status_path, state)
                    code = process.wait()
                row.update(status='completed' if code == 0 else 'failed', return_code=code,
                           elapsed_s=time.time() - row['started_unix'])
                if (case_output / 'summary.json').exists():
                    row['summary'] = json.loads((case_output / 'summary.json').read_text(encoding='utf-8'))
                atomic_json(status_path, state)
            verify()
            state['status'] = 'completed_with_failures' if any(row['status'] == 'failed' for row in rows) else 'completed'
            state['source_stable'] = True
            state['completed_unix'] = time.time()
            atomic_json(status_path, state)
            print(json.dumps({'status': state['status'], 'cases': len(rows)}))
        finally:
            if power:
                power(0x80000000)


if __name__ == '__main__':
    main()
