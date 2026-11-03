"""Offline sequential experiments. Administrative commands never import the model/oracle.

Run from a normal terminal; no Codex service, LLM, network or polling agent required.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile
import uuid

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
PROFILES = {
    'smoke': dict(queries=12, seed_queries=3, seed_seconds=30, seconds=90, oracle=30, dp=5),
    'diagnostic': dict(queries=24, seed_queries=4, seed_seconds=300, seconds=1500, oracle=300, dp=60),
    'research': dict(queries=128, seed_queries=16, seed_seconds=7200, seconds=7200, oracle=1800, dp=120),
}
ARMS = ['seed_only', 'random', 'greedy', 'atomic_rl', 'bottleneck_rl']

def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.2, .02 * 2**attempt))

def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def sources(root=APP):
    files = {}
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', '.git', 'output', 'results', '.venv', 'tmp', 'node_modules'))
        for name in sorted(names):
            p = Path(folder) / name
            if name != 'snapshot_manifest.json' and p.suffix in ('.py','.js','.html','.json','.yaml','.yml','.toml','.css','.txt','.cpp','.h','.dll','.so','.ps1','.md'):
                files[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return files

def environment():
    dependencies = {}
    for name in ('numpy', 'torch', 'torchvision'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    ram = None
    if os.name == 'nt':
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        m = Memory(); m.length = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            ram = m.total
    return dict(python=sys.version, executable=sys.executable, dependencies=dependencies,
        os=platform.platform(), cpu_count=os.cpu_count(), ram_bytes=ram,
        threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})

@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.write(b'0'); stream.flush(); stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

def build_plan(args):
    profile = dict(PROFILES[args.profile])
    for key in profile:
        if getattr(args, key, None) is not None:
            profile[key] = getattr(args, key)
    if any(v <= 0 for v in profile.values()) or profile['queries'] < 6:
        raise ValueError('positive budgets and queries >= 6 required')
    if profile['seed_queries'] > profile['queries'] // 3:
        raise ValueError('seed queries must be <= total queries // 3')
    if args.profile != 'research' and profile['seed_seconds'] + profile['seconds'] > 1800:
        raise ValueError('over 30 minutes/arm requires explicit research profile')
    if args.architecture:
        import re
        match = re.fullmatch(r'single_array_c(\d+)_(\d+)x(\d+)', args.architecture)
        if not match or not (1 <= int(match[1]) <= 16 and int(match[2]) * int(match[3]) == int(match[1])):
            raise ValueError('invalid complete v03 architecture ID')
        if args.phase != 'p0' and args.phase != 'p6' and int(match[1]) < 2:
            raise ValueError('fixed experiments require >=2 Compute dies')
    phase = args.phase
    if args.profile == 'smoke' and phase != 'p0':
        raise ValueError('smoke profile is restricted to p0 Toy')
    models = ['toy'] if phase == 'p0' else ['resnet50', 'mobilenet_v2'] if phase == 'p4' else ['resnet18']
    seeds = [7, 17, 37] if phase == 'p5' else [args.seed]
    space = 'mapping_fusion' if phase == 'p3' else 'joint' if phase == 'p6' else 'mapping_only'
    cases = []
    for model in models:
        for seed in seeds:
            group = f'{model}/{seed}'
            cases.append(dict(id=group + '/seed', model=model, seed=seed, space=space, fusion='off',
                arms='seed_only', generator='guided', features='on', common=None, kind='seed'))
            variants = [('methods', ','.join(ARMS), 'guided', 'on')]
            if phase == 'p2':
                variants = [(f'G{g}F{f}', 'bottleneck_rl', 'guided' if g else 'generic', 'on' if f else 'off')
                            for g in (0, 1) for f in (0, 1)]
            elif phase in ('p3', 'p4', 'p5', 'p6'):
                variants = [('methods', 'seed_only,greedy,bottleneck_rl', 'guided', 'on')]
            if args.arms:
                if any(a not in ARMS for a in args.arms.split(',')):
                    raise ValueError('unknown arm')
                variants = [(n, args.arms, g, f) for n, _, g, f in variants]
            if phase == 'p3':
                cases.append(dict(id=group+'/methods_off', model=model, seed=seed, space=space, fusion='off',
                    arms='seed_only,greedy,bottleneck_rl', generator='guided', features='on', common=group+'/seed', kind='search'))
            for name, arms, generator, features in variants:
                cases.append(dict(id=group + '/' + name, model=model, seed=seed, space=space,
                    fusion='on' if phase in ('p3', 'p6') else 'off', arms=arms,
                    generator=generator, features=features, common=group + '/seed', kind='search',
                    start_rank=seeds.index(seed) if phase=='p5' else 0))
            if phase == 'p3':
                cases.append(dict(id=group + '/fusion_pair', kind='pair', run=group + '/methods',
                    model=model, seed=seed))
                cases.append(dict(id=group+'/action_pairs',kind='action_pairs',run=group+'/methods',model=model,seed=seed))
    return dict(schema_version=1, profile=args.profile, phase=phase, budgets=profile, cases=cases,
        fixed_architecture=args.architecture if space != 'joint' else None,
        objective='balanced', batch=1, input_size=224, sequential=True,
        limitations=['analytical uncalibrated PPA; not silicon', 'atomic and composite use different action spaces',
            'whole-flow waits are partial attribution', 'no calibrated C-panel/spill or SRAM macros'])

def command(case, plan, out, resume=False):
    snapshot = out / 'source_snapshot'
    b = plan['budgets']; target = out / 'cases' / case['id']
    if case['kind'] == 'action_pairs':
        return [sys.executable, '-X', 'utf8', str(snapshot/'scripts/run_local_action_pairs.py'),
            '--snapshot', str(snapshot), '--run', str(out/'cases'/case['run']), '--out', str(target),
            '--seconds', str(b['seconds']), '--eval-timeout', str(b['oracle']), '--max-pairs', '5',
            '--stop-file', str(out/'STOP_REQUESTED')]
    if case['kind'] == 'pair':
        return [sys.executable, '-X', 'utf8', str(snapshot / 'scripts/run_single_array_counterfactual.py'),
            '--snapshot', str(snapshot), '--run', str(out / 'cases' / case['run']), '--out', str(target),
            '--seconds', str(min(b['seconds'], b['oracle'] * 2)), '--eval-timeout', str(b['oracle'])]
    cmd = [sys.executable, '-X', 'utf8', str(snapshot / 'single_array_dse.py'), '--engine', 'v03',
        '--model', case['model'], '--seed', str(case['seed']), '--search-space', case['space'],
        '--fusion', case['fusion'], '--generator', case['generator'], '--bottleneck-features', case['features'],
        '--arms', case['arms'], '--queries', str(b['queries']), '--seed-queries', str(b['seed_queries']),
        '--seed-seconds', str(b['seed_seconds']), '--seconds', str(b['seconds']),
        '--seed-dp-seconds', str(b['dp']), '--eval-timeout', str(b['oracle']), '--shortlist', '32',
        '--preference', 'balanced', '--start-rank', str(case.get('start_rank',0)), '--stop-file', str(out / 'STOP_REQUESTED'), '--out', str(target)]
    if plan['fixed_architecture']:
        cmd += ['--fixed-architecture', plan['fixed_architecture']]
    if case['common']:
        cmd += ['--common-seed', str(out / 'cases' / case['common'])]
    if resume and (target / 'source_identity.json').exists():
        cmd += ['--resume']
    return cmd

def worst_case(plan):
    b = plan['budgets']
    # Conservative CPU bound includes common seed once per group and every arm's search.
    seconds = queries = 0
    for case in plan['cases']:
        if case['kind'] == 'seed':
            seconds += b['seed_seconds']; queries += b['seed_queries']
        elif case['kind'] == 'action_pairs':
            seconds += b['seconds']; queries += 10
        elif case['kind'] == 'pair':
            seconds += min(b['seconds'], b['oracle'] * 2); queries += 2
        else:
            n = len([a for a in case['arms'].split(',') if a != 'seed_only'])
            seconds += n * b['seconds']; queries += n * (b['queries'] - 1)
    return dict(unique_backend_upper_s=seconds, unique_full_ppa_upper=queries,
        charged_wall_upper_per_search_arm_s=b['seed_seconds'] + b['seconds'],
        note='PPA/wall caps; startup/report overhead extra; reaching a cap is not a completion guarantee')

def verify_snapshot(out):
    manifest = read(out / 'source_snapshot_manifest.json')
    if not manifest or sources(out / 'source_snapshot') != manifest['files']:
        raise ValueError('immutable snapshot missing/changed; preserve output and create a new run')

def run(out, plan=None, research_confirm=False):
    out = out.resolve()
    # One local diagnostic suite per checkout, even if output directories differ.
    with lock(ROOT / 'output/.local_rl_diagnostics.lock'), lock(out / 'run.lock'):
        if plan is not None:
            if any(p.name != 'run.lock' for p in out.iterdir()):
                raise ValueError('output is nonempty; use resume for an existing run or choose a new output; no overwrite')
            if plan['profile'] == 'research' and not research_confirm:
                raise ValueError('research requires --confirm-research after reviewing dry-run')
            files = sources()
            snapshot = out / 'source_snapshot'
            for relative in files:
                target = snapshot / relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(APP / relative, target)
            manifest = dict(files=files, identity=sha(files))
            write(out / 'source_snapshot_manifest.json', manifest)
            write(snapshot / 'snapshot_manifest.json', manifest)
            write(out / 'effective_config.json', plan)
            from local_rl_feature_schema import schema
            write(out/'feature_schema.json', schema())
            write(out / 'manifest.json', dict(schema_version=1, run_id=sha([files, plan, time.time()]),
                created_utc=datetime.now(timezone.utc).isoformat(), source_identity=sha(files),
                plan_hash=sha(plan), environment=environment()))
        else:
            plan = read(out / 'effective_config.json')
            manifest = read(out / 'manifest.json')
            if not plan or manifest['plan_hash'] != sha(plan):
                raise ValueError('effective config changed or missing')
            if manifest['environment'] != environment():
                raise ValueError('Python/dependencies/OS/thread environment differs; use original environment')
        verify_snapshot(out)
        (out / 'STOP_REQUESTED').unlink(missing_ok=True)
        state = read(out / 'suite_status.json', {})
        state.update(status='running', owner_pid=os.getpid(), cases=state.get('cases', []), last_error=None)
        began = time.monotonic()
        write(out / 'suite_status.json', state)
        try:
            for case in plan['cases']:
                if (out / 'STOP_REQUESTED').exists():
                    state['status'] = 'interrupted'; break
                old = next((r for r in state['cases'] if r['id'] == case['id']), None)
                if old and old['status'] == 'completed':
                    continue
                verify_snapshot(out)
                target = out / 'cases' / case['id']
                cmd = command(case, plan, out, resume=True)
                row = dict(id=case['id'], status='running', command=cmd, started_utc=datetime.now(timezone.utc).isoformat())
                if old:
                    state['cases'].remove(old)
                state['cases'].append(row)
                log = out / 'logs' / (case['id'].replace('/', '_') + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                row['log'] = str(log)
                with log.open('ab') as stream:
                    child_env = os.environ.copy()
                    child_env['INMIN_LOCAL_RUN_ID'] = read(out/'manifest.json')['run_id']
                    child = subprocess.Popen(cmd, cwd=out / 'source_snapshot', stdout=stream, stderr=stream, env=child_env,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    row['pid'] = child.pid
                    write(out / 'suite_status.json', state)
                    try:
                        code = child.wait()
                    except KeyboardInterrupt:
                        (out / 'STOP_REQUESTED').touch()
                        state['status'] = 'stopping_after_current_oracle'
                        write(out / 'suite_status.json', state)
                        code = child.wait()
                row['return_code'] = code
                if (out / 'STOP_REQUESTED').exists():
                    row['status'] = 'interrupted'; state['status'] = 'interrupted'; break
                row['status'] = 'completed' if code == 0 else 'failed'
                write(out / 'suite_status.json', state)
                if code != 0:
                    state.update(status='failed', last_error=f"{case['id']}: exit {code}; see {log}")
                    break
                if case['kind'] == 'seed' and plan['phase'] != 'p0':
                    seeds = read(target / 'seed_attempts.json', {}).get('bootstrap', [])
                    if not seeds or (plan['fixed_architecture'] and not any(
                            len({pe.split('/')[0] for m in s['state']['mapping']['layer_mappings'] for pe in m['chiplets']}) >= 2
                            and s['metrics'].get('nop_hop_bytes', 0) > 0 for s in seeds)):
                        state.update(status='failed', last_error='fixed seed not feasible with multi-die work/NoP traffic; inspect seed capacity evidence')
                        break
            else:
                state['status'] = 'completed'
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', last_error=str(error))
            raise
        finally:
            state['unique_runner_wall_s'] = state.get('unique_runner_wall_s', 0) + time.monotonic() - began
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            write(out / 'suite_status.json', state)
            from analyze_local_rl_diagnostics import analyze
            analyze(out)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if state['status'] == 'completed' else 1

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('help','list','doctor','dry-run','run','status','stop','resume','summarize','audit','export','replay'))
    parser.add_argument('--profile', choices=PROFILES, default='diagnostic')
    parser.add_argument('--phase', choices=[f'p{i}' for i in range(7)], default='p1')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/local_rl_v03_r18')
    parser.add_argument('--architecture', default='single_array_c2_2x1')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--arms', help='optional explicit comma-separated arm selection')
    parser.add_argument('--confirm-research', action='store_true')
    for key in PROFILES['diagnostic']:
        parser.add_argument('--' + key.replace('_','-'), type=int if 'queries' in key else float)
    args = parser.parse_args(argv); out = args.out.resolve()
    supplied = sys.argv[1:] if argv is None else argv
    if args.command == 'run' and ('--profile' not in supplied or '--phase' not in supplied):
        parser.error('run requires explicit --profile and --phase; inspect dry-run first')
    if args.command in ('help','list'):
        print(json.dumps(dict(profiles=PROFILES, phases={f'p{i}': n for i,n in enumerate(
            ['Toy smoke','ResNet18 mapping','ResNet18 2x2 ablation','ResNet18 fusion + one fixed pair',
             'ResNet50/MobileNetV2 mapping','ResNet18 RNG 7/17/37','optional joint'])}, arms=ARMS,
            budgets={p:{f'p{i}':worst_case(build_plan(argparse.Namespace(**{**vars(args),'profile':p,'phase':f'p{i}'})))
                        for i in ([0] if p=='smoke' else range(1,7))} for p in PROFILES}), indent=2))
    elif args.command == 'doctor':
        env = environment(); files = sources()
        active = False
        try:
            with lock(ROOT / 'output/.local_rl_diagnostics.lock'):
                pass
        except OSError:
            active = True
        print(json.dumps(dict(environment=env, free_disk_bytes=shutil.disk_usage(ROOT).free,
            source_identity=sha(files), source_files=len(files), runner_active=active,
            output_exists=out.exists(), ready=all(env['dependencies'].values()),
            legacy_workflow=read(ROOT/'output/inmin_v03_20261102/formal_v4/workflow_control/workflow_status.json',{}).get('status'),
            note='No model/oracle run. Close unrelated heavy experiments before a timing comparison.'), indent=2))
    elif args.command == 'dry-run':
        plan = build_plan(args)
        print(json.dumps(dict(plan=plan, worst_case=worst_case(plan), commands=[command(c,plan,out) for c in plan['cases']]), indent=2))
    elif args.command == 'run':
        return run(out, build_plan(args), args.confirm_research)
    elif args.command == 'resume':
        return run(out)
    elif args.command == 'stop':
        if not (out / 'manifest.json').exists():
            raise ValueError('not a local runner output')
        (out / 'STOP_REQUESTED').touch()
        print('Stop requested: finish current bounded oracle, save checkpoint, then exit. No unrelated process is killed.')
    elif args.command == 'status':
        from analyze_local_rl_diagnostics import status
        print(json.dumps(status(out), indent=2, ensure_ascii=False))
    elif args.command in ('summarize','audit','replay'):
        from analyze_local_rl_diagnostics import analyze, replay
        if args.command == 'replay':
            print(json.dumps(replay(out), indent=2))
        else:
            result = analyze(out)
            print(json.dumps(result, indent=2))
            return int(args.command == 'audit' and bool(result['errors']))
    elif args.command == 'export':
        from analyze_local_rl_diagnostics import analyze
        analyze(out)
        target = out.parent / (out.name + '_evidence_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.zip')
        with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.md','.gz','.py','.js','.html','.yaml','.yml','.toml','.css','.txt') and not any(
                        part in ('__pycache__','completed','verified') for part in path.relative_to(out).parts):
                    archive.write(path, path.relative_to(out))
        with zipfile.ZipFile(target) as archive:
            if archive.testzip():
                raise ValueError('ZIP CRC failed')
        print(target)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
