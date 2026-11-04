"""One-off frozen-engine QA, declared ablations and stronger local baselines."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.venv/Scripts/python.exe').is_file() and (p / 'inputs').is_dir())
DEV_APP = ROOT / 'tmp/inmin_v06_graph/app'
if not (DEV_APP / 'simple_rapidchiplet/hybrid/tensor_boundary.py').exists():
    DEV_APP = ROOT / 'simplified_rapidchiplet'
HELPER_APP = Path(__file__).resolve().parent / 'engine'
sys.path.insert(0, str((HELPER_APP if HELPER_APP.exists() else DEV_APP) / 'scripts'))
from run_local_rl_diagnostics import read, write, sources, sha, lock

CONFIGS = {
    'F0P0': ['--no-boundary-features', '--no-resource-projection'],
    'F1P0': ['--no-resource-projection'],
    'F0P1': ['--no-boundary-features'],
    'F1P1': [],
}


def prepare(out, shared_root):
    engine = out / 'engine'
    existing = read(out / 'protocol.json')
    if existing:
        if sources(engine) != existing['source_files']:
            raise ValueError('prepared engine source changed')
        if Path(__file__).resolve() == (out / 'orchestrate.py').resolve() and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != existing['orchestrator_sha256']:
            raise ValueError('prepared orchestrator changed')
        return existing
    files = sources(DEV_APP)
    for name in files:
        target = engine / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEV_APP / name, target)
    if sources(engine) != files:
        raise ValueError('engine copy identity mismatch')
    script = out / 'orchestrate.py'
    shutil.copyfile(Path(__file__), script)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_identity=sha(files), source_files=files,
        orchestrator_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), models=['resnet18'],
        development_seed=7, confirmation_seed=17, configurations=CONFIGS,
        feature_definition='F =75 logical tensor-boundary +14 actual before-schedule +22 ordered shard/CG-locality edit features; graph branch zero-initialized,149 shared base',
        queries='measured shared full queries +64', new_query_ceiling=64, seconds=900, oracle=240, objective='edp', shared_root=str(shared_root),
        shared_source='identical-physical-oracle common +50 layouts +new paid widened-DP/inner observations; every feasible/invalid completed query and full baseline time charged to all arms',
        selection='development smallest observed strict-feasible EDP under the declared query and wall ceilings; all alternatives retained',
        baseline='wider Tangram-style DP/inner first; same-domain SA/Random on that stronger start; SA shared-start temperature .08 and elite restart=8; extra Gemini-style LP5, frozen-Q and uniform-horizon controls',
        status='prepared_no_new_ppa')
    write(out / 'protocol.json', protocol)
    return protocol


def study(out, protocol, qa_only=False):
    engine, python = out / 'engine', sys.executable
    results = read(out / 'execution.json', dict(stages=[]))
    def run(arguments, label, essential=True):
        arguments = list(map(str,arguments))
        tick = time.perf_counter()
        write(out / 'status.json', dict(status='running', stage=label, source_identity=protocol['source_identity']))
        print(json.dumps(dict(stage=label, started_utc=datetime.now(timezone.utc).isoformat())), flush=True)
        with (out / 'commands.log').open('ab') as stream:
            stream.write((json.dumps(arguments) + '\n').encode())
            response = subprocess.run([python, '-u', '-X', 'utf8', *map(str, arguments)], cwd=ROOT, stdout=stream, stderr=stream)
        row = dict(stage=label, return_code=response.returncode, wall_s=time.perf_counter() - tick)
        results['stages'].append(row)
        write(out / 'execution.json', results)
        if response.returncode and essential:
            raise RuntimeError(label + ' failed; see commands.log')
        return response.returncode == 0

    acceptance = out / 'acceptance'
    if not read(acceptance / 'result.json', {}).get('passed'):
        run([engine / 'scripts/test_single_array_acceptance.py', '--out', acceptance], 'full_regression')
    toy = ROOT / 'output/inmin_v05_20261103/workflow_shipping/full/seed_input_snapshot'
    if not (toy / 'seed_attempts.json').exists():
        toy = out / 'toy_seed'
        if not (toy / 'seed_attempts.json').exists():
            run([engine / 'scripts/create_toy_seed.py', '--out', toy], 'create_physical_toy_fixture')
    workflow = out / 'workflow'
    if not read(workflow / 'result.json', {}).get('passed'):
        run([engine / 'scripts/test_full_system_workflow.py', '--seed-case', toy, '--out', workflow], 'real_workflow')
    if not read(out / 'gui_validation.json', {}).get('passed'):
        before = time.perf_counter()
        with (out / 'gui.log').open('w', encoding='utf-8') as stream:
            response = subprocess.run([python, '-u', '-X', 'utf8', str(engine / 'scripts/verify_full_gui.py'), '--seed-case', str(toy)],
                cwd=ROOT, stdout=stream, stderr=stream, text=True)
        if response.returncode:
            raise RuntimeError('GUI workflow failed; see gui.log')
        parsed = json.loads((out / 'gui.log').read_text(encoding='utf-8').splitlines()[-1])
        write(out / 'gui_validation.json', dict(**parsed, elapsed_s=time.perf_counter() - before))

    if qa_only:
        write(out / 'status.json',dict(status='qa_completed',source_identity=protocol['source_identity']))
        return

    # Strengthen initialization before comparing controllers. Both the DP
    # improvements and their real query/time costs are a shared paid stage.
    strong = out / 'strong_seed_baseline'
    if not read(strong / 'summary.json'):
        run([engine / 'scripts/run_strong_seed_comparison.py', '--donor', Path(protocol['shared_root']) / 'resnet18/7',
             '--out', strong], 'wider_tangram_style_before_search')
    strong_inputs = out / 'strong_shared_inputs'
    if not read(strong_inputs / 'summary.json'):
        baseline_stages = [s for s in results['stages'] if s['stage'] == 'wider_tangram_style_before_search']
        extra_time = ['--baseline-wall-s', str(sum(s['wall_s'] for s in baseline_stages))] if baseline_stages else []
        run([engine / 'scripts/build_strong_shared_start.py', '--original', Path(protocol['shared_root']) / 'resnet18/7',
             '--baseline', strong, '--out', strong_inputs, *extra_time], 'build_charged_strong_shared_start')
    shared_budget = read(strong_inputs / 'summary.json')['charged_full_ppa']
    query_ceiling = shared_budget + protocol['new_query_ceiling']
    write(out / 'shared_protocol.json', dict(shared_full_ppa=shared_budget, each_new_full_ppa_ceiling=protocol['new_query_ceiling'],
        each_total_full_ppa_ceiling=query_ceiling, shared_source=str(strong_inputs),
        rule='all methods/RNGs receive exactly the same measured strong initial archive and pay the same physical history/time'))

    def case(path, seed, flags, methods='rl'):
        donor = strong_inputs / f'resnet18/{seed}'
        if read(donor / 'status.json', {}).get('status') != 'completed':
            raise ValueError('original comparison is incomplete: ' + str(donor))
        if read(path / 'status.json', {}).get('status') == 'completed':
            return read(path / 'summary.json')
        arguments = [engine / 'full_system_dse.py', '--seed-case', ROOT / 'inputs/replay_v04/resnet18',
            '--shared-case', donor, '--out', path, '--methods', methods, '--seed', str(seed),
            '--queries', str(query_ceiling), '--seconds', str(protocol['seconds']),
            '--eval-timeout', str(protocol['oracle']), '--objective', 'edp', '--sa-elite-restart',
            '--sa-temperature-reference', 'shared', *flags]
        if (path / 'effective_config.json').exists():
            arguments = [engine / 'full_system_dse.py', '--out', path, '--resume']
        if not run(arguments, str(path.relative_to(out)), essential=False):
            return None
        run([engine / 'scripts/analyze_full_system.py', '--out', path], 'audit ' + str(path.relative_to(out)))
        return read(path / 'summary.json')

    development = {}
    for name, flags in CONFIGS.items():
        development[name] = case(out / 'development' / name / '7', 7, flags)
        write(out / 'development_results.json', development)
    feasible = {name: value['rl'] for name, value in development.items() if value and value.get('rl', {}).get('strict_feasible')}
    if not feasible:
        raise RuntimeError('all development variants failed or lacked a feasible design')
    chosen = min(feasible, key=lambda name: (feasible[name]['recommended_cost'], name))
    selection = dict(selected=chosen, flags=CONFIGS[chosen], development_seed=7, confirmation_seed=17,
        criterion=protocol['selection'], values={name: v['recommended_cost'] for name, v in feasible.items()},
        caution='development selection is not confirmation evidence; all tuning cost/results remain in the ledger')
    write(out / 'selection.json', selection)
    case(out / 'development/selected_controls/7', 7, CONFIGS[chosen], 'sa,random')
    case(out / 'development/gemini5/7', 7,
         ['--no-boundary-features', '--no-resource-projection', '--operators', 'gemini5'], 'sa')
    selected_confirmation = case(out / 'confirmation/selected/17', 17, CONFIGS[chosen], 'rl,sa,random')
    if chosen != 'F0P0':
        case(out / 'confirmation/plain/17', 17, CONFIGS['F0P0'])
    else:
        write(out / 'confirmation/plain_alias.json', dict(alias_of='selected/17', new_full_ppa=0,
                                                        reason='selected configuration is the identical plain controller; no duplicate experiment'))
    case(out / 'confirmation/frozen_shared_q/17', 17,
         [*CONFIGS[chosen], '--freeze-search-policy'])
    case(out / 'confirmation/uniform_horizon/17', 17,
         [*CONFIGS[chosen], '--uniform-policy', '--freeze-search-policy', '--no-shared-warmstart'])
    summarize(out, protocol, selection)
    passed = all(s['return_code'] == 0 for s in results['stages']) and bool(selected_confirmation)
    write(out / 'status.json', dict(status='completed' if passed else 'completed_with_failures',
        source_identity=protocol['source_identity'], selected=chosen, finished_utc=datetime.now(timezone.utc).isoformat()))
    package_ok = run([engine / 'scripts/package_full_system.py', '--tag', '20261104_graph_study',
         '--evidence', out, '--evidence', ROOT / 'output/inmin_v05_20261103/night_edp',
         '--evidence', ROOT / 'output/inmin_v05_20261103/fixed_mapping_fusion',
         '--evidence', ROOT / 'output/inmin_v05_20261103/fixed_inner_controls'], 'package_graph_study', essential=False)
    passed = passed and package_ok
    # Packaging is administrative; preserve the completed study status after it.
    write(out / 'status.json', dict(status='completed' if passed else 'completed_with_failures',
        source_identity=protocol['source_identity'], selected=chosen, finished_utc=datetime.now(timezone.utc).isoformat()))


def summarize(out, protocol, selection):
    rows = []
    for phase in ('development', 'confirmation'):
        for path in sorted((out / phase).glob('*/*/summary.json')):
            config, seed = path.parent.parent.name, path.parent.name
            for method, result in read(path).items():
                rows.append(dict(phase=phase, configuration=config, seed=seed, method=method,
                    new_full_ppa=result['search_full_ppa'], charged_full_ppa=result['full_ppa'],
                    best_cost=result['recommended_cost'], search_s=result['search_wall_s'],
                    gain_after_shared_pct=result['gain_after_shared_screen_pct'],
                    metrics=result['best'], algorithm=result.get('algorithm', method), source=str(path.relative_to(out))))
    write(out / 'summary.json', dict(protocol_source=protocol['source_identity'], selection=selection, rows=rows,
        scope='new graph/projection ablations; prototype feature/proposal effects and adapted baselines; no original-paper executable superiority claim'))
    lines = ['# Graph-edit RL study', '', 'All physical PPA comes from the unchanged complete INMIN oracle. Wider-DP and declared inner refinements are evaluated FIRST and shared as the stronger paid starting archive, including completed invalid outcomes. Gain after shared measures additional search improvement. Lower EDP is better. Development selection uses RNG7; RNG17 is independent policy initialization.', '',
        '|Phase|Configuration|RNG|Method|New full PPA|Charged full PPA|EDP|Gain after shared %|Search s|', '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        value = f"{row['best_cost']:.8f}" if row['best_cost'] is not None else 'none'
        lines.append(f"|{row['phase']}|{row['configuration']}|{row['seed']}|{row['method']}|{row['new_full_ppa']}|{row['charged_full_ppa']}|{value}|{row['gain_after_shared_pct']:.3f}|{row['search_s']:.3f}|")
    lines += ['', 'Selected on development: ' + selection['selected'] + '. This is a tuning choice, not a statistical superiority claim.', '',
        'Every case retains raw events, full query/time ledgers, source/input hashes, strict-feasible recommendation, equal-budget audit and negative results. The strong-seed REPORT separately documents widened DP costs and infeasible proposals.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT / 'output/inmin_v06_20261104/study_v8'))
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--qa-only',action='store_true',help='complete real regression/workflow/GUI gates, then stop before new DNN evaluations')
    p.add_argument('--shared-root', default=str(ROOT / 'output/inmin_v05_20261103/night_edp'))
    args = p.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to((ROOT / 'output').resolve()):
        raise ValueError('study output escapes workspace output')
    out.mkdir(parents=True, exist_ok=True)
    protocol = prepare(out, Path(args.shared_root).resolve())
    if args.prepare_only:
        print(json.dumps(dict(status='prepared_no_new_ppa', out=str(out), source_identity=protocol['source_identity'])))
        return
    with lock(out / 'study.lock'):
        try:
            study(out, protocol,args.qa_only)
        except Exception as error:
            write(out / 'status.json', dict(status='failed', error=str(error), source_identity=protocol['source_identity']))
            raise


if __name__ == '__main__':
    main()
