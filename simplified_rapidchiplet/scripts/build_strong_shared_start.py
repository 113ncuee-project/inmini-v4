"""Combine charged architecture observations and wider-DP PPA into a shared start.

No policy or search-arm label is read. Full metrics come from durable completed
oracle responses. Invalid completed PPA remains charged, never becomes reward.
The entire chain of prior data/source identities is preserved for audit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from run_local_rl_diagnostics import read, write, sources, sha
from full_system_dse import objective_cost
from simple_rapidchiplet.hybrid.full_space import DesignTrial
from simple_rapidchiplet.hybrid.rl_schema import stable_hash
from simple_rapidchiplet.hybrid.shared_evidence import verify_physical_sources
from profile_mapping_oracle import load_workload


def build(original, baseline, out, seeds=(7, 17), baseline_wall_s=None, model_id='resnet18'):
    original, baseline, out = Path(original).resolve(), Path(baseline).resolve(), Path(out).resolve()
    current_files = sources(APP)
    verify_physical_sources(read(original / 'source_snapshot_manifest.json')['files'], current_files)
    common = read(original / 'common.json')
    workload = load_workload(original / 'seed_input_snapshot/workload.json')
    base_screen = read(original / 'screening.json',dict(rows=[],full_ppa=0,wall_s=0.,completed=False))
    original_config = read(original / 'effective_config.json')
    if 'full_ppa' not in base_screen:
        if base_screen.get('rows') or original_config.get('architecture_screen')!='off':
            raise ValueError('nonempty architecture screen lacks its physical query ledger')
        # Older valid off-screen runs stored an empty phase without this key.
        base_screen = dict(base_screen,full_ppa=0)
    if original_config.get('architecture_screen') == 'all' and not base_screen['completed']:
        raise ValueError('bounded architecture-screen phase must finish before refining its shared start')
    objective = SimpleNamespace(**original_config)
    results = read(baseline / 'summary.json')
    if not results or not results['source_stable']:
        raise ValueError('stronger DP comparison did not complete with stable sources')
    identity = sha(current_files)
    rows = [dict(**r, physical_origin=dict(phase='original_architecture_screen', source_identity=read(original / 'source_snapshot_manifest.json')['identity']))
            for r in base_screen['rows']]
    additional_full, additional_wall, origins = 0, 0., []
    for result in results['architectures']:
        folder = baseline / result['architecture']
        events = [json.loads(l) for l in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        durable = {}
        for path in (folder / 'worker/completed').glob('*.json'):
            data = read(path)
            durable[stable_hash(dict(architecture=data['architecture'], mapping=data['mapping']))] = data['metrics']
        ledger = read(folder / 'worker/accounting.json', [])
        full = sum(r.get('full_evaluations', 0) for r in ledger)
        if full != result['new_full_ppa']:
            raise ValueError('strong baseline accounting mismatch')
        completed = 0
        for event in events:
            trial = DesignTrial.from_dict(event['trial'])
            metrics = durable.get(stable_hash(dict(architecture=trial.state.architecture_candidate_id, mapping=trial.state.mapping.to_dict())))
            row = dict(architecture=trial.state.architecture_candidate_id,
                trial=DesignTrial(trial.state, 'global_mapping', tuple(m['layer_id'] for m in trial.state.mapping.to_dict()['layer_mappings']),
                    dict(shared_stage='widened_dp_and_inner_refinement', original_trial=trial.action)).to_dict(),
                physical_origin=dict(phase='widened_dp_shared', source_identity=result['source_identity'], baseline_folder=str(folder)),
                status='full_completed' if metrics else event['status'], cost=None, utility=None)
            if metrics:
                completed += 1
                if metrics['executed_macs'] != workload.total_macs + metrics.get('recompute_macs', 0):
                    raise ValueError('strong shared MAC conservation failure')
                if not math.isclose(metrics['energy_j'], metrics['power_w'] * metrics['latency_s'], rel_tol=1e-10):
                    raise ValueError('strong shared power/energy identity failure')
                value = objective_cost(metrics, common['reference'], objective)
                if value != event['cost']:
                    raise ValueError('durable full cost differs from event')
                row.update(metrics=metrics, cost=value, utility=10 * math.log(common['seed_cost'] / value) if value is not None else None)
            else:
                row['reason'] = event.get('reason', 'no complete physical observation')
            rows.append(row)
        if completed != full:
            raise ValueError('completed shared strong rows differ from charged PPA')
        additional_full += full
        additional_wall += result['generation_wall_s'] + result['search_wall_s']
        origins.append(dict(architecture=result['architecture'], full_ppa=full,
                            generation_wall_s=result['generation_wall_s'], search_wall_s=result['search_wall_s']))
    if baseline_wall_s is not None:
        if baseline_wall_s + 1e-6 < additional_wall:
            raise ValueError('whole baseline stage time cannot be smaller than its recorded generation/evaluation time')
        additional_wall = baseline_wall_s
    screen = dict(rows=rows, completed=True, complete_domain_attempted=base_screen.get('complete_domain_attempted', False),
        full_ppa=base_screen['full_ppa'] + additional_full, wall_s=base_screen['wall_s'] + additional_wall,
        note='same physically observed shared architecture screen plus paid wider-DP/inner observations; no search policy/result import')
    if not model_id or Path(model_id).name != model_id or any(c in model_id for c in ('/', '\\', ':')):
        raise ValueError('model archive label must be a simple directory name')
    for seed in seeds:
        folder = out / model_id / str(seed)
        if folder.exists():
            raise ValueError('strong shared donor already exists; keep it immutable')
        folder.mkdir(parents=True)
        # Copy the exact current engine so controller changes are distinguished
        # from locked physical semantics. Original observations retain origin.
        for name in current_files:
            target = folder / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(APP / name, target)
        write(folder / 'source_snapshot_manifest.json', dict(identity=identity, files=current_files))
        for name in ('seed_input_manifest.json', 'environment.json', 'common.json'):
            shutil.copyfile(original / name, folder / name)
        shutil.copytree(original / 'seed_input_snapshot', folder / 'seed_input_snapshot')
        config = dict(original_config)
        config.update(seed=seed, seed_case=str(folder / 'seed_input_snapshot'), out=str(folder))
        write(folder / 'effective_config.json', config)
        write(folder / 'screening.json', screen)
        write(folder / 'status.json', dict(status='completed', phase='shared_presearch_only', return_code=0))
        write(folder / 'shared_origin.json', dict(original=str(original), baseline=str(baseline),
            original_source_identity=read(original / 'source_snapshot_manifest.json')['identity'],
            current_controller_source_identity=identity, extra_physical_queries=additional_full,
            extra_charged_wall_s=additional_wall, charged_full_ppa=common['full_ppa'] + screen['full_ppa'],
            time_scope='whole baseline parent stage' if baseline_wall_s is not None else 'generation+oracle stage; parent startup separately logged',
            original_screen_full_ppa=base_screen['full_ppa'], new_observations=origins,
            state='shared before any compared RL/SA/Random arm; identical observations used for all RNGs; no trained policy transfer'))
    summary = dict(charged_full_ppa=common['full_ppa'] + screen['full_ppa'], extra_full_ppa=additional_full,
        charged_wall_s=common['wall_s'] + screen['wall_s'], source_identity=identity,
        best_cost=min([common['seed_cost']] + [r['cost'] for r in rows if r.get('cost') is not None]),
        feasible_observations=sum(r.get('cost') is not None for r in rows), observations=len(rows), rngs=list(seeds), model_id=model_id)
    write(out / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--baseline-wall-s', type=float)
    p.add_argument('--model-id', default='resnet18')
    args = p.parse_args()
    build(args.original, args.baseline, args.out, baseline_wall_s=args.baseline_wall_s, model_id=args.model_id)
