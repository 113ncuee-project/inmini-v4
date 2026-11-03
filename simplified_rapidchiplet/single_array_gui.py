"""GUI04 v3: single-array oracle, fixed UI intent, text-only artifacts."""
from pathlib import Path
from types import SimpleNamespace
import json

from simple_rapidchiplet.hybrid.auto_runtime import atomic_json
from simple_rapidchiplet.hybrid.preferences import constraint_report
from simple_rapidchiplet.hybrid.single_array_hardware import PROFILE, enumerate_single_array_architectures
from rl_dse import PPA_PREFERENCES, make_objective, rank_joint_rows
from single_array_dse import run


def run_suite(config, folder, update=lambda **_: None, *, resume=False):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if resume and (folder / 'config.json').exists() and json.loads((folder / 'config.json').read_text(encoding='utf-8')) != config:
        raise ValueError('saved v0.2 suite settings differ')
    atomic_json(folder / 'config.json', config)
    objective = make_objective(config)
    modern = config.get('engine_profile') == 'gui04_v4'
    from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
    candidates = {c.id: c for c in (enumerate_single_array_architectures_v3() if modern else enumerate_single_array_architectures())}
    profile = next(iter(candidates.values())).hardware.profile_id
    rows, searches, models = [], {}, {}
    for index, model in enumerate(config['models']):
        update(completed=index, total=len(config['models']), message=f'{model}: preparing v0.2 Batch1 search')
        model_folder = folder / model
        args = SimpleNamespace(model=model, input_size=224, seed=config['seed'],
            queries=config['budget'], seconds=config['max_wall_time_s'], eval_timeout=config['evaluation_timeout_s'],
            shortlist=64 if modern else 48, engine='v03' if modern else 'v02',
            seed_seconds=config.get('seed_wall_time_s', 7200.), seed_queries=config.get('seed_queries',16),
            seed_dp_seconds=config.get('seed_dp_seconds',120.), fusion='on', arms='bottleneck_rl', out=str(model_folder), resume=resume,
            preference=config['preference'], max_power=config['max_power'], max_area=config['max_area'],
            max_latency_ms=config['max_latency_ms'], soft_limits=False, seed_method='tangram_style_v02')
        try:
            summary = run(args, objective_override=objective, requested_ui=config, update=update)['bottleneck_rl']
            data = json.loads((model_folder / 'bottleneck_rl' / 'joint_search.json').read_text(encoding='utf-8'))
            searches[model] = {'status': summary['status'], 'summary': summary}
            for ident, archived in data['architecture_archive'].items():
                metrics = archived['metrics']
                constraints = constraint_report(metrics, objective)
                rows.append({'model': model, 'architecture': ident, 'metrics': metrics,
                    'cost': archived['cost'], 'constraints': constraints, 'within_limits': constraints['strict_compliant'],
                    'all_targets_met': constraints['all_targets_met'], 'folder': str(model_folder),
                    'hardware_resources': candidates[ident].hardware.resource_totals,
                    'hardware_profile': profile, 'text_report_file': 'REPORT.md', 'error': None})
            models[model] = {'profile': profile, 'batch_size': 1, 'source_identity': str(model_folder / 'source_identity.json')}
        except Exception as error:
            searches[model] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            rows.append({'model': model, 'architecture': 'no_completed_candidate', 'metrics': {'feasible': False},
                         'cost': None, 'constraints': {'physical_feasible': False, 'strict_compliant': False,
                         'all_targets_met': False, 'limits': []}, 'within_limits': False,
                         'all_targets_met': False, 'folder': str(model_folder), 'error': str(error)})
        atomic_json(folder / 'trials.json', rows)
        update(completed=index + 1, total=len(config['models']), rows=rows)
    rankable = [row for row in rows if row['metrics'].get('feasible')]
    ranking, recommendation = rank_joint_rows(rankable, config['models'])
    result = {'config': config, 'models': models, 'runs': rows, 'searches': searches,
              'ranking': ranking, **recommendation, 'chiplet_profile': profile,
              'weights': PPA_PREFERENCES[config['preference']],
              'report_format': 'text_numeric_tables',
              'interpretation': 'best verified feasible design within finite search budget; uncalibrated analytical model'}
    atomic_json(folder / 'suite.json', result)
    atomic_json(folder / 'recommendation.json', recommendation)
    update(completed=len(config['models']), total=len(config['models']), result=result,
           message='v0.2 search finished; see verified numerical results and actual coverage')
    return result
