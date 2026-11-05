"""Import only an identical physical oracle's shared, pre-search observations.

This never reads RL/SA/Random search labels or trained policies. Historical
queries and time stay charged to every new arm, and input/physical sources are
verified before reuse. A changed physical model requires new evaluations.
"""
import hashlib
import json
from pathlib import Path
import shutil


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_query_count(reference_state, initial_state):
    """Two logical calls may share one exact physical durable-cache payload.

    Schema3 uses one versioned physical profile for these architecture IDs.
    Derived state reports/budgets are not oracle inputs; grouping, PE order,
    all mapping fields and the architecture remain in the identity.
    """
    def key(state):
        return json.dumps(dict(architecture=state['architecture_candidate_id'],mapping=state['mapping']),sort_keys=True)
    return len({key(reference_state),key(initial_state)})


def verify_physical_sources(donor_files, current_files):
    prefix = 'simple_rapidchiplet/'
    excluded = {prefix + 'hybrid/full_space.py', prefix + 'hybrid/multi_step_q.py',
                prefix + 'hybrid/single_array_seed_v3.py', prefix + 'hybrid/mapping_proposals_v4.py',
                prefix + 'hybrid/rl_actions.py'}
    extra = {prefix + 'hybrid/tensor_boundary.py', prefix + 'hybrid/resource_projection.py',
             prefix + 'hybrid/shared_evidence.py', prefix + 'hybrid/lp_five.py', prefix + 'hybrid/temporal_projection.py',
             prefix + 'hybrid/tensor_locality.py', prefix + 'hybrid/cost_value_q.py',
             prefix + 'hybrid/architecture_refresh.py', prefix + 'hybrid/service_features.py',
             prefix + 'hybrid/residual_cost_q.py'}
    def physical(name):
        return (name.startswith(prefix) and name not in excluded and name not in extra
                and Path(name).suffix in ('.py', '.cpp', '.h', '.dll', '.so', '.json', '.yaml', '.toml')) or name == 'scripts/profile_mapping_oracle.py'
    before, after = {n: v for n, v in donor_files.items() if physical(n)}, {n: v for n, v in current_files.items() if physical(n)}
    if not before or before != after:
        changed = sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))
        raise ValueError('shared history physical source mismatch: ' + ', '.join(changed))
    return before


def import_shared(donor, out, current_files, seed_manifest, config):
    donor, out = Path(donor).resolve(), Path(out).resolve()
    if donor == out:
        raise ValueError('shared history donor must differ from output')
    source = _read(donor / 'source_snapshot_manifest.json')
    checked = verify_physical_sources(source['files'], current_files)
    for name, digest in checked.items():
        if _digest(donor / 'source_snapshot' / name) != digest:
            raise ValueError('donor frozen physical source changed: ' + name)
    if _read(donor / 'seed_input_manifest.json')['files'] != seed_manifest['files']:
        raise ValueError('shared history workload/seed inputs mismatch')
    old_environment, new_environment = _read(donor / 'environment.json'), _read(out / 'environment.json')
    for key in ('python', 'dependencies', 'os', 'threads'):
        if old_environment.get(key) != new_environment.get(key):
            raise ValueError('shared history numerical environment mismatch: ' + key)
    previous = _read(donor / 'effective_config.json')
    for key in ('objective', 'max_area', 'max_power', 'max_latency_ms', 'fixed_architecture', 'architecture_screen'):
        if previous.get(key) != getattr(config, key, None):
            raise ValueError('shared history protocol mismatch: ' + key)
    from ppa_preferences import LEGACY_OBJECTIVES,objective_contract
    descriptor=donor/'objective.json'
    if descriptor.exists():
        prior_contract=_read(descriptor)['contract']
    elif previous['objective'] in LEGACY_OBJECTIVES:
        # These four formulas/weights are retained exactly from full v0.9.
        prior_contract=objective_contract(previous['objective'])
    else:
        raise ValueError('shared PPA preference requires its versioned objective descriptor')
    if prior_contract!=objective_contract(config.objective):
        raise ValueError('shared history user PPA preference formula/weights mismatch')
    common = _read(donor / 'common.json')
    screen = _read(donor / 'screening.json') if (donor / 'screening.json').exists() else dict(rows=[], full_ppa=0, wall_s=0., completed=False)
    if config.architecture_screen == 'all' and (not screen['completed'] or not screen['complete_domain_attempted']):
        raise ValueError('shared architecture screen is not complete')
    if not common['metrics']['feasible'] or not common['reference']['feasible']:
        raise ValueError('shared common evaluations must be executable')
    reference_state = _read(donor / 'seed_input_snapshot/seed_attempts.json')['bootstrap'][0]['state']
    expected_common = common_query_count(reference_state, common['initial_state'])
    if common['full_ppa'] != expected_common or screen.get('full_ppa', 0) != sum(r['status'] == 'full_completed' for r in screen['rows']):
        raise ValueError('shared historical query ledger mismatch')
    if any(r['status'] == 'full_completed' and 'metrics' not in r for r in screen['rows']):
        raise ValueError('completed shared observations require actual metrics, including invalid outcomes')
    copied = {}
    history = out / 'shared_physical_history'
    history.mkdir(parents=True, exist_ok=True)
    for name in ('common.json', 'screening.json', 'seed_input_manifest.json', 'source_snapshot_manifest.json', 'environment.json', 'effective_config.json'):
        target = history / name
        if name == 'screening.json' and not (donor / name).exists():
            target.write_text(json.dumps(screen, indent=2), encoding='utf-8')
        else:
            shutil.copyfile(donor / name, target)
        copied[name] = _digest(target)
    if descriptor.exists():
        target=history/'objective.json'
        shutil.copyfile(descriptor,target)
        copied['objective.json']=_digest(target)
    for name in ('common.json', 'screening.json'):
        shutil.copyfile(history / name, out / name)
    return dict(donor=str(donor), donor_source_identity=source['identity'], data_hashes=copied,
        physical_source_files=checked, inherited_full_ppa=common['full_ppa'] + screen.get('full_ppa', 0),
        inherited_wall_s=common['wall_s'] + screen['wall_s'], additional_full_ppa=0,
        scope='only shared pre-search physical observations; historical full queries/time charged to every arm; no control/search labels or policy imported')
