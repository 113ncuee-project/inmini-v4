"""Read-only, text/numeric audit of a frozen full-system DSE experiment."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_local_rl_diagnostics import read, write, sources
from full_system_dse import objective_cost, physical_violation
from simple_rapidchiplet.hybrid.shared_evidence import verify_physical_sources
from simple_rapidchiplet.hybrid.full_space import DesignTrial
from simple_rapidchiplet.hybrid.rl_schema import DesignState


def analyze(out):
    out = Path(out).resolve()
    errors, warnings = [], []
    config = SimpleNamespace(**read(out / 'effective_config.json'))
    manifest = read(out / 'source_snapshot_manifest.json')
    if sources(out / 'source_snapshot') != manifest['files']:
        errors.append('frozen source hash mismatch')
    for name, digest in read(out / 'seed_input_manifest.json')['files'].items():
        if hashlib.sha256((Path(config.seed_case) / name).read_bytes()).hexdigest() != digest:
            errors.append('seed input hash mismatch: ' + name)
    common = read(out / 'common.json')
    if not common:
        return dict(passed=False, errors=errors + ['common seed evaluation not completed'], warnings=warnings)
    inherited = read(out / 'imported_shared.json')
    if inherited:
        history = out / 'shared_physical_history'
        for name, digest in inherited['data_hashes'].items():
            if hashlib.sha256((history / name).read_bytes()).hexdigest() != digest:
                errors.append('imported shared history hash mismatch: ' + name)
        for name in ('common.json', 'screening.json'):
            if hashlib.sha256((out / name).read_bytes()).hexdigest() != inherited['data_hashes'][name]:
                errors.append('imported current shared evidence differs: ' + name)
        try:
            verify_physical_sources(read(history / 'source_snapshot_manifest.json')['files'], manifest['files'])
        except ValueError as error:
            errors.append(str(error))
    shared = read(out / 'shared_start.json', dict(full_ppa=common['full_ppa'], wall_s=common['wall_s']))
    from profile_mapping_oracle import load_workload
    w = load_workload(Path(config.seed_case) / 'workload.json')
    base_macs = sum(l.macs_per_sample for l in w.layers) * w.batch_size
    curves, coverage, stats, architecture_rows, pareto_rows = [], [], [], [], []
    screen = read(out / 'screening.json', dict(rows=[]))
    canonical_groups = bool(read(out / 'design_space.json', {}).get('decision_identity'))
    for method in config.methods.split(','):
        folder = out / method
        ledger = read(folder / 'worker/accounting.json', [])
        full = sum(r.get('full_evaluations', 0) for r in ledger)
        physical_keys = [r.get('durable_key') for r in ledger if r.get('full_evaluations')]
        if len(set(physical_keys)) != len(physical_keys):
            errors.append(method + ': duplicated charged durable PPA')
        path = folder / 'events.jsonl'
        events = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        for e in events:
            expected_hash = (DesignTrial.from_dict(e['action']).fingerprint if canonical_groups else
                             DesignState.from_dict(e['action']['state']).fingerprint)
            if expected_hash != e['state_hash']:
                errors.append(method + ': decision hash mismatch at step ' + str(e['step']))
            ps = e['probabilities']
            if any(p < 0 or not math.isfinite(p) for p in ps) or abs(sum(ps) - 1) > 1e-6:
                errors.append(method + ': bad probabilities at step ' + str(e['step']))
            if abs(ps[e['selected_index']] - e['behavior_probability']) > 1e-8:
                errors.append(method + ': selected probability mismatch')
            if e['status'] == 'full_completed':
                m = e['metrics']
                if m['executed_macs'] != base_macs + m['recompute_macs']:
                    errors.append(method + ': MAC conservation failure at ' + str(e['step']))
                if not math.isclose(m['power_w'] * m['latency_s'], m['energy_j'], rel_tol=1e-10):
                    errors.append(method + ': power/energy identity failure')
                if m['feasible']:
                    value = objective_cost(m, common['reference'], config)
                    if not math.isclose(e['cost'], value, rel_tol=1e-10):
                        errors.append(method + ': cost identity failure')
                    expected = 10 * math.log(e['before_cost'] / value)
                    if not math.isclose(e['reward'], expected, rel_tol=1e-10, abs_tol=1e-12):
                        errors.append(method + ': reward identity failure')
                elif e.get('reward') is not None:
                    errors.append(method + ': capacity-invalid candidate supplied reward')
            elif e.get('reward') is not None:
                errors.append(method + ': unfinished oracle supplied reward')
            curves.append(dict(method=method, step=e['step'], full_ppa=e['charged_full_ppa'],
                charged_wall_s=shared['wall_s'] + e['search_wall_s'], best_cost=e['best_cost'],
                gain_vs_seed_pct=100 * (1 - e['best_cost'] / common['seed_cost']), phase=e['phase']))
        if sum(e.get('new_full_ppa', 0) for e in events) != full:
            warnings.append(method + ': orphan completed response or interrupted bookkeeping; inspect durable evidence')
        rec = read(folder / 'recommendation.json', {})
        if rec.get('strict_feasible'):
            if not rec.get('state') or not rec['metrics']['feasible'] or physical_violation(rec['metrics'], config):
                errors.append(method + ': strict recommendation violates a physical constraint')
        elif rec.get('state') is not None:
            errors.append(method + ': infeasible recommendation exposes a design as feasible')
        learn_path = folder / 'learning.jsonl'
        learning = [json.loads(l) for l in learn_path.read_text(encoding='utf-8').splitlines()] if learn_path.exists() else []
        valid_steps = {e['step']: e for e in events if e.get('feasible') and e.get('new_full_ppa')}
        if len({l['transition_step'] for l in learning}) != len(learning):
            errors.append(method + ': repeated physical transition trained twice as a new observation')
        for row in learning:
            if row['transition_step'] not in valid_steps:
                errors.append(method + ': replay transition has no corresponding completed feasible PPA')
        summary = read(folder / 'summary.json', {})
        if summary and summary['full_ppa'] != shared['full_ppa'] + full:
            errors.append(method + ': full-query summary mismatch')
        measured = [dict(state=r['trial']['state'], metrics=r['metrics'], cost=r['cost'], source='shared_screen')
                    for r in screen['rows'] if r.get('cost') is not None]
        measured += [read(p) for p in (folder / 'evaluations').glob('*.json') if read(p).get('cost') is not None]
        by_arch = {}
        for r in measured:
            arch = r['state']['architecture_candidate_id']
            if arch not in by_arch or r['cost'] < by_arch[arch]['cost']:
                by_arch[arch] = r
        architecture_rows.extend(dict(method=method, architecture=a, **r) for a, r in by_arch.items())
        # Keep whole measured states; never stitch the minimum of different
        # mappings into a fabricated PPA tuple.
        frontier = []
        axes = ('latency_s', 'energy_j', 'area_mm2')
        for r in sorted(measured, key=lambda r: tuple(r['metrics'][k] for k in axes)):
            if not any(all(p['metrics'][k] <= r['metrics'][k] for k in axes) for p in frontier):
                frontier = [p for p in frontier if not all(r['metrics'][k] <= p['metrics'][k] for k in axes)]
                frontier.append(r)
        pareto_rows.extend(dict(method=method, **r) for r in frontier)
        families = {}
        for e in events:
            f = e['action']['family']
            r = families.setdefault(f, dict(method=method, family=f, selected=0, full_completed=0, feasible=0, improvements=0))
            r['selected'] += 1
            r['full_completed'] += e.get('new_full_ppa', 0)
            r['feasible'] += int(bool(e.get('feasible')))
            r['improvements'] += int(bool(e.get('improved_global_best')))
        coverage.extend(families.values())
        stats.append(dict(method=method, full_ppa=shared['full_ppa'] + full,
            timeouts=sum(r.get('status') == 'timeout' for r in ledger),
            feasible=sum(bool(e.get('feasible')) for e in events), learning_transitions=len(learning),
            multi_step_transitions=sum(not l['terminal'] for l in learning),
            completed=bool(summary), best_cost=summary.get('best_cost')))
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    write(analysis / 'architecture_best.json', architecture_rows)
    write(analysis / 'measured_pareto.json', pareto_rows)
    for name, rows in (('curves.csv', curves), ('family_coverage.csv', coverage), ('methods.csv', stats)):
        if rows:
            with (analysis / name).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    # Same completed PPA floor and same charged wall floor; actual counts are
    # used instead of the configured maximum.
    equal = []
    method_names = config.methods.split(',')
    for axis in ('full_ppa', 'charged_wall_s'):
        available = {m: [r for r in curves if r['method'] == m] for m in method_names}
        if not all(available.values()):
            continue
        floor = min(max(r[axis] for r in available[m]) for m in method_names)
        for m, rows in available.items():
            chosen = [r for r in rows if r[axis] <= floor]
            equal.append(dict(axis=axis, shared_budget=floor, method=m,
                best_cost=min((r['best_cost'] for r in chosen), default=common['seed_cost'])))
    write(analysis / 'equal_budget.json', equal)
    audit = dict(passed=not errors, errors=errors, warnings=warnings, source_identity=manifest['identity'],
                 methods=stats, physical_scope='analytical PPA; same schema3 hardware; full schedules and lifetime checks',
                 comparison_scope='project Tangram-style seed and same-domain SA; no original Gemini executable comparison')
    write(analysis / 'audit.json', audit)
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = analyze(args.out)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(not result['passed'])
