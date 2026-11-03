"""Unified full schema-3 DSE with multi-step neural replay RL and matched controls.

python simplified_rapidchiplet/full_system_dse.py --out output/full_v05_r18
python simplified_rapidchiplet/full_system_dse.py --resume --out output/full_v05_r18
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP / 'scripts'))
from run_local_rl_diagnostics import read, write, sources, sha, lock, environment
from profile_mapping_oracle import load_workload
from replay_mapping_dse import compact, cost, append
from simple_rapidchiplet.hybrid.full_space import FullDesignSpace, DesignTrial
from simple_rapidchiplet.hybrid.multi_step_q import MultiStepQ
from simple_rapidchiplet.hybrid.rl_actions import validate_state
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.search import _rng_tuple, plan_from_dict
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.single_array_runtime import SingleArrayOraclePool


def physical_violation(metrics, args):
    return sum(max(0., metrics[key] / limit - 1) for key, limit in (
        ('area_mm2', getattr(args, 'max_area', None)), ('power_w', getattr(args, 'max_power', None)),
        ('latency_s', getattr(args, 'max_latency_ms', None) / 1000 if getattr(args, 'max_latency_ms', None) else None)) if limit is not None)


def objective_cost(metrics, reference, args):
    value = cost(metrics, reference, args.objective)
    return None if value is None else value + 10 * physical_violation(metrics, args)


def execute(args):
    out, case = Path(args.out), Path(args.seed_case)
    manifest = read(out / 'source_snapshot_manifest.json')
    if sources(APP) != manifest['files']:
        raise ValueError('frozen source hash mismatch')
    for name, digest in read(out / 'seed_input_manifest.json')['files'].items():
        if hashlib.sha256((case / name).read_bytes()).hexdigest() != digest:
            raise ValueError('seed input changed: ' + name)
    workload = load_workload(case / 'workload.json')
    seed_data = read(case / 'seed_attempts.json')
    if read(case / 'source_identity.json')['workload'] != workload.fingerprint:
        raise ValueError('seed workload identity mismatch')
    architectures = enumerate_single_array_architectures_v3()
    if args.fixed_architecture:
        architectures = tuple(c for c in architectures if c.id == args.fixed_architecture)
        if not architectures:
            raise ValueError('unknown architecture')
    lookup = {c.id: c for c in architectures}
    old_reference = seed_data['bootstrap'][0]
    selected = min(seed_data['bootstrap'], key=lambda r: objective_cost(r['metrics'], old_reference['metrics'], args))
    initial = DesignState.from_dict(selected['state'])
    if initial.architecture_candidate_id not in lookup:
        raise ValueError('fixed architecture must match the supplied seed')
    common = read(out / 'common.json')
    if not common:
        tick = time.perf_counter()
        pool = SingleArrayOraclePool(workload, out / 'common_worker', deadline=time.monotonic() + args.eval_timeout * 3 + 30,
                                     evaluation_timeout_s=args.eval_timeout)
        try:
            reference_state = DesignState.from_dict(old_reference['state'])
            c = lookup[initial.architecture_candidate_id]
            reference_candidate = next(c for c in enumerate_single_array_architectures_v3() if c.id == reference_state.architecture_candidate_id)
            reference = pool.evaluate(reference_candidate, reference_state.mapping, phase='common_reference')['metrics']
            initial_metrics = pool.evaluate(c, initial.mapping, phase='common_tangram_best')['metrics']
            for actual, expected in ((reference, old_reference['metrics']), (initial_metrics, selected['metrics'])):
                for key in ('latency_s', 'energy_j', 'power_w', 'area_mm2', 'executed_macs', 'dram_read_bytes',
                            'dram_write_bytes', 'noc_hop_bytes', 'nop_hop_bytes'):
                    if not math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=1e-18):
                        raise ValueError('physical oracle equivalence failed: ' + key)
            if not initial_metrics['feasible']:
                raise ValueError('common best seed must be executable')
            common = dict(initial_state=initial.to_dict(), metrics=initial_metrics, reference=reference,
                          full_ppa=pool.full_evaluations, wall_s=time.perf_counter() - tick,
                          seed_cost=objective_cost(initial_metrics, reference, args), original_seed_source=str(case),
                          seed_generation='archived capacity-screened Tangram-style DP/beam; historical generation time excluded and provenance preserved',
                          equivalence='original physical metrics rel_tol=1e-12')
            write(out / 'common.json', common)
        finally:
            pool.close()
    initial, reference = DesignState.from_dict(common['initial_state']), common['reference']
    initial_metrics, seed_cost = common['metrics'], common['seed_cost']
    space = FullDesignSpace(workload, architectures, shortlist=args.shortlist, max_depth=args.max_fusion_depth,
                           inner=not args.no_inner, composites=not args.no_composites,
                           guidance=args.guidance, fusion=not args.fusion_off, legacy_generator=args.legacy_generator)
    write(out / 'design_space.json', space.domain())
    # Architecture enumeration is a shared outer phase, paid once physically
    # and charged identically to every comparator. No layout is pruned from
    # the subsequent RL domain. Its gains are reported before policy gains.
    screening = read(out / 'screening.json', dict(rows=[], wall_s=0., completed=False))
    if args.architecture_screen == 'all' and not screening['completed']:
        tick = time.perf_counter()
        screen_pool = SingleArrayOraclePool(workload, out / 'screen_worker',
            deadline=time.monotonic() + max(.01, args.screen_seconds - screening['wall_s']), evaluation_timeout_s=args.eval_timeout)
        try:
            done = {r['architecture'] for r in screening['rows']}
            representatives = []
            for count in sorted({c.hardware.compute_count for c in architectures}):
                representatives.append(min((c for c in architectures if c.hardware.compute_count == count),
                    key=lambda c: (abs(c.hardware.rows - c.hardware.cols), c.hardware.rows)))
            first_ids = {c.id for c in representatives}
            screen_order = [*representatives, *(c for c in architectures if c.id not in first_ids)]
            for target in screen_order:
                if target.id in done:
                    continue
                if (out / 'STOP_REQUESTED').exists():
                    screening['wall_s'] += time.perf_counter() - tick
                    write(out / 'screening.json', screening)
                    write(out / 'status.json', dict(status='interrupted', phase='architecture_screen'))
                    return 130
                if common['full_ppa'] + screen_pool.full_evaluations >= args.queries - 4:
                    break
                if time.monotonic() >= screen_pool.deadline:
                    break
                moved = space.architecture_trial(initial, target)
                t = space.bank(moved.state).build((8, 'balanced', 'spread', 'interleave', 'auto'), base=moved.state)
                trial = DesignTrial(t.state, 'architecture_mapping', t.targets,
                    dict(target=target.id, screening_mapping=(8, 'balanced', 'spread', 'interleave', 'auto')))
                write(out / 'status.json', dict(status='running', phase='architecture_screen', architecture=target.id,
                    completed=len(done), domain=len(architectures)))
                row = dict(architecture=target.id, trial=trial.to_dict(), features=space.features(initial, initial_metrics, reference, [trial])[0].tolist())
                try:
                    metrics = screen_pool.evaluate(target, trial.state.mapping, phase='shared_architecture_screen')['metrics']
                    value = objective_cost(metrics, reference, args)
                    row.update(status='full_completed', metrics=metrics, cost=value,
                               utility=10 * math.log(seed_cost / value) if value is not None else None)
                except (ValueError, RuntimeError, TimeoutError) as error:
                    row.update(status='timeout' if isinstance(error, TimeoutError) else 'compiler_or_worker_reject', reason=str(error), cost=None)
                screening['rows'].append(row)
                done.add(target.id)
                write(out / 'screening.json', {**screening, 'wall_s': screening['wall_s'] + time.perf_counter() - tick})
                print(json.dumps(dict(phase='shared_architecture_screen', completed=len(done), target=target.id,
                    cost=row['cost'], status=row['status'], full_ppa=screen_pool.full_evaluations)), flush=True)
            screening.update(completed=True, full_ppa=screen_pool.full_evaluations,
                wall_s=screening['wall_s'] + time.perf_counter() - tick,
                complete_domain_attempted=len(done) == len(architectures),
                note='all architecture candidates remain reachable after this bounded shared screen')
            write(out / 'screening.json', screening)
        finally:
            screen_pool.close()
    start_state, start_m, start_cost = initial, initial_metrics, seed_cost
    for row in screening['rows']:
        if row['cost'] is not None and row['cost'] < start_cost:
            start_state, start_m, start_cost = DesignTrial.from_dict(row['trial']).state, row['metrics'], row['cost']
    common_charge = common['full_ppa'] + screening.get('full_ppa', 0)
    common_wall = common['wall_s'] + screening['wall_s']
    write(out / 'shared_start.json', dict(state=start_state.to_dict(), metrics=start_m, cost=start_cost,
        original_seed_cost=seed_cost, full_ppa=common_charge, wall_s=common_wall,
        gain_before_rl_pct=100 * (1 - start_cost / seed_cost),
        screen_domain_attempted=screening.get('complete_domain_attempted', False)))
    summaries = read(out / 'summary.json', {})
    control = dict(status='running', pid=os.getpid(), source_identity=manifest['identity'])
    write(out / 'status.json', control)
    for method in args.methods.split(','):
        if method in summaries:
            continue
        # Equal proposal-cache startup conditions for each comparator. These
        # caches only contain pure descriptions, never another arm's labels.
        space = FullDesignSpace(workload, architectures, shortlist=args.shortlist, max_depth=args.max_fusion_depth,
            inner=not args.no_inner, composites=not args.no_composites, guidance=args.guidance,
            fusion=not args.fusion_off, legacy_generator=args.legacy_generator)
        folder = out / method
        folder.mkdir(exist_ok=True)
        saved = read(folder / 'checkpoint.json', {})
        current = DesignState.from_dict(saved['current']) if saved else start_state
        current_m = saved.get('current_metrics', start_m)
        elite = DesignState.from_dict(saved['elite']) if saved else start_state
        elite_m, elite_cost = saved.get('elite_metrics', start_m), saved.get('elite_cost', start_cost)
        feasible_saved = saved.get('best_feasible')
        feasible_state = DesignState.from_dict(feasible_saved['state']) if feasible_saved else (
            initial if not physical_violation(initial_metrics, args) else None)
        feasible_m = feasible_saved['metrics'] if feasible_saved else (initial_metrics if feasible_state else None)
        feasible_cost = feasible_saved['cost'] if feasible_saved else (seed_cost if feasible_state else None)
        if not saved:
            for row in screening['rows']:
                if row['cost'] is not None and not physical_violation(row['metrics'], args) and (feasible_cost is None or row['cost'] < feasible_cost):
                    feasible_state, feasible_m, feasible_cost = DesignTrial.from_dict(row['trial']).state, row['metrics'], row['cost']
        screen_states = {DesignTrial.from_dict(r['trial']).fingerprint for r in screening['rows']}
        visited = set(saved.get('visited', ())) | {initial.fingerprint} | screen_states
        steps, elapsed = saved.get('steps', 0), saved.get('elapsed_s', 0.)
        episode = saved.get('episode', 0)
        epoch_steps = saved.get('epoch_steps', 0)
        coverage = saved.get('coverage', {r['architecture']: dict(status=r['status'], shared_screen=True) for r in screening['rows']})
        best_step = saved.get('best_step', 0)
        rng, accept_rng = random.Random(args.seed), random.Random(args.seed ^ 0x5A17)
        if saved:
            rng.setstate(_rng_tuple(saved['rng']))
            accept_rng.setstate(_rng_tuple(saved['accept_rng']))
        space.calls = saved.get('generator_calls', 0)
        cached_trials = [DesignTrial.from_dict(t) for t in saved.get('candidate_set', ())]
        policy_revision = saved.get('policy_revision', 0)
        policy_history = saved.get('policy_history', [])
        # Feature dimension is declarative, independent of which action family
        # happens to be present at startup.
        started = time.perf_counter()
        size = space.features(initial, initial_metrics, reference, [DesignTrial(initial, 'A2')]).shape[1]
        policy = MultiStepQ(size, args.seed, gamma=args.gamma)
        if saved.get('policy_file'):
            policy.load(folder / saved['policy_file'])
        elif method == 'rl':
            policy.save(folder / 'policy_initial.pt')
            for row in screening['rows']:
                if row.get('utility') is not None:
                    policy.observe(row['features'], row['utility'], np.empty((0, size), np.float32), True, args.replay_updates)
            write(folder / 'shared_warmstart.json', dict(physical_samples=sum(r.get('utility') is not None for r in screening['rows']),
                policy_updates=policy.updates, charged_shared_full_ppa=screening.get('full_ppa', 0),
                scope='terminal physical improvement labels from the identical shared architecture screen; zero new PPA'))
            policy.save(folder / 'policy_after_shared.pt')
        # Persist pending observed transitions before drawing successors. On a
        # stop, their Bellman target is filled from the identical saved state.
        pending = saved.get('pending_transition')
        deadline = time.monotonic() + max(.01, args.seconds - elapsed - (time.perf_counter() - started))
        pool = SingleArrayOraclePool(workload, folder / 'worker', deadline=deadline, evaluation_timeout_s=args.eval_timeout)
        control.update(method=method, status='running')
        write(out / 'status.json', control)
        # Recover complete orphan results; never invent a missing RL update.
        for item in pool.completed_evidence():
            state = replace(initial, architecture_candidate_id=item['architecture'], mapping=plan_from_dict(item['mapping']))
            value = objective_cost(item['metrics'], reference, args)
            if value is not None and value < elite_cost:
                elite, elite_m, elite_cost = state, item['metrics'], value
            if value is not None and not physical_violation(item['metrics'], args) and (feasible_cost is None or value < feasible_cost):
                feasible_state, feasible_m, feasible_cost = state, item['metrics'], value
            visited.add(state.fingerprint)

        def save(status):
            nonlocal policy_revision, policy_history
            name = None
            obsolete = []
            if method == 'rl':
                policy_revision += 1
                name = f'policy_generation_{policy_revision:06d}.pt'
                target = folder / name
                temporary = target.with_suffix('.tmp')
                policy.save(temporary)
                temporary.replace(target)
                history = [name, *policy_history]
                obsolete, policy_history = history[2:], history[:2]
            write(folder / 'checkpoint.json', dict(version=1, status=status, current=current.to_dict(), current_metrics=current_m,
                elite=elite.to_dict(), elite_metrics=elite_m, elite_cost=elite_cost, visited=sorted(visited), steps=steps,
                epoch_steps=epoch_steps, episode=episode, rng=rng.getstate(), accept_rng=accept_rng.getstate(),
                generator_calls=space.calls, elapsed_s=elapsed + time.perf_counter() - started, policy_file=name,
                policy_revision=policy_revision, policy_history=policy_history,
                policy_updates=policy.updates, pending_transition=pending, coverage=coverage, best_step=best_step,
                candidate_set=[t.to_dict() for t in cached_trials], source_identity=manifest['identity'],
                best_feasible=dict(state=feasible_state.to_dict(), metrics=feasible_m, cost=feasible_cost) if feasible_state else None))
            # Only explicitly recorded generations created by this runner are
            # retired, after the new checkpoint is durable. Initial Q remains.
            for old in obsolete:
                if Path(old).name == old and old.startswith('policy_generation_'):
                    (folder / old).unlink(missing_ok=True)
            write(folder / 'recommendation.json', dict(state=feasible_state.to_dict() if feasible_state else None,
                metrics=feasible_m, cost=feasible_cost, strict_feasible=feasible_state is not None,
                closest=dict(state=elite.to_dict(), metrics=elite_m, penalized_cost=elite_cost), source_identity=manifest['identity']))

        stop_reason = 'full_query_budget_exhausted'
        try:
            while common_charge + pool.full_evaluations < args.queries and steps < args.queries * 8:
                if (out / 'STOP_REQUESTED').exists():
                    stop_reason = 'interrupted'
                    break
                if time.monotonic() >= deadline:
                    stop_reason = 'wall_time_budget_exhausted'
                    break
                progress = epoch_steps / args.horizon
                # A complete all-layout screen is independently selectable.
                # Each point is a charged oracle action, shared by all methods.
                phase = 'uniform_warmup' if pool.full_evaluations < args.warmup else 'learning_search'
                cached_trials = [t for t in cached_trials if t.fingerprint not in visited]
                if not cached_trials:
                    cached_trials, generation = space.propose(current, current_m, rng, visited)
                    cached_trials = list(cached_trials)
                    append(folder / 'proposal_ledger.jsonl', dict(step=steps + 1, **generation))
                if not cached_trials:
                    stop_reason = 'proposal_stalled'
                    break
                features = space.features(current, current_m, reference, cached_trials, progress)
                if pending and method == 'rl':
                    update_before = policy.updates
                    policy.observe(pending['features'], pending['reward'], features,
                                   terminal=pending['terminal'], updates=args.replay_updates)
                    append(folder / 'learning.jsonl', dict(transition_step=pending['step'], before_decision=steps + 1,
                        reward=pending['reward'], terminal=pending['terminal'], successor_count=len(features),
                        updates_before=update_before, updates_after=policy.updates,
                        gradient_norm=policy.last_gradient_norm, loss=float(np.mean(policy.losses[-args.replay_updates:]))))
                    pending = None
                epsilon = max(.10, .45 * (1 - pool.full_evaluations / max(1, args.queries - common_charge)))
                if method == 'rl' and phase == 'learning_search':
                    probabilities, q = policy.probabilities(features, epsilon)
                else:
                    probabilities, q = np.ones(len(features)) / len(features), np.zeros(len(features))
                chosen = rng.choices(range(len(features)), probabilities, k=1)[0]
                trial = cached_trials[chosen]
                before = objective_cost(current_m, reference, args)
                updates_before = policy.updates
                request = dict(step=steps + 1, phase=phase, method=method, state_hash=trial.fingerprint,
                    before_state_hash=current.fingerprint, action=trial.to_dict(), features=features[chosen].tolist(),
                    behavior_probability=float(probabilities[chosen]), probabilities=probabilities.tolist(),
                    q_values=q.tolist(), selected_index=chosen, candidate_hashes=[t.fingerprint for t in cached_trials],
                    episode=episode, epoch_steps=epoch_steps, policy_updates_before=updates_before,
                    before_cost=before, source_identity=manifest['identity'])
                write(folder / 'pending_trial.json', request)
                full_before = pool.full_evaluations
                tick = time.perf_counter()
                steps += 1
                visited.add(trial.fingerprint)
                if phase == 'architecture_screen':
                    coverage[trial.state.architecture_candidate_id] = dict(status='attempted', step=steps)
                try:
                    validate_state(workload, trial.state, lookup)
                    result = pool.evaluate(lookup[trial.state.architecture_candidate_id], trial.state.mapping, phase=phase)
                    metrics = result['metrics']
                    value = objective_cost(metrics, reference, args)
                    valid = value is not None
                    improved = valid and value < elite_cost
                    if improved:
                        elite, elite_m, elite_cost, best_step = trial.state, metrics, value, steps
                    if valid and not physical_violation(metrics, args) and (feasible_cost is None or value < feasible_cost):
                        feasible_state, feasible_m, feasible_cost = trial.state, metrics, value
                    write(folder / 'evaluations' / (trial.fingerprint + '.json'), dict(state=trial.state.to_dict(),
                        metrics=metrics, cost=value, step=steps, full_completed=pool.full_evaluations > full_before))
                    reward = 10 * math.log(before / value) if valid else None
                    accepted = False
                    terminal = False
                    if valid:
                        if phase in ('architecture_screen', 'uniform_warmup') or method == 'random':
                            current, current_m = elite, elite_m
                            accepted, terminal = improved, True
                        elif method == 'sa':
                            temperature = max(.002, .08 * (1 - pool.full_evaluations / max(1, args.queries - common_charge)))
                            accepted = value <= before or accept_rng.random() < math.exp(min(0, (before - value) / (temperature * seed_cost)))
                            if accepted:
                                current, current_m = trial.state, metrics
                        else:
                            # RL follows every executable trial, including a
                            # temporarily worse state. Archive remains separate.
                            current, current_m, accepted = trial.state, metrics, True
                            epoch_steps += 1
                            terminal = epoch_steps >= args.horizon
                            if terminal:
                                current, current_m, epoch_steps = elite, elite_m, 0
                                episode += 1
                        if method == 'rl' and pool.full_evaluations > full_before:
                            pending = dict(step=steps, features=features[chosen].tolist(), reward=reward, terminal=terminal)
                    coverage[trial.state.architecture_candidate_id] = dict(status='full_evaluated', feasible=valid, step=steps)
                    outcome = dict(status='full_completed', feasible=valid, cost=value, reward=reward,
                                   improved_global_best=improved, accepted=accepted, terminal=terminal, metrics=compact(metrics))
                except (ValueError, RuntimeError, TimeoutError) as error:
                    outcome = dict(status='timeout' if isinstance(error, TimeoutError) else 'compiler_or_worker_reject',
                        reason=str(error), feasible=None, cost=None, reward=None, improved_global_best=False, accepted=False)
                # Generate successor candidates only once per accepted state.
                # Rejected/invalid trials reuse the untried set at the same base.
                cached_trials = [] if outcome['accepted'] or phase == 'architecture_screen' else [t for t in cached_trials if t.fingerprint not in visited]
                event = {**request, **outcome, 'best_cost': elite_cost, 'new_full_ppa': pool.full_evaluations - full_before,
                    'charged_full_ppa': common_charge + pool.full_evaluations, 'evaluation_wall_s': time.perf_counter() - tick,
                    'search_wall_s': elapsed + time.perf_counter() - started, 'policy_updates_after': policy.updates,
                    'gradient_norm': policy.last_gradient_norm, 'loss': float(np.mean(policy.losses[-args.replay_updates:])) if policy.losses else None}
                append(folder / 'events.jsonl', event)
                save('running')
                print(json.dumps(dict(method=method, step=steps, full_ppa=event['charged_full_ppa'],
                    gain_vs_seed_pct=100 * (1 - elite_cost / seed_cost), architecture=elite.architecture_candidate_id,
                    fusion_depth=max((len(s.layer_ids) for s in elite.mapping.fusion_stacks), default=0),
                    policy_updates=policy.updates, last_status=outcome['status'])), flush=True)
            if pending and method == 'rl' and stop_reason != 'interrupted':
                # End-of-budget truncation has no executable successor within
                # this experiment. Never bootstrap from unmeasured states.
                update_before = policy.updates
                policy.observe(pending['features'], pending['reward'], np.empty((0, size), np.float32), True, args.replay_updates)
                append(folder / 'learning.jsonl', dict(transition_step=pending['step'], before_decision=None,
                    reward=pending['reward'], terminal=True, successor_count=0, budget_truncation=True,
                    updates_before=update_before, updates_after=policy.updates,
                    gradient_norm=policy.last_gradient_norm, loss=float(np.mean(policy.losses[-args.replay_updates:]))))
                pending = None
            if steps >= args.queries * 8:
                stop_reason = 'attempt_budget_exhausted'
            save(stop_reason)
        finally:
            pool.close()
        if stop_reason == 'interrupted':
            control.update(status='interrupted', method=method)
            write(out / 'status.json', control)
            return 130
        summary = dict(method=method, status=stop_reason, common_full_ppa=common_charge, search_full_ppa=pool.full_evaluations,
            full_ppa=common_charge + pool.full_evaluations, seed_cost=seed_cost, best_cost=elite_cost,
            gain_vs_seed_pct=100 * (1 - elite_cost / seed_cost), best=compact(elite_m), initial=compact(initial_metrics),
            best_step=best_step, best_state_hash=elite.fingerprint, best_architecture=elite.architecture_candidate_id,
            best_fusion_depth=max((len(s.layer_ids) for s in elite.mapping.fusion_stacks), default=0),
            policy_updates=policy.updates, search_wall_s=elapsed + time.perf_counter() - started,
            charged_wall_s=common_wall + elapsed + time.perf_counter() - started, architecture_coverage=coverage,
            gain_after_shared_screen_pct=100 * (1 - elite_cost / start_cost),
            gain_before_rl_pct=100 * (1 - start_cost / seed_cost),
            latency_improvement_pct=100 * (1 - elite_m['latency_s'] / initial_metrics['latency_s']),
            energy_improvement_pct=100 * (1 - elite_m['energy_j'] / initial_metrics['energy_j']),
            area_improvement_pct=100 * (1 - elite_m['area_mm2'] / initial_metrics['area_mm2']),
            objective=args.objective, gamma=args.gamma, domain=space.version)
        summary['strict_feasible'] = feasible_state is not None
        summary['recommended_cost'] = feasible_cost
        summary['note'] = 'best_* and gain fields describe penalized search elite; recommendation.json contains only strict-feasible designs'
        write(folder / 'summary.json', summary)
        summaries[method] = summary
        write(out / 'summary.json', summaries)
        report(out, summaries)
    control.update(status='completed', source_stable=sources(APP) == manifest['files'])
    write(out / 'status.json', control)
    return 0


def report(out, summaries):
    lines = ['# INMIN full-system DSE', '', 'Complete schema3 analytical oracle; fixed reference and original Tangram-style seed. All methods use the same design domain.', '',
        '|Method|Full PPA|Cost gain %|Latency gain %|Energy gain %|Area gain %|Architecture|Fusion depth|SGD|Charged s|',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    keys = ('method', 'full_ppa', 'gain_vs_seed_pct', 'latency_improvement_pct', 'energy_improvement_pct', 'area_improvement_pct',
            'best_architecture', 'best_fusion_depth', 'policy_updates', 'charged_wall_s')
    for row in summaries.values():
        lines.append('|' + '|'.join(str(row[k]) for k in keys) + '|')
    lines += ['', 'RL: multi-step action-conditioned Double Q with target network and physical improvement reward. Random/SA are controls on this same oracle, not original Gemini/Tangram binaries.',
              'SGD, proposal proxies, cache hits and timeouts are not full PPA. Capacity-invalid completed schedules count as full PPA. No global optimum or silicon accuracy is claimed.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-case', default=str(APP.parent / 'inputs/replay_v04/resnet18'))
    p.add_argument('--out', required=True)
    p.add_argument('--methods', default='rl,sa,random')
    p.add_argument('--queries', type=int, default=512)
    p.add_argument('--seconds', type=float, default=7200)
    p.add_argument('--eval-timeout', type=float, default=300)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--shortlist', type=int, default=64)
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--gamma', type=float, default=.9)
    p.add_argument('--replay-updates', type=int, default=24)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--max-fusion-depth', type=int, default=4)
    p.add_argument('--objective', choices=('balanced', 'edp', 'latency', 'energy'), default='edp')
    p.add_argument('--max-power', type=float)
    p.add_argument('--max-area', type=float)
    p.add_argument('--max-latency-ms', type=float)
    p.add_argument('--architecture-screen', choices=('all', 'off'), default='all')
    p.add_argument('--screen-seconds', type=float, default=3600)
    p.add_argument('--fixed-architecture')
    p.add_argument('--no-inner', action='store_true')
    p.add_argument('--no-composites', action='store_true')
    p.add_argument('--legacy-generator', action='store_true')
    p.add_argument('--guidance', action='store_true')
    p.add_argument('--fusion-off', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--frozen-child', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.queries, args.shortlist) < 4 or min(args.seconds, args.eval_timeout, args.horizon, args.replay_updates) <= 0:
        p.error('positive budgets/horizon/updates and queries/shortlist >=4 required')
    if not 0 <= args.gamma < 1 or args.warmup < 0 or args.max_fusion_depth < 2:
        p.error('gamma in [0,1), warmup>=0 and fusion depth>=2 required')
    if any(v is not None and (v <= 0 or not math.isfinite(v)) for v in (args.max_power, args.max_area, args.max_latency_ms)):
        p.error('physical limits must be positive finite values')
    if any(m not in ('rl', 'sa', 'random') for m in args.methods.split(',')) or len(set(args.methods.split(','))) != len(args.methods.split(',')):
        p.error('methods must be distinct rl,sa,random')
    args.out, args.seed_case = str(Path(args.out).resolve()), str(Path(args.seed_case).resolve())
    if args.frozen_child:
        return execute(args)
    out = Path(args.out)
    with lock(APP.parent / 'output/.local_rl_diagnostics.lock'), lock(out / 'run.lock'):
        if args.resume:
            saved = read(out / 'effective_config.json')
            if not saved:
                raise ValueError('no saved experiment to resume')
            for k, v in saved.items():
                setattr(args, k, v)
            args.resume = True
        else:
            if any(f.name != 'run.lock' for f in out.iterdir()):
                raise ValueError('output not empty; choose a new output or --resume')
            inputs, hashes = out / 'seed_input_snapshot', {}
            for name in ('workload.json', 'seed_attempts.json', 'source_identity.json'):
                source, target = Path(args.seed_case) / name, inputs / name
                target.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(source, target)
                hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
            write(out / 'seed_input_manifest.json', dict(original_case=args.seed_case, files=hashes))
            args.seed_case = str(inputs)
            files = sources(APP)
            for rel in files:
                target = out / 'source_snapshot' / rel
                target.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(APP / rel, target)
            write(out / 'source_snapshot_manifest.json', dict(identity=sha(files), files=files))
            write(out / 'environment.json', environment())
            write(out / 'effective_config.json', {k: v for k, v in vars(args).items() if k not in ('resume', 'frozen_child')})
        manifest = read(out / 'source_snapshot_manifest.json')
        if sources(out / 'source_snapshot') != manifest['files'] or environment() != read(out / 'environment.json'):
            raise ValueError('frozen source/environment changed')
        if args.resume and read(out / 'status.json', {}).get('status') == 'completed':
            print(json.dumps(dict(status='already_completed', additional_ppa=0)))
            return 0
        write(out / 'status.json', dict(status='starting', return_code=None))
        (out / 'STOP_REQUESTED').unlink(missing_ok=True)
        command = [sys.executable, '-u', '-X', 'utf8', str(out / 'source_snapshot/full_system_dse.py'), '--frozen-child']
        for key, value in vars(args).items():
            if key in ('resume', 'frozen_child') or value is None:
                continue
            flag = '--' + key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend((flag, str(value)))
        with (out / 'runner.log').open('ab') as log:
            tick, began = time.perf_counter(), time.time()
            child = subprocess.Popen(command, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            print(json.dumps(dict(pid=child.pid, out=str(out), log=str(out / 'runner.log'))), flush=True)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                (out / 'STOP_REQUESTED').touch()
                code = child.wait()
            status = read(out / 'status.json', {})
            status['return_code'] = code
            if code and status.get('status') != 'interrupted':
                status.update(status='failed', return_code=code)
            write(out / 'status.json', status)
            append(out / 'parent_invocations.jsonl', dict(started_unix=began, finished_unix=time.time(), return_code=code,
                wall_s=time.perf_counter() - tick, scope='child startup + common + search + cleanup; source copy and historical seed generation separate'))
            return code


if __name__ == '__main__':
    raise SystemExit(main())
