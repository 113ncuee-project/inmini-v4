"""Replay a verified Fusion state and remove only Fusion on frozen sources.

This is a separately charged two-query experiment. Search-on/search-off
results alone do not establish the effect at a fixed hardware and mapping.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seconds', type=float, default=700.)
    parser.add_argument('--eval-timeout', type=float, default=300.)
    args = parser.parse_args()
    snapshot, run, out = (Path(value).resolve() for value in (args.snapshot, args.run, args.out))
    sys.path.insert(0, str(snapshot))
    from simple_rapidchiplet.hybrid.auto_runtime import atomic_json
    from simple_rapidchiplet.hybrid.rl_schema import ActionPlan, DesignState
    from simple_rapidchiplet.hybrid.rl_actions import apply_transaction
    from simple_rapidchiplet.hybrid.rl_search import JointObjective
    from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures
    from simple_rapidchiplet.hybrid.single_array_runtime import SingleArrayOraclePool
    from simple_rapidchiplet.hybrid.workload import Workload, TensorSpec, LayerSpec
    from single_array_dse import exclusive_run, scalar_metrics, source_identity
    with exclusive_run(out):
        if (out / 'summary.json').exists():
            print('existing counterfactual retained: ' + str(out))
            return
        manifest = read(snapshot / 'snapshot_manifest.json')
        for relative, expected in manifest['files'].items():
            if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('frozen source changed: ' + relative)
        data = read(run / 'workload.json')
        tensors = tuple(TensorSpec(**{**item, 'shape': tuple(item['shape'])}) for item in data['tensors'])
        layers = []
        tuple_keys = ('inputs', 'kernel', 'stride', 'padding', 'dilation', 'weight_shape', 'post_ops')
        for item in data['layers']:
            layers.append(LayerSpec(**{key: tuple(value) if key in tuple_keys else value for key, value in item.items()}))
        workload = Workload(data['name'], tensors, tuple(layers), tuple(data['input_ids']), tuple(data['output_ids']),
                            data['batch_size'], data['source'])
        if workload.fingerprint != data['fingerprint']:
            raise ValueError('reconstructed workload fingerprint differs')
        from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
        candidates = enumerate_single_array_architectures_v3() if read(run / 'effective_config.json')['hardware_profile'].endswith('_3') else enumerate_single_array_architectures()
        lookup = {item.id: item for item in candidates}
        fixed = read(run / 'effective_config.json').get('fixed_architecture')
        if fixed:
            candidates = tuple(c for c in candidates if c.id == fixed)
            lookup = {item.id: item for item in candidates}
        identity = source_identity(workload, candidates)
        if identity['identity'] != read(run / 'source_identity.json')['identity']:
            raise ValueError('source identity differs from the original experiment')
        seed_data = read(run / 'seed_attempts.json')
        initial_states = {DesignState.from_dict(row['state']).fingerprint: DesignState.from_dict(row['state'])
                          for row in seed_data['bootstrap']}
        evidence, replay_errors = {}, []
        objective = JointObjective(**read(run / 'effective_config.json')['objective'])
        reference = seed_data['bootstrap'][0]['metrics']
        for checkpoint_path in sorted(run.glob('*/checkpoint.json')):
            checkpoint = read(checkpoint_path)
            reference = checkpoint['reference']
            states = dict(initial_states)
            for entry in checkpoint['architecture_archive'].values():
                state = DesignState.from_dict(entry['state'])
                states[state.fingerprint] = state
                if state.mapping.fusion_stacks and entry['metrics'].get('feasible'):
                    evidence[state.fingerprint] = (state, entry['metrics'], checkpoint_path.parent.name)
            for row in checkpoint['trace']:
                if not row['valid']:
                    continue
                before = states.get(row['before_state_hash'])
                if before is None:
                    replay_errors.append({'arm': checkpoint_path.parent.name, 'step': row['step'], 'reason': 'before state unavailable'})
                    continue
                try:
                    state = apply_transaction(workload, before, ActionPlan.from_dict(row['action_plan']), candidates)
                    if state.fingerprint != row['after_state_hash']:
                        raise ValueError('replayed after-state differs')
                    states[state.fingerprint] = state
                    if state.mapping.fusion_stacks and row['current_metrics'].get('feasible'):
                        evidence[state.fingerprint] = (state, row['current_metrics'], checkpoint_path.parent.name)
                except ValueError as error:
                    replay_errors.append({'arm': checkpoint_path.parent.name, 'step': row['step'], 'reason': str(error)})
        atomic_json(out / 'selection_evidence.json', {'verified_fusion_states': len(evidence), 'replay_errors': replay_errors,
                    'selection': 'lowest cost feasible verified Fusion state across stored arm traces/archives'})
        atomic_json(out / 'source_identity.json', identity)
        if not evidence:
            atomic_json(out / 'summary.json', {'status': 'not_run_no_verified_fusion_state', 'extra_full_ppa': 0,
                        'scope': 'not_run is not evidence that all Fusion is worse'})
            return
        state, original, arm = min(evidence.values(), key=lambda item: objective.cost(item[1], reference))
        off = replace(state.mapping, fusion_stacks=())
        atomic_json(out / 'selected_state.json', {'state': state.to_dict(), 'original_metrics': original, 'source_arm': arm,
                    'off_mapping': off.to_dict(), 'only_intervention': 'remove fusion_stacks; preserve hardware/groups/PEs/partition/DRAM/cache fields'})
        oracle_phase_started = time.monotonic()
        pool = SingleArrayOraclePool(workload, out / 'worker', deadline=started + args.seconds,
                                    evaluation_timeout_s=args.eval_timeout)
        metrics, errors = {}, []
        try:
            for label, mapping in [('on', state.mapping), ('off', off)]:
                try:
                    response = pool.evaluate(lookup[state.architecture_candidate_id], mapping, phase='fixed_mapping_counterfactual')
                    metrics[label] = response['metrics']
                    pool.persist(lookup[state.architecture_candidate_id], mapping, out / label, vars(args))
                except (ValueError, RuntimeError, TimeoutError) as error:
                    errors.append({'intervention': label, 'reason': str(error)})
        finally:
            pool.close()
        matched = all(metrics.get(label, {}).get('feasible') for label in ('on', 'off'))
        keys = ('latency_s', 'energy_j', 'power_w', 'area_mm2', 'executed_macs', 'recompute_macs',
                'dram_read_bytes', 'dram_write_bytes', 'noc_hop_bytes', 'nop_hop_bytes', 'l1_l2_service_bytes', 'unit_count')
        summary = {'status': 'completed_matched_pair' if matched else 'incomplete_or_infeasible_pair',
            'extra_full_ppa': pool.full_evaluations, 'elapsed_s': time.monotonic() - started,
            'elapsed_scope': 'end-to-end including selection/replay and oracle',
            'oracle_phase_elapsed_s': time.monotonic() - oracle_phase_started,
            'architecture': state.architecture_candidate_id, 'source_arm': arm, 'errors': errors,
            'metrics': {label: scalar_metrics(value) for label, value in metrics.items()},
            'delta_on_minus_off': {key: metrics['on'][key] - metrics['off'][key] for key in keys} if matched else None,
            'cost_delta_on_minus_off': objective.cost(metrics['on'], reference) - objective.cost(metrics['off'], reference) if matched else None,
            'scope': 'one selected fixed hardware/mapping pair; not a universal Fusion benefit or RL advantage'}
        if 'on' in metrics:
            summary['replayed_on_matches_original'] = all(abs(metrics['on'][key] - original[key]) <= max(1e-18, abs(original[key]) * 1e-12)
                                                         for key in ('latency_s', 'energy_j', 'area_mm2', 'power_w'))
        atomic_json(out / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
