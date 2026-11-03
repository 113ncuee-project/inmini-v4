"""Finite-domain toy joint mapping/DF oracle; no global-optimum claim."""
import argparse
from dataclasses import replace
from itertools import product
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simple_rapidchiplet.hybrid.auto_runtime import atomic_json
from simple_rapidchiplet.hybrid.mapping import FusionStack, LayerGroup, Partition, initial_mapping
from simple_rapidchiplet.hybrid.rl_search import JointObjective
from simple_rapidchiplet.hybrid.single_array_evaluator import SingleArrayEvaluator
from simple_rapidchiplet.hybrid.single_array_hardware import SingleArrayHardware
from simple_rapidchiplet.hybrid.workload import make_toy_workload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if (out / 'summary.json').exists():
        raise ValueError('completed finite oracle exists; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    w, h = make_toy_workload(batch=1, channels=5, spatial=7), SingleArrayHardware(1, 1)
    oracle = SingleArrayEvaluator(w, h)
    baseline = initial_mapping(w, h.compute_slots, ('memory:0', 'memory:1'), max_group_layers=1)
    reference = oracle(baseline).metrics
    objective, rows, paired = JointObjective(), [], []
    best = [objective.cost(reference, reference), baseline, oracle(baseline)]
    seen, rejects = set(), []
    for mode, quota, direction, placement, memory in product(('temporal', 'spatial'), (1, 2, 4), ('K', 'P'), (0, 1), ('memory:0', 'interleave')):
        group = LayerGroup('closed_toy', tuple(layer.id for layer in w.layers), mode)
        order = h.compute_slots if not placement else tuple(h.compute_slots[index] for index in (0, 15, 3, 12, 1, 14, 2, 13, 4, 11, 7, 8, 5, 10, 6, 9))
        mappings = []
        for index, item in enumerate(baseline.layer_mappings):
            offset = index * quota if mode == 'spatial' else 0
            part = Partition(1, quota, 1, 1) if direction == 'K' else Partition(1, 1, 1, quota)
            mappings.append(replace(item, chiplets=order[offset:offset + quota], partition=part,
                                    input_dram=memory, weight_dram=memory, output_dram=memory))
        off_mapping = replace(baseline, groups=(group,), layer_mappings=tuple(mappings))
        off_hash = hashlib.sha256(json.dumps(off_mapping.to_dict(), sort_keys=True).encode()).hexdigest()
        configurations = [('off', None)] + [(cache, tile) for cache in ('recompute', 'horizontal_cache', 'full_cache') for tile in ((2, 2), (4, 4), (7, 7))]
        off_metrics = None
        for cache, tile in configurations:
            mapping = off_mapping if tile is None else replace(off_mapping, fusion_stacks=(FusionStack(group.layer_ids, tile, cache),))
            key = hashlib.sha256(json.dumps(mapping.to_dict(), sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            label = {'mode': mode, 'quota_per_layer': quota, 'partition': direction, 'placement': placement,
                     'memory': memory, 'fusion': cache, 'tile_hw': tile, 'state_hash': key, 'paired_off_state_hash': off_hash}
            try:
                result = oracle(mapping)
                m = result.metrics
                row = {**label, 'feasible': m['feasible'], 'cost': objective.cost(m, reference),
                       **{name: m[name] for name in ('latency_s', 'area_mm2', 'power_w', 'energy_j', 'executed_macs', 'recompute_macs',
                                                   'dram_read_bytes', 'dram_write_bytes', 'noc_hop_bytes', 'nop_hop_bytes', 'unit_count')}}
                rows.append(row)
                if cache == 'off':
                    off_metrics = row
                elif off_metrics is not None:
                    paired.append({**label, 'cost_delta_on_minus_off': row['cost'] - off_metrics['cost'],
                                   **{'delta_' + name: row[name] - off_metrics[name] for name in ('latency_s', 'energy_j',
                                      'executed_macs', 'dram_read_bytes', 'dram_write_bytes', 'noc_hop_bytes', 'unit_count')}})
                if m['feasible'] and row['cost'] < best[0]:
                    best = [row['cost'], mapping, result]
            except ValueError as error:
                rejects.append({**label, 'reason': str(error)})
            oracle.cache.clear()
    # Exact state identity pairs even when a duplicate off design was already
    # evaluated under an equivalent quota/direction/placement label.
    off_by_hash = {row['state_hash']: row for row in rows if row['fusion'] == 'off'}
    paired = []
    for row in rows:
        if row['fusion'] == 'off':
            continue
        off = off_by_hash[row['paired_off_state_hash']]
        paired.append({'on_state_hash': row['state_hash'], 'off_state_hash': off['state_hash'],
            'cost_delta_on_minus_off': row['cost'] - off['cost'],
            **{'delta_' + name: row[name] - off[name] for name in ('latency_s', 'energy_j', 'executed_macs',
                'dram_read_bytes', 'dram_write_bytes', 'noc_hop_bytes', 'unit_count')}})
    atomic_json(out / 'domain.json', {'hardware': h.to_dict(), 'workload': w.to_dict(),
        'mode': ['temporal', 'spatial'], 'quotas': [1, 2, 4], 'partition': ['K_only', 'W_only'], 'placement': ['contiguous', 'spread'],
        'memory': ['IO0', 'interleave'], 'fusion': ['off', 'recompute', 'horizontal_cache', 'full_cache'],
        'tiles': [[2, 2], [4, 4], [7, 7]], 'scope': 'finite enumerated domain, fixed one-compute/two-IO hardware; not global optimum'})
    atomic_json(out / 'results.json', rows)
    atomic_json(out / 'paired_fusion.json', paired)
    atomic_json(out / 'rejected.json', rejects)
    atomic_json(out / 'best_evaluation.json', best[2].to_dict())
    if rows:
        with (out / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {'full_evaluations': oracle.evaluations, 'unique_domain_states': len(seen), 'feasible_states': sum(row['feasible'] for row in rows),
               'rejected_states': len(rejects), 'best_cost': best[0], 'baseline_cost': 1.,
               'best_latency_us': best[2].metrics['latency_s'] * 1e6, 'baseline_latency_us': reference['latency_s'] * 1e6,
               'best_mapping': best[1].to_dict(), 'elapsed_s': time.monotonic() - started,
               'fusion_win_count': sum(row['cost_delta_on_minus_off'] < 0 for row in paired),
               'fusion_loss_count': sum(row['cost_delta_on_minus_off'] > 0 for row in paired),
               'source_hashes': oracle.sources, 'scope': 'finite-domain toy oracle, not full-model/RL effectiveness evidence'}
    summary['fusion_tie_count'] = sum(row['cost_delta_on_minus_off'] == 0 for row in paired)
    summary['paired_fusion_count'] = len(paired)
    from single_array_dse import source_identity
    from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures
    atomic_json(out / 'source_identity.json', source_identity(w, enumerate_single_array_architectures()))
    atomic_json(out / 'summary.json', summary)
    print({key: value for key, value in summary.items() if key not in ('best_mapping', 'source_hashes')})


if __name__ == '__main__':
    main()
