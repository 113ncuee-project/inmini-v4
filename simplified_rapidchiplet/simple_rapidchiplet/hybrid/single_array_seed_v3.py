"""Capacity-screened segment/top-N seed search informed by nn_dataflow.

The supplied upstream scheduler caches layer schedules and retains top schemes
by ending layer. We apply that decomposition to INMIN's own physical contract;
upstream Eyeriss estimates are never substituted for the final oracle.
"""
from dataclasses import replace
from itertools import product
import math
import time
import json
from pathlib import Path
from .auto_runtime import atomic_json
from .search import plan_from_dict

from .mapping import LayerGroup, MappingPlan, initial_mapping, validate_plan, TensorRegion
from .single_array_seed import SingleArraySeedProposal, _segment
from .single_array_compiler import tile_requirements, _fit_full_channel_tile
from .rl_schema import stable_hash

VERSION = 'capacity_screened_segment_beam_v3'


def generate_seeds_v3(workload, hardware, *, max_candidates=4, beam_width=8,
                      max_group_layers=4, timeout_s=120., update=lambda **_: None, checkpoint=None):
    started = time.monotonic()
    memories = tuple(f'memory:{i}' for i in range(hardware.io_count))
    baseline = initial_mapping(workload, hardware.compute_slots, memories, max_group_layers=1)
    common = {'implementation': VERSION, 'reference_zip_sha256':
        '8fae65512bb780894d8cba5a0f9c6c66f40a9845178e468d435738bde028a57b',
        'upstream_original_engine_executed': False, 'global_optimum': False,
        'physical_validity': 'requires full INMIN oracle', 'beam_width': beam_width,
        'max_group_layers': max_group_layers, 'timeout_s': timeout_s}
    output = [SingleArraySeedProposal(baseline, {**common, 'source': 'preserved_original_integer_stripes'},
                                     {'formal_PPA': False})]
    # Cache an actual fitting PE fragment for each local layer/partition choice.
    local_cache, segment_cache = {}, {}
    counters = {'local_cache_hits': 0, 'local_cache_misses': 0, 'capacity_pruned_segments': 0,
                'prefix_transitions': 0, 'timed_out': False}
    beams = {0: [(0., (), ())]}
    identity = stable_hash({'workload': workload.fingerprint, 'hardware': hardware.fingerprint,
        'algorithm': VERSION, 'beam_width': beam_width, 'max_group_layers': max_group_layers, 'max_candidates': max_candidates})
    prefix = 0
    checkpoint = Path(checkpoint) if checkpoint else None
    if checkpoint and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError('seed DP checkpoint source/config/workload differs')
        beams = {}
        for key, entries in saved['beams'].items():
            beams[int(key)] = []
            for entry in entries:
                plan = plan_from_dict(entry['mapping'])
                beams[int(key)].append((entry['cost'], plan.groups, plan.layer_mappings))
        counters.update(saved['counters'])
        counters['timed_out'] = False
        prefix = saved['prefix_completed']
    for end in range(prefix + 1, len(workload.layers) + 1):
        rows = []
        for depth in range(1, min(max_group_layers, end) + 1):
            start = end - depth
            if not beams.get(start):
                continue
            layers = workload.layers[start:end]
            for mode, direction, flow in product(('temporal', 'spatial'), ('K', 'P'),
                                                ('output_stationary', 'weight_reuse', 'activation_reuse')):
                if time.monotonic() - started > timeout_s:
                    counters['timed_out'] = True
                    break
                key = (start, end, mode, direction, flow)
                if key not in segment_cache:
                    try:
                        group, mapping, proxy, traffic = _segment(workload, hardware, layers, mode=mode,
                            direction=direction, placement=int(direction == 'P'), memory_policy='interleave')
                        mapping = tuple(replace(item, dataflow=flow) for item in mapping)
                        # Conservative scratchpad screen for a non-fused group:
                        # retained whole outputs must fit; no implicit DRAM spill.
                        outputs = {}
                        for item in mapping:
                            layer = workload.layer_map[item.layer_id]
                            tensor = workload.tensor_map[layer.output]
                            dies = {slot.split('/')[0] for slot in item.chiplets}
                            for die in dies:
                                share = sum(slot.startswith(die + '/') for slot in item.chiplets) / len(item.chiplets)
                                outputs[die] = outputs.get(die, 0) + math.ceil(tensor.bytes * share)
                        if depth > 1 and max(outputs.values(), default=0) > .75 * hardware.l2_output_bytes:
                            counters['capacity_pruned_segments'] += 1
                            segment_cache[key] = None
                            continue
                        local = 0.
                        for item in mapping:
                            local_key = (item.layer_id, item.partition.as_tuple(), flow)
                            if local_key in local_cache:
                                counters['local_cache_hits'] += 1
                            else:
                                counters['local_cache_misses'] += 1
                                layer = workload.layer_map[item.layer_id]
                                tensor = workload.tensor_map[layer.output]
                                # Bound the largest shard, then select a real
                                # fitting fragment. This remains a ranking model.
                                shape = tuple(math.ceil(size / split) for size, split in zip(tensor.shape, item.partition.as_tuple()))
                                region = TensorRegion(tuple((0, size) for size in shape))
                                tile = _fit_full_channel_tile(workload, layer, region, hardware)
                                req = tile_requirements(workload, layer, TensorRegion(tuple((0, size) for size in tile)), hardware.assumptions.psum_storage_bytes)
                                fragments = math.prod(math.ceil(a / b) for a, b in zip(shape, tile))
                                macs = tensor.elements * math.prod(layer.weight_shape[1:]) if layer.weight_shape else 0
                                c = layer.weight_shape[1] if layer.weight_shape else 0
                                kw = math.prod(layer.kernel)
                                positions = tile[0] * tile[2] * tile[3]
                                blocks = math.ceil(tile[1] / 16)
                                wread = req['weight_bytes'] * (1 if flow == 'weight_reuse' else positions)
                                aread = positions * c * kw * (1 if flow == 'activation_reuse' else blocks)
                                psum = (max(0, 2 * math.ceil(c / 16) * kw - 2) * req['psum_bytes']
                                        if flow != 'output_stationary' else 0)
                                local_cache[local_key] = (macs / (len(item.chiplets) * 256 * hardware.compute_clock_hz)
                                    + fragments * (wread + aread + psum) / hardware.l1_bandwidth_Bps
                                    + fragments * (req['activation_bytes'] + req['weight_bytes'] + req['output_bytes'] + req['psum_bytes']) / hardware.l2_bandwidth_Bps)
                            local += local_cache[local_key]
                        segment_cache[key] = (group, mapping, local + traffic / (hardware.io_count * hardware.dram_bandwidth_Bps))
                    except ValueError:
                        counters['capacity_pruned_segments'] += 1
                        segment_cache[key] = None
                segment = segment_cache[key]
                if segment is None:
                    continue
                group, mapping, value = segment
                for old, groups, mappings in beams[start]:
                    counters['prefix_transitions'] += 1
                    rows.append((old + value, (*groups, group), (*mappings, *mapping)))
            if counters['timed_out']:
                break
        if counters['timed_out']:
            break
        # Top-N with a small flow-diversity reservation prevents all beam
        # entries from collapsing to equivalent expensive grouping choices.
        ranked = sorted(rows, key=lambda row: row[0])
        kept, seen = [], set()
        for row in ranked:
            key = stable_hash({'groups': [(g.layer_ids, g.mode) for g in row[1]],
                               'mapping': [(m.layer_id, m.partition.as_tuple(), m.dataflow) for m in row[2]]})
            if key not in seen:
                seen.add(key)
                kept.append(row)
            if len(kept) == beam_width:
                break
        beams[end] = kept
        prefix = end
        if checkpoint:
            retained = {key: entries for key, entries in beams.items() if key > end - max_group_layers}
            atomic_json(checkpoint, {'identity': identity, 'prefix_completed': end, 'counters': counters,
                'beams': {str(key): [{'cost': cost, 'mapping': MappingPlan(groups, mapping, 1, 'multicast', memories).to_dict()}
                            for cost, groups, mapping in entries] for key, entries in retained.items()}})
        update(seed_prefix_completed=end, seed_layer_count=len(workload.layers))
    for value, groups, mapping in beams.get(len(workload.layers), ()):
        plan = MappingPlan(groups, mapping, 1, 'multicast', memories)
        validate_plan(workload, plan, hardware.compute_slots)
        if plan.to_dict() == baseline.to_dict():
            continue
        output.append(SingleArraySeedProposal(plan, {**common, **counters, 'source': 'capacity_screened_segment_beam',
            'elapsed_s': time.monotonic() - started}, {'rank_cost_s': value, 'formal_PPA': False,
            'capacity_screen': 'conservative whole-output bound for multi-layer groups + fitting physical PE fragment; final lifetime oracle still required'}))
        if len(output) >= max_candidates:
            break
    output[0] = SingleArraySeedProposal(baseline, {**common, **counters,
        'source': 'preserved_original_integer_stripes', 'prefix_completed': prefix,
        'total_layers': len(workload.layers), 'DP_complete': prefix == len(workload.layers),
        'elapsed_s': time.monotonic() - started}, {'formal_PPA': False})
    return tuple(output)
