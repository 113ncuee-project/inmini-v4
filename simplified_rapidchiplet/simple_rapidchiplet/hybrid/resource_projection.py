"""Integer resource projection for coupled layer-group proposals.

This is an exact DP for a declared separable *proposal* proxy, not a PPA
optimizer. It repairs a whole spatial group jointly, permits idle PEs and every
integer quota with a nonempty B/K/H/W partition. Complete physical evaluation
and reward always take place outside this module.
"""
from dataclasses import replace
from functools import lru_cache
import math

from .actions import partition_options
from .mapping import Partition
from .rl_actions import validate_state


def integer_projection(options, capacity):
    """Minimize sum(option.cost) with one choice/layer and sum(quota)<=capacity.

    An option is (quota, cost, payload). The result is None if no legal cover
    exists. Deterministic ties prefer fewer PEs, then the supplied option order.
    Keeping an idle resource is valid, including at a performance plateau.
    """
    states = {0: (0., ())}
    for choices in options:
        next_states = {}
        for used, (score, path) in states.items():
            for quota, value, payload in choices:
                if not isinstance(quota, int) or quota < 1 or not math.isfinite(value):
                    raise ValueError('invalid projection option')
                total = used + quota
                if total > capacity:
                    continue
                if total not in next_states or score + value < next_states[total][0] - 1e-18:
                    next_states[total] = score + value, (*path, (quota, payload))
        states = next_states
        if not states:
            return None
    if not states:
        return None
    used = min(states, key=lambda u: (states[u][0], u))
    return states[used][1], states[used][0], used


class ResourceProjector:
    version = 'conserving_integer_group_projection_v1'

    def __init__(self, workload, candidates):
        self.w, self.candidates = workload, candidates

    @lru_cache(maxsize=32768)
    def option(self, architecture, lid, quota, memory_weight=1.):
        h = self.candidates[architecture].hardware
        w, layer = self.w, self.w.layer_map[lid]
        output = w.tensor_map[layer.output]
        scored = []
        for factors in partition_options(w, lid, quota, 1):
            b, k, oh, ow = (math.ceil(s / p) for s, p in zip(output.shape, factors))
            # Upper shard shape and halo are a bounded geometric proxy. They
            # preserve channels/groups but omit exact tile lifetimes/arbitration.
            inputs = 0
            for tid in layer.inputs:
                t = w.tensor_map[tid]
                if layer.kind in ('Conv2d', 'MaxPool2d', 'AvgPool2d'):
                    ih = min(t.shape[2], (oh - 1) * layer.stride[0] + layer.dilation[0] * (layer.kernel[0] - 1) + 1)
                    iw = min(t.shape[3], (ow - 1) * layer.stride[1] + layer.dilation[1] * (layer.kernel[1] - 1) + 1)
                    if layer.kind == 'Conv2d':
                        channels = min(t.shape[1], math.ceil(k / max(1, output.shape[1] // layer.groups)) * (t.shape[1] // layer.groups))
                    else:
                        channels = k
                    elements = b * channels * ih * iw
                elif layer.kind in ('Linear', 'AdaptiveAvgPool2d', 'Flatten'):
                    elements = math.ceil(t.elements / max(1, factors[0] * factors[2] * factors[3]))
                else:
                    elements = b * k * oh * ow
                inputs += elements * t.bytes_per_element
            weights = k * math.prod(layer.weight_shape[1:]) * layer.weight_bits // 8 if layer.weight_shape else 0
            out = b * k * oh * ow * output.bytes_per_element
            if layer.weight_shape:
                mac_cycles = b * oh * ow * math.ceil(k / h.lanes) * math.ceil(layer.weight_shape[1] / h.vector_size) * math.prod(layer.kernel)
            else:
                mac_cycles = 0
            # Shared VPU lower bound by active physical dies; not fictitious
            # q independent vector engines or a steady-state batch-1 pipeline.
            active_dies = min(h.compute_count, quota)
            vector_s = output.elements * layer.vector_ops_per_element / max(1, active_dies * h.vector_lanes * h.compute_clock_hz)
            service = mac_cycles / h.compute_clock_hz + vector_s
            logical_stage_bytes = (inputs + weights + out) * quota
            memory_s = logical_stage_bytes / (h.compute_count * h.l2_bandwidth_Bps)
            score = service + memory_weight * memory_s
            scored.append((score, factors, dict(ideal_service_s=service, logical_stage_bytes=logical_stage_bytes,
                                               memory_proxy_s=memory_s)))
        if not scored:
            return None
        score, factors, evidence = min(scored, key=lambda r: (r[0], r[1]))
        return quota, score, dict(partition=factors, **evidence)

    def project(self, base, group_ids, *, memory_weight=1., placement='preserve'):
        """Jointly remap specified *existing* spatial groups, keeping all else.

        The caller may first make a valid topological merge/mode transaction.
        Retention, independent FD and dataflow remain unchanged; no hidden
        compiler repair or free buffer expansion takes place.
        """
        c = self.candidates[base.architecture_candidate_id]
        h, slots = c.hardware, c.hardware.compute_slots
        spread = tuple(f'compute:{d}/pe:{p}' for p in range(16) for d in range(h.compute_count))
        wanted, changed, records = set(group_ids), {}, []
        fused = {lid for s in base.mapping.fusion_stacks for lid in s.layer_ids}
        for group in base.mapping.groups:
            if group.id not in wanted:
                continue
            if group.mode != 'spatial' or fused.intersection(group.layer_ids):
                raise ValueError('projection needs a spatial group without a Fusion stack')
            if len(group.layer_ids) > len(slots):
                raise ValueError('spatial group cannot exceed physical PEs')
            choices = []
            upper = len(slots) - len(group.layer_ids) + 1
            for lid in group.layer_ids:
                choices.append(tuple(v for q in range(1, upper + 1)
                                     if (v := self.option(c.id, lid, q, memory_weight)) is not None))
            result = integer_projection(choices, len(slots))
            if result is None:
                raise ValueError('no conserving group projection')
            allocated, proxy, used = result
            # Allocate every layer's retained prefix first. Filling stage by
            # stage could otherwise steal a later layer's retained physical PEs.
            assigned, occupied = {}, set()
            for lid, (quota, info) in zip(group.layer_ids, allocated):
                old = base.mapping.mapping_map[lid]
                selected = tuple(p for p in old.chiplets if p not in occupied)[:quota] if placement == 'preserve' else ()
                assigned[lid] = list(selected)
                occupied.update(selected)
            order = spread if placement == 'spread' else slots
            idle = iter(p for p in order if p not in occupied)
            for lid, (quota, info) in zip(group.layer_ids, allocated):
                assigned[lid].extend(next(idle) for _ in range(quota - len(assigned[lid])))
                old = base.mapping.mapping_map[lid]
                changed[lid] = replace(old, chiplets=tuple(assigned[lid]), partition=Partition(*info['partition']))
            records.append(dict(group=group.id, used_pes=used, capacity=len(slots), proxy=proxy,
                                allocations={lid: dict(quota=q, **info) for lid, (q, info) in zip(group.layer_ids, allocated)}))
        mapping = replace(base.mapping, layer_mappings=tuple(changed.get(m.layer_id, m) for m in base.mapping.layer_mappings))
        state = replace(base, mapping=mapping, evaluation_report=None, remaining_budget=None)
        validate_state(self.w, state, self.candidates)
        return state, dict(version=self.version, memory_weight=memory_weight, placement=placement,
                           objective='sum of ideal shard service plus logical staging proxy; batch-1 fill/drain, not max-stage throughput',
                           groups=records, proxy_is_full_ppa=False)
