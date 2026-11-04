"""Measured-state cost projection for whole Temporal Group mapping proposals.

Temporal stages reuse the same paid PEs. They must NOT share a spatial sum-quota
constraint. A small inner frontier uses the real fitting-fragment and kernel
models, then linearizes EDP around the measured current schedule. The result is
only a proposal: full routing, queueing, buffer lifetimes and global PPA remain
mandatory. No local estimate is ever supplied as an RL reward.
"""
from dataclasses import replace
from functools import lru_cache
import math
from types import SimpleNamespace

from .actions import partition_options
from .mapping import Partition, TensorRegion
from .single_array_compiler import _fit_full_channel_tile, tile_requirements
from .single_array_evaluator import SingleArrayEvaluator
from .rl_actions import validate_state


class TemporalProjector:
    version = 'measured_edp_temporal_projection_v1'

    def __init__(self, workload, candidates):
        self.w, self.candidates = workload, candidates

    @lru_cache(maxsize=3)
    def oracle(self, architecture):
        # Only pure local kernel/SRAM functions are used. evaluate() is never
        # called here; this cannot create an unaccounted physical PPA sample.
        return SingleArrayEvaluator(self.w, self.candidates[architecture].hardware)

    @lru_cache(maxsize=50)
    def idle_power(self, architecture):
        h = self.candidates[architecture].hardware
        area = h.area_estimate
        return (.02 * area['compute_total_mm2'] + .008 * area['io_total_mm2']
                + area['external_dram_area_mm2'] * h.assumptions.external_dram_idle_w_per_mm2)

    @lru_cache(maxsize=32768)
    def geometry(self, architecture, lid, factors, flow, persistent):
        h = self.candidates[architecture].hardware
        layer, output = self.w.layer_map[lid], self.w.tensor_map[self.w.layer_map[lid].output]
        quota = math.prod(factors)
        shape = tuple(math.ceil(n / p) for n, p in zip(output.shape, factors))
        owned = TensorRegion(tuple((0, n) for n in shape))
        tile = _fit_full_channel_tile(self.w, layer, owned, h)
        fragment = TensorRegion(tuple((0, n) for n in tile))
        req = tile_requirements(self.w, layer, fragment, h.assumptions.psum_storage_bytes)
        tiles = math.prod(math.ceil(n / t) for n, t in zip(shape, tile))
        repetitions = tiles * quota
        kernel = self.oracle(architecture)._kernel(SimpleNamespace(
            macs=fragment.elements * math.prod(layer.weight_shape[1:]) if layer.weight_shape else 0,
            output_region=fragment, weight_bytes=req['weight_bytes'], dataflow=flow), layer)
        mac_s = kernel['mac_cycles'] / h.compute_clock_hz
        l1_s = (kernel['l1_read_bytes'] + kernel['l1_write_bytes'] + req['psum_bytes']) / h.l1_bandwidth_Bps
        active_dies = min(h.compute_count, quota)
        operations = layer.vector_ops_per_element + (dict(h.assumptions.vector_cycles_per_element)['requant'] if layer.weight_shape else 0)
        if layer.kind in ('AvgPool2d', 'AdaptiveAvgPool2d'):
            operations += 3
        vector_ops = fragment.elements * operations
        qparams = tile[1] * 8 if layer.weight_shape else 0
        vector_input = req['psum_bytes'] if layer.weight_shape else req['activation_bytes']
        l2_unit = (req['activation_bytes'] + req['weight_bytes'] + req['psum_bytes'] if layer.weight_shape else 0)
        l2_unit += vector_input + req['output_bytes'] + qparams
        shared_l2_s = repetitions * l2_unit / (active_dies * h.l2_bandwidth_Bps)
        vector_s = repetitions * vector_ops / (active_dies * h.vector_lanes * h.compute_clock_hz)
        # Largest-shard roofline; exact nonuniform tail work and overlap are
        # recomputed by the full compiler/calendar. No steady-state B1 gain.
        # One fragment's atomic staging also affects first-result latency.
        # This is a burst-size proposal term, not a simulated queue benefit.
        burst_s = max(req['activation_bytes'] + req['weight_bytes'], vector_input + req['output_bytes']) / h.l2_bandwidth_Bps
        service = max(tiles * max(mac_s, l1_s), shared_l2_s, vector_s) + burst_s
        local = self.oracle(architecture)
        if layer.weight_shape:
            l1_j = local._sram_j('WL1', read=kernel.get('weight_read_bytes', 0), write=req['weight_bytes'])
            l1_j += local._sram_j('AL1', read=kernel.get('activation_read_bytes', 0), write=req['activation_bytes'])
            psum = kernel.get('partial_sum_rw_bytes', 0)
            l1_j += local._sram_j('OL1', read=psum // 2 + req['psum_bytes'], write=psum - psum // 2 + req['psum_bytes'])
            l2_j = local._sram_j('WL2', read=req['weight_bytes']) + local._sram_j('AL2', read=req['activation_bytes'] + qparams)
            l2_j += local._sram_j('OL2', read=vector_input, write=req['psum_bytes'] + req['output_bytes'])
        else:
            # A die VPU must not acquire fictitious PE L1 traffic.
            l1_j = 0.
            l2_j = local._sram_j('AL2', read=vector_input) + local._sram_j('OL2', write=req['output_bytes'])
        dram_weights = layer.weight_bytes if persistent else repetitions * req['weight_bytes']
        # Named coefficient convention is the SAME uncalibrated profile. Route
        # replication/contention are intentionally absent, not declared free.
        dynamic = repetitions * (l1_j + l2_j + vector_ops * .024e-12)
        dynamic += layer.macs_per_sample * .024e-12 + dram_weights * 8 * 8.75e-12
        idle_w = self.idle_power(architecture)
        return dict(time_s=service, energy_j=dynamic + idle_w * service,
                    fitting_tile=tile, largest_shard=shape, estimated_fragments=repetitions,
                    logical_stage_bytes=repetitions * l2_unit,
                    staging_burst_proxy_s=burst_s,
                    model_scope='local fragment rooflines/SRAM and ideal weight service; no global routing/queue/lifetime prediction')

    def project(self, base, group_id, bank, report, *, cap=32, energy_multiplier=1., placement='spread'):
        c = self.candidates[base.architecture_candidate_id]
        h = c.hardware
        group = next(g for g in base.mapping.groups if g.id == group_id)
        if group.mode != 'temporal':
            raise ValueError('temporal projection cannot overlap spatial ownership')
        if any(set(group.layer_ids).intersection(s.layer_ids) for s in base.mapping.fusion_stacks):
            raise ValueError('Fusion tile ownership needs its separate inner compiler')
        cap = min(cap, len(h.compute_slots))
        order = bank.order(placement)
        latency = max(1e-12, report.get('latency_s', .01))
        energy = max(1e-12, report.get('energy_j', .01))
        changed, records = {}, []
        quotas = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256}
        for lid in group.layer_ids:
            old = base.mapping.mapping_map[lid]
            qold = len(old.chiplets)
            options = []
            for q in sorted({v for v in quotas | {qold - 1, qold, qold + 1} if 1 <= v <= cap}):
                parts = partition_options(self.w, lid, q, 1)
                if not parts:
                    continue
                shape = self.w.tensor_map[self.w.layer_map[lid].output].shape
                selected = {
                    min(parts, key=lambda p: (-p[1], p[2] * p[3], p)),
                    min(parts, key=lambda p: (p[1], shape[2] / p[2] + shape[3] / p[3], p)),
                    min(parts, key=lambda p: (shape[1] / p[1] + shape[2] / p[2] + shape[3] / p[3], p)),
                }
                if old.partition.as_tuple() in parts:
                    selected.add(old.partition.as_tuple())
                allocated = tuple((*old.chiplets[:q], *(p for p in order if p not in old.chiplets)))[:q] if placement == 'preserve' else order[:q]
                for part in sorted(selected):
                    retention = bank.retention_policy(lid, Partition(*part), allocated, 'auto')
                    flows = ('output_stationary', 'weight_reuse', 'activation_reuse') if self.w.layer_map[lid].weight_shape else ('output_stationary',)
                    for flow in flows:
                        try:
                            estimate = self.geometry(c.id, lid, part, flow, retention == 'persistent')
                        except ValueError:
                            continue
                        score = estimate['time_s'] / latency + energy_multiplier * estimate['energy_j'] / energy
                        options.append((score, q, part, flow, retention, allocated, estimate))
            if not options:
                raise ValueError('no fitting temporal local option: ' + lid)
            _, q, part, flow, retention, allocated, estimate = min(options, key=lambda r: (r[0], r[1], r[2], r[3]))
            changed[lid] = replace(old, chiplets=tuple(allocated), partition=Partition(*part), dataflow=flow, weight_policy=retention)
            records.append(dict(layer=lid, quota=q, partition=part, dataflow=flow, retention=retention, **estimate))
        mapping = replace(base.mapping, layer_mappings=tuple(changed.get(m.layer_id, m) for m in base.mapping.layer_mappings))
        state = replace(base, mapping=mapping, evaluation_report=None, remaining_budget=None)
        validate_state(self.w, state, self.candidates)
        return state, dict(version=self.version, group=group_id, mode='temporal', maximum_per_layer_quota=cap,
            energy_multiplier=energy_multiplier, placement=placement, reference=dict(latency_s=latency, energy_j=energy),
            objective='first-order EDP projection: estimated_local_T/measured_global_T + estimated_local_E/measured_global_E',
            allocations=records, proxy_is_full_ppa=False,
            resource_contract='each serial stage may reuse paid physical PEs; each quota<=capacity; no spatial sum-quota restriction')
