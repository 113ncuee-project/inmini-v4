"""Label-free DAG boundary and proposed-edit encoding for mapping Q learning.

The sketch describes logical tensor ownership, group cuts and resource demand.
It deliberately does not call the compiler, scheduler or physical evaluator.
Logical bytes/liveness and ideal service are *features*, not predicted PPA,
capacity proofs, reward labels or a reason to discard a legal design.
"""
import math

import numpy as np


NAMES = (
    'external_read', 'cross_group_read', 'boundary_write', 'internal_edge_bytes',
    'cross_group_edge_fraction', 'residual_cross_read', 'max_group_live_al2',
    'max_group_live_ol2', 'max_persistent_wl2', 'dram_load_peak',
    'dram_load_cv', 'die_hops_per_edge', 'same_die_overlap', 'same_pe_overlap',
    'partition_disagreement', 'ideal_sum_pe_service', 'ideal_dag_path',
    'ideal_largest_stage', 'spatial_resource_fraction', 'max_spatial_resource_fraction',
    'spatial_group_fraction', 'largest_group_fraction', 'max_group_cross_outputs',
    'fusion_halo_risk',
)

OBSERVED_NAMES = ('observed_target_critical_fraction', 'observed_mac_per_pe', 'observed_l1_per_pe',
    'observed_vpu_per_die', 'observed_max_job_wait', 'observed_max_transfer_wait', 'observed_memory_pressure',
    'proposed_ideal_mac_delta', 'proposed_ideal_l1_delta', 'proposed_ideal_vpu_delta',
    'observed_critical_mac_share', 'weight_reuse_change_at_observed_l1',
    'activation_reuse_change_at_observed_l1', 'allocation_change_at_observed_pressure')


class TensorBoundaryEncoder:
    """A fixed, named graph reduction plus exact logical cut-edit deltas.

    One tensor with two remote consumers has one logical write and two logical
    reads. A residual with both local and remote users keeps both edge classes.
    Physical replication, DMA staging, tiles, NoP/NoC arbitration and capacity
    remain the responsibility of the unchanged complete oracle.
    """
    version = 'tensor_boundary_edit_v1'

    def __init__(self, workload, candidates):
        self.w = workload
        self.candidates = candidates
        self.activation_bytes = max(1, sum(t.bytes for t in workload.tensors))
        self.work = max(1, sum(l.macs_per_sample for l in workload.layers))
        self.cache = {}
        self.die_distances = {}

    def _distances(self, candidate):
        if candidate.id not in self.die_distances:
            n = candidate.hardware.compute_count
            package = candidate.package
            self.die_distances[candidate.id] = np.asarray([
                [len(package.route(f'compute:{a}', f'compute:{b}')) - 1 for b in range(n)] for a in range(n)
            ], dtype=np.float64)
        return self.die_distances[candidate.id]

    def sketch(self, state, *, key=None):
        key = key or state.fingerprint
        if key in self.cache:
            return self.cache[key]
        w, plan = self.w, state.mapping
        c = self.candidates[state.architecture_candidate_id]
        h, n = c.hardware, c.hardware.compute_count
        count = max(1, len(w.layers))
        group = {lid: gi for gi, g in enumerate(plan.groups) for lid in g.layer_ids}
        mappings = plan.mapping_map
        bytes_scale = self.activation_bytes
        loads = np.zeros(h.io_count, dtype=np.float64)
        cuts, external, internal, residual = {}, 0, 0, 0
        write_bytes, read_bytes = 0, 0
        hist, pe_sets = {}, {}
        for lid, m in mappings.items():
            hist[lid] = np.bincount([int(p.split('/')[0].split(':')[1]) for p in m.chiplets], minlength=n) / len(m.chiplets)
            pe_sets[lid] = set(m.chiplets)

        def charge(selector, amount):
            if selector == 'interleave':
                loads[:] += amount / len(loads)
            elif selector.startswith('memory:'):
                loads[int(selector.split(':')[1])] += amount
            # A direct internal edge is not a DRAM demand. Invalid external
            # direct selectors are still rejected by the original validator.

        for t in w.tensors:
            if not t.producer:
                continue
            remote = [u for u in w.consumers[t.id] if group[u] != group[t.producer]]
            if remote or t.id in w.output_ids:
                write_bytes += t.bytes
                charge(mappings[t.producer].output_dram, t.bytes)
            for user in remote:
                cuts[(t.id, user)] = t.bytes
                read_bytes += t.bytes
                charge(mappings[t.producer].output_dram, t.bytes)
                if len(w.consumers[t.id]) > 1:
                    residual += t.bytes

        edge_bytes, hops, die_overlap, pe_overlap, partition_gap = 0, 0., 0., 0., 0.
        distances = self._distances(c)
        for layer in w.layers:
            for tid in layer.inputs:
                t = w.tensor_map[tid]
                previous = t.producer
                if not previous:
                    external += t.bytes
                    charge(mappings[layer.id].input_dram, t.bytes)
                    continue
                edge_bytes += t.bytes
                if group[previous] == group[layer.id]:
                    internal += t.bytes
                hops += t.bytes * float(hist[previous] @ distances @ hist[layer.id])
                die_overlap += t.bytes * float(np.minimum(hist[previous], hist[layer.id]).sum())
                pe_overlap += t.bytes * len(pe_sets[previous] & pe_sets[layer.id]) / max(1, min(len(pe_sets[previous]), len(pe_sets[layer.id])))
                partition_gap += t.bytes * sum(abs(math.log2(a / b)) for a, b in zip(
                    mappings[previous].partition.as_tuple(), mappings[layer.id].partition.as_tuple())) / 24
            if layer.weight_shape:
                # Unique logical weights only, not an invented cold/persistent
                # traffic saving. Repeated tile reloads require the oracle.
                charge(mappings[layer.id].weight_dram, layer.weight_bytes)

        live_peaks, persistent_peaks, outgoing_peaks = [], [], []
        spatial_used, largest_group = [], 0
        for g in plan.groups:
            ids = set(g.layer_ids)
            largest_group = max(largest_group, len(ids))
            index = {lid: i for i, lid in enumerate(g.layer_ids)}
            intervals = []
            outgoing = 0
            for t in w.tensors:
                users = [u for u in w.consumers[t.id] if u in ids]
                if t.producer in ids:
                    left = index[t.producer]
                    boundary = t.id in w.output_ids or any(u not in ids for u in w.consumers[t.id])
                    right = len(ids) if boundary else max((index[u] for u in users), default=left)
                    if boundary:
                        outgoing += t.bytes
                elif users:
                    left, right = -1, max(index[u] for u in users)
                else:
                    continue
                intervals.append((left, right, t.bytes))
            # Whole tensors at topological boundaries, no claim that all of
            # this volume is simultaneously resident on one physical die.
            live_peaks.append(max((sum(b for lo, hi, b in intervals if lo <= step <= hi)
                                   for step in range(-1, len(ids) + 1)), default=0))
            persistent_peaks.append(sum(w.layer_map[lid].weight_bytes for lid in ids
                                        if mappings[lid].weight_policy == 'persistent'))
            outgoing_peaks.append(outgoing)
            if g.mode == 'spatial':
                spatial_used.append(sum(len(mappings[lid].chiplets) for lid in ids) / len(h.compute_slots))

        service, finish = {}, {}
        ideal_total = self.work / max(1, h.lanes * h.vector_size * h.compute_clock_hz)
        for layer in w.layers:
            # Ideal per-PE MAC work plus an explicit shared vector lower bound.
            # This ignores tiling/contention; using it as final latency is wrong.
            m = mappings[layer.id]
            service[layer.id] = (layer.macs_per_sample / max(1, len(m.chiplets) * h.lanes * h.vector_size * h.compute_clock_hz)
                + w.tensor_map[layer.output].elements * layer.vector_ops_per_element / max(1, n * h.vector_lanes * h.compute_clock_hz))
            finish[layer.id] = service[layer.id] + max((finish.get(w.tensor_map[t].producer, 0.) for t in layer.inputs), default=0.)
        halo = sum(sum(max(0, max(w.layer_map[lid].kernel) - 1) for lid in s.layer_ids) / max(1, min(s.tile_hw))
                   for s in plan.fusion_stacks if s.overlap != 'full_cache')
        values = np.asarray((
            external / bytes_scale, read_bytes / bytes_scale, write_bytes / bytes_scale, internal / bytes_scale,
            len(cuts) / max(1, sum(len(l.inputs) for l in w.layers)), residual / bytes_scale,
            max(live_peaks, default=0) / (n * h.l2_activation_bytes), max(live_peaks, default=0) / (n * h.l2_output_bytes),
            max(persistent_peaks, default=0) / (n * h.l2_weight_bytes),
            float(loads.max()) / bytes_scale, float(loads.std() / max(1., loads.mean())),
            hops / max(1, edge_bytes * 32), die_overlap / max(1, edge_bytes), pe_overlap / max(1, edge_bytes),
            partition_gap / max(1, edge_bytes), sum(service.values()) / max(1e-30, ideal_total),
            max(finish.values(), default=0.) / max(1e-30, ideal_total), max(service.values(), default=0.) / max(1e-30, ideal_total),
            sum(spatial_used) / max(1, len(spatial_used)), max(spatial_used, default=0.),
            len(spatial_used) / max(1, len(plan.groups)), largest_group / count,
            max(outgoing_peaks, default=0) / bytes_scale, halo / count,
        ), dtype=np.float64)
        if len(self.cache) > 4096:
            self.cache.clear()
        self.cache[key] = values, cuts
        return values, cuts

    @property
    def names(self):
        return tuple(prefix + name for prefix in ('before_', 'after_', 'delta_') for name in NAMES) + (
            'removed_cut_bytes', 'added_cut_bytes', 'changed_cut_fraction')

    def encode(self, before, after, *, before_sketch=None, after_key=None):
        b, bc = before_sketch if before_sketch is not None else self.sketch(before)
        a, ac = self.sketch(after, key=after_key)
        # Edges are keyed by tensor and consumer, not layer/group names alone.
        removed, added = set(bc) - set(ac), set(ac) - set(bc)
        delta = (sum(bc[e] for e in removed) / self.activation_bytes,
                 sum(ac[e] for e in added) / self.activation_bytes,
                 (len(removed) + len(added)) / max(1, len(bc) + len(ac)))
        return np.clip(np.asarray((*b, *a, *(a - b), *delta), dtype=np.float32), -8, 8)

    def observed_edit(self, before, report, after, targets):
        """Actual BEFORE-schedule evidence and explicit proposed resource edits.

        Service-per-PE/die normalization is a feature, not a claim to uniquely
        attribute overlapping waits or to predict the candidate critical path.
        No after-state PPA or future observation is read here.
        """
        evidence = {r['layer_id']: r for r in report.get('diagnostics', {}).get('layers', ())}
        targets = tuple(l for l in dict.fromkeys(targets) if l in self.w.layer_map)
        if not targets and before.architecture_candidate_id != after.architecture_candidate_id:
            targets = tuple(self.w.layer_map)
        n = max(1, len(targets))
        latency = max(1e-12, report.get('latency_s', 1.))
        values = [0.] * len(OBSERVED_NAMES)
        for lid in targets:
            r = evidence.get(lid, {})
            old, new = before.mapping.mapping_map[lid], after.mapping.mapping_map[lid]
            qold, qnew = len(old.chiplets), len(new.chiplets)
            dold = len({p.split('/')[0] for p in old.chiplets})
            dnew = len({p.split('/')[0] for p in new.chiplets})
            critical = float(r.get('criticality', 0) or 0)
            mac = (r.get('mac_service_s', 0) or 0) / max(1, qold) / latency
            l1 = (r.get('l1_service_s', 0) or 0) / max(1, qold) / latency
            vpu = (r.get('vector_busy_s', 0) or 0) / max(1, dold) / latency
            pressure = float(r.get('memory_pressure', 0) or 0)
            values[0] += critical / n
            values[1] += mac
            values[2] += l1
            values[3] += vpu
            values[4] = max(values[4], (r.get('max_job_queue_wait_s', 0) or 0) / latency)
            values[5] = max(values[5], (r.get('max_transfer_queue_wait_s', 0) or 0) / latency)
            values[6] = max(values[6], pressure)
            values[7] += critical * mac * (qold / qnew - 1)
            values[8] += critical * l1 * (qold / qnew - 1)
            values[9] += critical * vpu * (dold / max(1, dnew) - 1)
            values[10] += critical * self.w.layer_map[lid].macs_per_sample / self.work
            values[11] += l1 * (int(new.dataflow == 'weight_reuse') - int(old.dataflow == 'weight_reuse'))
            values[12] += l1 * (int(new.dataflow == 'activation_reuse') - int(old.dataflow == 'activation_reuse'))
            values[13] += pressure * (qnew - qold) / max(1, len(self.candidates[after.architecture_candidate_id].hardware.compute_slots) * n)
        return np.clip(np.asarray(values, dtype=np.float32), -8, 8)
