"""Label-free local-fragment service observations for after-state learning.

Unlike an ideal MAC/quota feature, this exposes actual kernel ceiling, fitting
fragment, shared die L2/VPU placement and temporal/spatial group structure.
It never compiles an executable schedule, calls evaluate(), supplies a reward,
proves feasibility, or discards an action. Routing, queueing, exact tails,
Fusion reuse/recomputation and buffer lifetimes still require the full oracle.
"""
from collections import Counter
import math

import numpy as np

from .temporal_projection import TemporalProjector


NAMES = ('sum_fragment_service', 'group_path_service', 'dependency_path_service',
         'largest_stage_service', 'pe_roofline_service', 'shared_l2_service',
         'shared_vpu_service', 'local_dynamic_energy', 'idle_at_group_service',
         'fragment_count', 'staging_burst', 'stage_bytes', 'active_die_work_fraction',
         'peak_die_assignment_fraction', 'geometry_unknown_fraction', 'fusion_layer_fraction')


class ServiceFeatureEncoder:
    version = 'local_fragment_shared_resource_after_features_v1'

    def __init__(self, workload, candidates):
        self.w, self.candidates = workload, candidates
        self.projector = TemporalProjector(workload, candidates)
        self.cache = {}

    @property
    def names(self):
        return tuple(prefix + name for prefix in ('before_', 'after_', 'delta_') for name in NAMES)

    def sketch(self, state):
        if state.fingerprint in self.cache:
            return self.cache[state.fingerprint]
        c = self.candidates[state.architecture_candidate_id]
        h = c.hardware
        rows, ends, failed = {}, {}, 0
        work = max(1, sum(l.macs_per_sample + self.w.tensor_map[l.output].elements for l in self.w.layers))
        active, peak, dynamic, fragments, burst, staged = 0., 0., 0., 0., 0., 0.
        for layer in self.w.layers:
            m = state.mapping.mapping_map[layer.id]
            histogram = Counter(p.split('/')[0] for p in m.chiplets)
            q = len(m.chiplets)
            mass = layer.macs_per_sample + self.w.tensor_map[layer.output].elements
            active += mass * len(histogram) / h.compute_count / work
            peak += mass * max(histogram.values()) / q / work
            try:
                g = self.projector.geometry(c.id, layer.id, m.partition.as_tuple(), m.dataflow,
                                            m.weight_policy == 'persistent')
                # The local geometry assumes an even ideal spread; actual
                # physical die ownership is an observable proposed decision.
                ideal_dies = min(h.compute_count, q)
                l2 = {die: g['shared_l2_time_s'] * ideal_dies * count / q for die, count in histogram.items()}
                vpu = {die: g['vector_time_s'] * ideal_dies * count / q for die, count in histogram.items()}
                t = max(g['pe_service_time_s'], max(l2.values()), max(vpu.values())) + g['staging_burst_proxy_s']
                rows[layer.id] = dict(time=t, pe=g['pe_service_time_s'], l2=l2, vpu=vpu)
                dynamic += g['dynamic_energy_j']
                fragments += g['estimated_fragments']
                burst += g['staging_burst_proxy_s']
                staged += g['logical_stage_bytes']
            except ValueError:
                # Unknown local fitting is a mask, not a fabricated illegal
                # design or an infinity that prevents full Fusion evaluation.
                failed += 1
                t = layer.macs_per_sample / max(1, q*h.lanes*h.vector_size*h.compute_clock_hz)
                rows[layer.id] = dict(time=t, pe=t, l2={}, vpu={})
            ends[layer.id] = t + max((ends.get(self.w.tensor_map[tid].producer, 0.) for tid in layer.inputs), default=0.)
        group_times = []
        for group in state.mapping.groups:
            if group.mode == 'temporal':
                group_times.append(sum(rows[lid]['time'] for lid in group.layer_ids))
            else:
                finish, loads = {}, {'l2': Counter(), 'vpu': Counter()}
                for lid in group.layer_ids:
                    finish[lid] = rows[lid]['time'] + max((finish.get(self.w.tensor_map[tid].producer, 0.)
                        for tid in self.w.layer_map[lid].inputs), default=0.)
                    for field in loads:
                        loads[field].update(rows[lid][field])
                group_times.append(max(max(finish.values(), default=0.),
                    *(max(loads[field].values(), default=0.) for field in loads)))
        fused = {lid for stack in state.mapping.fusion_stacks for lid in stack.layer_ids}
        values = [sum(r['time'] for r in rows.values()), sum(group_times), max(ends.values(), default=0.),
                  max((r['time'] for r in rows.values()), default=0.), sum(r['pe'] for r in rows.values()),
                  sum(max(r['l2'].values(), default=0.) for r in rows.values()),
                  sum(max(r['vpu'].values(), default=0.) for r in rows.values()), dynamic,
                  self.projector.idle_power(c.id)*sum(group_times), fragments, burst, staged,
                  active, peak, failed/max(1,len(rows)), len(fused)/max(1,len(rows))]
        scales = [1e-3]*7 + [1e-3, 1e-3, 4096., 1e-3, 16e6]
        encoded = np.asarray([math.log1p(v/scales[i]) if i < len(scales) else v
                              for i,v in enumerate(values)], dtype=np.float32)
        if len(self.cache) > 2048:
            self.cache.clear()
        self.cache[state.fingerprint] = encoded
        return encoded

    def encode(self, before, after):
        b, a = self.sketch(before), self.sketch(after)
        return np.asarray((*b, *a, *(a-b)), dtype=np.float32)
