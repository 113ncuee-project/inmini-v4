"""Full schema-3 design domain for the unified replay search.

The legacy transaction generator is retained, including its integer partitions,
independent tensor FD, ordered PE permutations, group edits and Fusion controls.
Structured global proposals are additional shortcuts, never a domain restriction.
All candidates still require the complete SingleArrayOraclePool evaluation.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import cached_property
import math
import time

import numpy as np

from .actions import partition_options
from .depth_first import legal_stack
from .mapping import FusionStack, LayerGroup, Partition, initial_mapping
from .mapping_proposals_v4 import StructuredMappings
from .rl_actions import apply_transaction, generate_candidates, resized_mapping, validate_state, _group_dict
from .rl_schema import DesignState, ActionEdit, make_action_plan, stable_hash
from .single_array_proposals import SingleArrayJointProposals
from .tensor_boundary import TensorBoundaryEncoder, OBSERVED_NAMES
from .resource_projection import ResourceProjector
from .lp_five import propose_five
from .temporal_projection import TemporalProjector
from .tensor_locality import TensorLocalityEncoder


FAMILIES = tuple('A' + str(i) for i in (*range(1, 11), 12)) + tuple('C' + str(i) for i in range(1, 10)) + (
    'global_mapping', 'retention', 'inner_dataflow', 'fusion_beam', 'architecture_mapping', 'group_projection')


def decision_hash(state):
    """Controller-only identity, invariant to display labels of ordered groups.

    Actual plans and the oracle's exact durable keys remain untouched. Physical
    group membership/order/mode, allocations, FD, tiles and all other decisions
    stay in this identity; only the arbitrary human labels are canonicalized.
    """
    data = replace(state, evaluation_report=None, remaining_budget=None).to_dict()
    data.pop('evaluation_report')
    data.pop('remaining_budget')
    for index, group in enumerate(data['mapping']['groups']):
        group['id'] = f'group{index}'
    return stable_hash(data)


@dataclass(frozen=True)
class DesignTrial:
    state: DesignState
    family: str
    targets: tuple = ()
    action: dict | None = None

    @cached_property
    def fingerprint(self):
        return decision_hash(self.state)

    def to_dict(self):
        return dict(state=self.state.to_dict(), family=self.family, targets=self.targets, action=self.action)

    @classmethod
    def from_dict(cls, data):
        return cls(DesignState.from_dict(data['state']), data['family'], tuple(data.get('targets', ())), data.get('action'))


class FullDesignSpace:
    version = 'v07_cost_value_full_schema3_1'

    def __init__(self, workload, candidates, *, shortlist=64, max_depth=4, inner=True,
                 composites=True, guidance=False, fusion=True, legacy_generator=False,
                 boundary_features=True, resource_projection=True, operators='full', architecture_scaling='proportional'):
        self.w = workload
        self.candidates = {c.id: c for c in candidates}
        self.shortlist, self.max_depth = shortlist, max_depth
        self.inner, self.composites, self.guidance, self.fusion = inner, composites, guidance, fusion
        self.legacy_generator = legacy_generator
        self.boundary_features, self.resource_projection = boundary_features, resource_projection
        self.operators = operators
        self.architecture_scaling = architecture_scaling
        if architecture_scaling not in ('legacy', 'proportional'):
            raise ValueError('unknown architecture quota scaling')
        if operators not in ('full', 'gemini5'):
            raise ValueError('unknown operator scope')
        self.boundaries = TensorBoundaryEncoder(workload, self.candidates)
        self.locality = TensorLocalityEncoder(workload, self.candidates)
        self.projector = ResourceProjector(workload, self.candidates)
        self.temporal_projector = TemporalProjector(workload, self.candidates)
        self.calls = 0
        self.banks = {}
        self.history = []

    def bank(self, state):
        key = state.architecture_candidate_id
        if key not in self.banks:
            self.banks[key] = StructuredMappings(self.w, self.candidates[key], state, inner=self.inner)
        return self.banks[key]

    @property
    def graph_feature_names(self):
        return (*self.boundaries.names, *OBSERVED_NAMES, *self.locality.names)

    @property
    def after_design_indices(self):
        """Only absolute after-design descriptions, invariant to the before.

        Omit measured before PPA, progress, family/targets, before sketches,
        deltas, and observed bottleneck features from the physical cost head.
        The continuation Q branch still receives the complete original vector.
        """
        prefix, design = 6 + len(FAMILIES), 34
        result = list(range(prefix + design, prefix + 2 * design))
        base = prefix + 3 * design + 15
        if self.boundary_features:
            boundary = (len(self.boundaries.names) - 3) // 3
            locality = (len(self.locality.names) - 1) // 3
            result.extend(range(base + boundary, base + 2 * boundary))
            offset = base + len(self.boundaries.names) + len(OBSERVED_NAMES)
            result.extend(range(offset + locality, offset + 2 * locality))
        return tuple(result)

    def domain(self):
        return dict(version=self.version, architectures=[c.to_dict() for c in self.candidates.values()],
            active_operator_scope=self.operators,
            architecture_quota_scaling=self.architecture_scaling,
            after_cost_features=dict(indices=self.after_design_indices,
                role='absolute after-design inputs for optional cost-factored Q; no measured after PPA or before-state dependence'),
            comparison_scope='full schema3' if self.operators == 'full' else 'Gemini-style OP1-OP5 only; grouping/architecture/inner decisions fixed for this comparison',
            decision_identity='all physical decisions with ordered group labels canonicalized; exact oracle inputs unchanged',
            families=FAMILIES, partition='all nonempty integer B/K/H/W supported by schema3; B=1',
            placement='ordered PE swaps, idle relocation, inter-layer swaps and explicit allocation',
            dram='independent external activation, output materialization and weight endpoints; paid interleave',
            groups='topological temporal/spatial merge/split and joint reallocation',
            fusion=dict(enabled=self.fusion, depth_limit=self.max_depth, tiles='all positive bounded sink H/W; atomic +/-1, halve/double and uniform jumps',
                        overlap=['recompute', 'horizontal_cache', 'full_cache']),
            microbatch=dict(values=[1], reason='same Batch=1 physical compiler contract as v3'),
            dataflows=['output_stationary', 'weight_reuse', 'activation_reuse'],
            inner_tiling='deterministic physical per-pool full-channel fitting inside each ownership region; inherited channel/microtile fields are not additional searched hardware modes',
            weight_policies=['cold', 'persistent'],
            graph_edit_features=dict(enabled=self.boundary_features, version=self.boundaries.version,
                                     ownership_version=self.locality.version,
                                     names=self.graph_feature_names, labels='before-schedule context and logical edits only; full oracle rewards only'),
            resource_projection=dict(enabled=self.resource_projection, version=self.projector.version,
                                     temporal_version=self.temporal_projector.version,
                                     role='additional spatial conserving DP and serial-stage measured-cost proposals; no domain pruning'),
            approximate='finite rotating proposals and inner proxy beam; no global optimality proof',
            unsupported=['input-channel package reduction', 'arbitrary floorplan/routing', 'vertical-only cache'])

    def architecture_trial(self, base, candidate, *, legacy=False):
        """One-step reachability to every rectangle, with explicit target mapping.

        The preserving-quota shortcut avoids unnecessarily activating all PEs.
        Legacy proportional resizing remains a separately sampled alternative.
        Neither operation changes the target's physical IO/SRAM/area costs.
        """
        if legacy and self.architecture_scaling == 'legacy':
            mapping = resized_mapping(self.w, base, candidate)
        else:
            h = candidate.hardware
            memories = tuple(candidate.package.memory_ids)
            # Retain pipeline boundaries and fusion. A spatial group longer
            # than target capacity is explicitly converted to temporal.
            groups = tuple(replace(g, mode='temporal') if len(g.layer_ids) > len(h.compute_slots) else g
                           for g in base.mapping.groups)
            mappings = []
            # Spread physical resources across the actual compute dies.
            slots = tuple(f'compute:{d}/pe:{p}' for p in range(16) for d in range(h.compute_count))
            for group in groups:
                cursor = 0
                for index, lid in enumerate(group.layer_ids):
                    old = base.mapping.mapping_map[lid]
                    remaining = len(slots) - cursor - (len(group.layer_ids) - index - 1)
                    # The historical "resized" Temporal proposal activated
                    # EVERY target PE, even if the source left most idle. A
                    # proportional shortcut preserves the paid utilization
                    # fraction and spreads it across the real target dies.
                    # Atomic allocation still reaches every legal quota.
                    ratio = len(slots) / len(self.candidates[base.architecture_candidate_id].hardware.compute_slots)
                    desired = max(1, round(len(old.chiplets) * ratio)) if legacy else len(old.chiplets)
                    quota = min(desired, remaining if group.mode == 'spatial' else len(slots))
                    while quota > 0:
                        parts = partition_options(self.w, lid, quota, 1)
                        if parts:
                            break
                        quota -= 1
                    if not quota:
                        raise ValueError('no nonempty target allocation')
                    part = min(parts, key=lambda p: sum(abs(math.log(a / b)) for a, b in zip(p, old.partition.as_tuple())))
                    allocated = slots[cursor:cursor + quota] if group.mode == 'spatial' else slots[:quota]
                    if group.mode == 'spatial':
                        cursor += quota
                    selectors = {key: getattr(old, key) if getattr(old, key) in (*memories, 'interleave', 'direct') else 'interleave'
                                 for key in ('input_dram', 'weight_dram', 'output_dram')}
                    mappings.append(replace(old, chiplets=allocated, partition=Partition(*part), **selectors))
            by_id = {m.layer_id: m for m in mappings}
            mapping = replace(base.mapping, groups=groups, memory_ids=memories,
                              layer_mappings=tuple(by_id[m.layer_id] for m in base.mapping.layer_mappings))
        state = replace(base, architecture_candidate_id=candidate.id, mapping=mapping,
                        evaluation_report=None, remaining_budget=None)
        validate_state(self.w, state, self.candidates)
        return DesignTrial(state, 'architecture_mapping', (), dict(target=candidate.id, legacy_resize=legacy,
                          allocation_mode=('historical_full_resize' if self.architecture_scaling == 'legacy' else 'proportional_spread') if legacy else 'preserve_spread',
                          complete_mapping=mapping.to_dict()))

    def _fusion_beam(self, base, rng, limit=6):
        """Propose closed DAG windows and tiles with a finite inner proxy beam.

        Proxy counts tensor/MAC work and explicitly accounts for recomputation
        and cache footprint directions. It is only a proposal rank, never PPA.
        Both temporal/spatial and off/on joint proposals also remain in v3.
        """
        order = tuple(l.id for l in self.w.layers)
        occupied = {l for s in base.mapping.fusion_stacks for l in s.layer_ids}
        windows = [order[i:i + d] for d in range(2, min(self.max_depth, len(order)) + 1)
                   for i in range(len(order) - d + 1)
                   if not occupied.intersection(order[i:i + d]) and legal_stack(self.w, order[i:i + d])]
        if not windows:
            return []
        # Rotate depth/window coverage rather than repeatedly choosing the
        # first pair. The random component keeps every legal window reachable.
        window = windows[(self.calls - 1) % len(windows)] if self.calls % 2 else rng.choice(windows)
        helper = SingleArrayJointProposals(self.w, tuple(self.candidates.values()))
        groups = helper._regroup(base.mapping, window, 'temporal')
        sink = self.w.tensor_map[self.w.layer_map[window[-1]].output]
        tiles = tuple(dict.fromkeys((min(t, sink.shape[2]), min(t, sink.shape[3])) for t in (2, 4, 8, 16, 32, 64)))
        from .rl_actions import _has_overlap
        modes = ('recompute', 'horizontal_cache', 'full_cache') if _has_overlap(self.w, window) else ('recompute',)
        candidates = []
        for tile in tiles:
            for mode in modes:
                stack = FusionStack(window, tile, mode)
                state = replace(base, mapping=replace(base.mapping, groups=groups,
                    fusion_stacks=(*base.mapping.fusion_stacks, stack)), evaluation_report=None)
                try:
                    validate_state(self.w, state, self.candidates)
                except ValueError:
                    continue
                boundary = sum(self.w.tensor_map[t].bytes for lid in window for t in self.w.layer_map[lid].inputs
                               if self.w.tensor_map[t].producer not in window)
                tiles_count = math.ceil(sink.shape[2] / tile[0]) * math.ceil(sink.shape[3] / tile[1])
                footprint = sum(tile[0] * tile[1] * self.w.tensor_map[self.w.layer_map[l].output].shape[1] for l in window)
                # Keep footprint/traffic/recompute as separate ranking axes.
                # Selecting representatives across axes preserves tradeoffs.
                h = self.candidates[base.architecture_candidate_id].hardware
                weights = sum(self.w.layer_map[l].weight_bytes for l in window
                              if base.mapping.mapping_map[l].weight_policy == 'cold')
                reload = weights * tiles_count
                halo = sum(max(self.w.layer_map[l].kernel) - 1 for l in window)
                recompute = (halo / max(1, min(tile))) if mode == 'recompute' else (halo / max(1, tile[0]) if mode == 'horizontal_cache' else 0.)
                memory_pressure = footprint * (3 if mode == 'full_cache' else 2 if mode == 'horizontal_cache' else 1) / max(1, h.l2_output_bytes)
                traffic_rank = (boundary + reload) * (1 + recompute) * (1 + max(0., memory_pressure - 1) * 8)
                score = (traffic_rank, footprint, tiles_count)
                candidates.append((score, DesignTrial(state, 'fusion_beam', window,
                    dict(window=window, tile=tile, cache=mode, proxy=score, proxy_is_full_ppa=False))))
        if not candidates:
            return []
        selected = [t for _, t in sorted(candidates, key=lambda row: row[0])[:max(1, limit - 2)]]
        # Random representatives and the atomic tile actions retain small-tile
        # reachability even when the traffic proxy favors larger sink tiles.
        selected.extend(t for _, t in rng.sample(candidates, min(limit, len(candidates))))
        return list({t.fingerprint: t for t in selected}.values())[:limit]

    def _sample_transactions(self, base, report, rng):
        """Lazy parameter sampling in the original V3 transaction domain.

        Avoid materializing every PE pair, partition and DRAM endpoint before
        throwing almost all away. Target layers/parameters have nonzero support;
        validation and execution still use the unmodified V3 transaction code.
        """
        plans = []
        h = self.candidates[base.architecture_candidate_id].hardware
        slots = h.compute_slots
        memories = (*base.mapping.memory_ids, 'interleave')
        def add(family, edits, targets=()):
            plans.append(make_action_plan(base, family, edits, targets=targets, sources=('v05_lazy_v3_domain',)))

        ids = rng.sample(tuple(self.w.layer_map), min(8, len(self.w.layers)))
        for lid in ids:
            item = base.mapping.mapping_map[lid]
            group = base.mapping.group_map[lid]
            quota = len(item.chiplets)
            parts = [p for p in partition_options(self.w, lid, quota, 1) if p != item.partition.as_tuple()]
            for p in rng.sample(parts, min(2, len(parts))):
                add('A2', [('A2', dict(layer_id=lid, partition=p))], (lid,))
            order = list(item.chiplets)
            if quota > 1:
                a, b = rng.sample(range(quota), 2)
                order[a], order[b] = order[b], order[a]
                add('A5', [('A5', dict(layer_id=lid, cores=order))], (lid,))
            occupied = {s for l in group.layer_ids for s in base.mapping.mapping_map[l].chiplets} if group.mode == 'spatial' else set(item.chiplets)
            idle = [p for p in slots if p not in occupied]
            if idle:
                order = list(item.chiplets)
                order[rng.randrange(quota)] = rng.choice(idle)
                add('A5', [('A5', dict(layer_id=lid, cores=order))], (lid,))
            # Every feasible quota is reachable, including 3,5,6,...; nearest
            # acquire/release directions are added for fine local traversal.
            counts = {max(1, quota - 1), min(quota + len(idle), quota + 1), rng.randint(1, quota + len(idle))}
            for count in counts:
                options = partition_options(self.w, lid, count, 1)
                if not options or count == quota:
                    continue
                selected = tuple(item.chiplets[:count]) if count < quota else (*item.chiplets, *rng.sample(idle, count - quota))
                add('A4', [('A4', dict(allocations={lid: dict(cores=selected, partition=rng.choice(options))}))], (lid,))
            layer = self.w.layer_map[lid]
            if layer.weight_shape:
                for flow in ('output_stationary', 'weight_reuse', 'activation_reuse'):
                    if flow != item.dataflow:
                        add('A12', [('A12', dict(layer_id=lid, dataflow=flow))], (lid,))
                add('A8', [('A8', dict(tensor_id='weight:' + lid, configuration_id=rng.choice(memories)))], (lid,))
            for tid in (*layer.inputs, layer.output):
                tensor = self.w.tensor_map[tid]
                active = (not tensor.producer or tid in self.w.output_ids or
                          any(base.mapping.group_map[u].id != base.mapping.group_map[tensor.producer].id for u in self.w.consumers[tid]))
                if active:
                    targets = tuple(dict.fromkeys((*( (tensor.producer,) if tensor.producer else ()), *self.w.consumers[tid])))
                    add('A8', [('A8', dict(tensor_id=tid, configuration_id=rng.choice(memories)))], targets)

        fused = {l for s in base.mapping.fusion_stacks for l in s.layer_ids}
        for stack in base.mapping.fusion_stacks if self.fusion else ():
            members = stack.layer_ids
            add('A1', [('A1', dict(operation='split', layer_ids=members))], members)
            sink = self.w.tensor_map[self.w.layer_map[members[-1]].output]
            for axis in (0, 1):
                # Halving/doubling alone cannot reach e.g. tile 5 from 4 in
                # every finite sink extent. Unit moves connect the whole
                # bounded integer domain; random jumps avoid a long walk.
                for delta in (-1, 1):
                    tile = list(stack.tile_hw)
                    tile[axis] = max(1, min(sink.shape[axis + 2], tile[axis] + delta))
                    add('A3', [('A3', dict(layer_ids=members, tile_hw=tile))], members)
                for factor in (.5, 2.):
                    tile = list(stack.tile_hw)
                    tile[axis] = max(1, min(sink.shape[axis + 2], int(tile[axis] * factor)))
                    edit = ActionEdit('A3', dict(layer_ids=members, tile_hw=tile))
                    add('A3', [edit], members)
                    if self.composites:
                        add('C7' if factor < 1 else 'C8', [edit, ActionEdit('A7', dict(layer_ids=members,
                            cache_h=factor > 1, cache_v=factor > 1))], members)
            tile = [rng.randint(1, sink.shape[2]), rng.randint(1, sink.shape[3])]
            add('A3', [('A3', dict(layer_ids=members, tile_hw=tile))], members)
            for ch, cv in ((False, False), (True, False), (True, True)):
                add('A7', [('A7', dict(layer_ids=members, cache_h=ch, cache_v=cv))], members)
        groups = base.mapping.groups
        for i in rng.sample(range(len(groups)), min(6, len(groups))):
            group = groups[i]
            if fused.intersection(group.layer_ids):
                continue
            if self.fusion and 2 <= len(group.layer_ids) <= self.max_depth and legal_stack(self.w, group.layer_ids):
                add('A1', [('A1', dict(operation='merge', layer_ids=group.layer_ids, tile_hw=(8, 8)))], group.layer_ids)
            alternatives = []
            if len(group.layer_ids) > 1:
                cut = rng.randrange(1, len(group.layer_ids))
                alternatives.append((*groups[:i], LayerGroup(group.id + '_l', group.layer_ids[:cut], group.mode),
                    LayerGroup(group.id + '_r', group.layer_ids[cut:], group.mode), *groups[i + 1:]))
            if i + 1 < len(groups) and not fused.intersection(groups[i + 1].layer_ids):
                members = (*group.layer_ids, *groups[i + 1].layer_ids)
                alternatives.append((*groups[:i], LayerGroup(group.id + '_merge', members, 'temporal'), *groups[i + 2:]))
            alternatives.append((*groups[:i], replace(group, mode='temporal' if group.mode == 'spatial' else 'spatial'), *groups[i + 1:]))
            for changed in alternatives:
                # Reallocate for spatial mode in one shadow transaction.
                shadow = replace(base, mapping=replace(base.mapping, groups=tuple(changed)))
                try:
                    remapped = self.architecture_trial(shadow, self.candidates[base.architecture_candidate_id]).state.mapping
                except ValueError:
                    continue
                add('A10', [('A10', dict(groups=_group_dict(changed), mapping=remapped.to_dict()))], group.layer_ids)
            if group.mode == 'spatial' and len(group.layer_ids) > 1:
                left, right = rng.sample(group.layer_ids, 2)
                li, ri = base.mapping.mapping_map[left], base.mapping.mapping_map[right]
                add('A6', [('A6', dict(left_layer_id=left, right_layer_id=right,
                                     left_indices=[rng.randrange(len(li.chiplets))], right_indices=[rng.randrange(len(ri.chiplets))]))], (left, right))
                if len(li.chiplets) > 1:
                    left_parts = partition_options(self.w, left, len(li.chiplets) - 1, 1)
                    right_parts = partition_options(self.w, right, len(ri.chiplets) + 1, 1)
                    if left_parts and right_parts:
                        allocations = {left: dict(cores=li.chiplets[:-1], partition=rng.choice(left_parts)),
                                       right: dict(cores=(*ri.chiplets, li.chiplets[-1]), partition=rng.choice(right_parts))}
                        add('A4', [('A4', dict(allocations=allocations))], (left, right))
        if self.composites:
            edges = [(self.w.tensor_map[t].producer, l.id) for l in self.w.layers for t in l.inputs if self.w.tensor_map[t].producer]
            for left, right in rng.sample(edges, min(4, len(edges))):
                edits = []
                for lid in (left, right):
                    m = base.mapping.mapping_map[lid]
                    options = partition_options(self.w, lid, len(m.chiplets), 1)
                    if options:
                        edits.append(ActionEdit('A2', dict(layer_id=lid, partition=rng.choice(options))))
                if len(edits) == 2:
                    add('C4', edits, (left, right))
        return tuple(plans)

    def _projected_groups(self, base, rng, report=None):
        """Additional joint proposals; original single-variable moves remain.

        Sample existing groups as well as adjacent merges. Tensor-heavy cuts
        receive an explicit proposal, with random representatives to avoid
        restricting group reachability to a handpicked chain.
        """
        groups = base.mapping.groups
        fused = {lid for s in base.mapping.fusion_stacks for lid in s.layer_ids}
        capacity = len(self.candidates[base.architecture_candidate_id].hardware.compute_slots)
        eligible = [i for i, g in enumerate(groups) if not fused.intersection(g.layer_ids)
                    and (g.mode == 'temporal' or 2 <= len(g.layer_ids) <= capacity)]
        edits, rows = [], []
        for i in rng.sample(eligible, min(2, len(eligible))):
            if groups[i].mode == 'temporal':
                cap = rng.choice(tuple(q for q in (16, 32, 64) if q <= capacity))
                placement = rng.choice(('preserve', 'spread'))
                for multiplier in (.5, 1., 2.):
                    try:
                        state, evidence = self.temporal_projector.project(base, groups[i].id, self.bank(base), report or {},
                            cap=cap, energy_multiplier=multiplier, placement=placement)
                        rows.append(DesignTrial(state, 'group_projection', groups[i].layer_ids, evidence))
                    except ValueError:
                        continue
                continue
            changed = (*groups[:i], replace(groups[i], mode='spatial'), *groups[i + 1:])
            edits.append((tuple(changed), groups[i].id, groups[i].layer_ids))
        merges = []
        for i in range(len(groups) - 1):
            members = (*groups[i].layer_ids, *groups[i + 1].layer_ids)
            if len(members) > min(capacity, 8) or fused.intersection(members):
                continue
            left, right = set(groups[i].layer_ids), set(groups[i + 1].layer_ids)
            cut = sum(self.w.tensor_map[t].bytes for lid in right for t in self.w.layer_map[lid].inputs
                      if self.w.tensor_map[t].producer in left)
            merges.append((cut, i, members))
        selected = []
        if merges:
            selected.append(max(merges, key=lambda row: (row[0], -row[1])))
            selected.extend(rng.sample(merges, min(2, len(merges))))
        for _, i, members in selected:
            group = LayerGroup(groups[i].id + '_project', members, 'spatial')
            edits.append(((*groups[:i], group, *groups[i + 2:]), group.id, members))
        for changed, gid, targets in edits:
            shadow = replace(base, mapping=replace(base.mapping, groups=tuple(changed)))
            try:
                state, evidence = self.projector.project(shadow, (gid,),
                    memory_weight=rng.choice((0., 1.)), placement=rng.choice(('preserve', 'spread')))
                rows.append(DesignTrial(state, 'group_projection', tuple(targets), evidence))
            except ValueError:
                continue
        return rows

    def propose(self, base, report, rng, excluded=()):
        started = time.perf_counter()
        self.calls += 1
        candidate = self.candidates[base.architecture_candidate_id]
        if self.operators == 'gemini5':
            rows = [DesignTrial(*row) for row in propose_five(self.w, base, candidate, rng)]
            seen, valid, rejected = set(excluded) | {decision_hash(base)}, [], Counter()
            rng.shuffle(rows)
            for row in rows:
                if row.fingerprint in seen:
                    continue
                try:
                    validate_state(self.w, row.state, self.candidates)
                except ValueError as error:
                    rejected[str(error)] += 1
                    continue
                seen.add(row.fingerprint)
                valid.append(row)
            valid = valid[:self.shortlist]
            return tuple(valid), dict(calls=self.calls, generation_wall_s=time.perf_counter() - started,
                operator_scope='gemini5', full_ppa=0, selected_families=dict(Counter(t.family for t in valid)),
                candidate_count=len(valid), rejected=dict(rejected))
        # Only architecture construction is factored out. The current-hardware
        # legacy transaction domain is passed through the original generator.
        if self.legacy_generator:
            plans, stats = generate_candidates(self.w, base, report, (candidate,), rng=rng,
                shortlist=max(48, self.shortlist), random_quota=max(12, self.shortlist // 3), max_per_family=16,
                proposal='composite' if self.composites else 'atomic', fusion_enabled=self.fusion,
                max_fusion_depth=self.max_depth, tile_sizes=((8, 8), (16, 16), (32, 32)), generator_guidance=self.guidance)
        else:
            plans = self._sample_transactions(base, report, rng)
            stats = dict(proposed=dict(Counter(p.family_id for p in plans)))
        extension = SingleArrayJointProposals(self.w, (candidate,), shortlist=max(48, self.shortlist),
            depth_cap=self.max_depth, transition_cap=12, timeout_s=60., fusion_enabled=self.fusion,
            generator_guidance=self.guidance)
        extension.calls = self.calls - 1
        # Original joint window generator is optional: lazy independent edits
        # already cover its complete underlying variables. Keep the exact old
        # generator for controlled legacy-proposal comparisons.
        if self.composites and self.legacy_generator:
            plans, stats = extension(base, report, plans, stats)
        rows = []
        for p in plans:
            try:
                rows.append(DesignTrial(apply_transaction(self.w, base, p, self.candidates), p.family_id, p.target_ids, p.to_dict()))
            except (ValueError, KeyError, IndexError, TypeError):
                continue
        bank = self.bank(base)
        # New complete mappings are built from the *current* design, so all
        # group/fusion changes survive subsequent structured mapping moves.
        for params in rng.sample(bank.domain, min(12, len(bank.domain))):
            try:
                t = bank.build(params, base=base)
                rows.append(DesignTrial(t.state, 'global_mapping', t.targets, dict(parameters=params)))
            except ValueError:
                pass
        for lid in rng.sample(tuple(self.w.layer_map), min(4, len(self.w.layers))):
            item = base.mapping.mapping_map[lid]
            if self.w.layer_map[lid].weight_shape:
                policy = 'cold' if item.weight_policy == 'persistent' else 'persistent'
                mapping = base.mapping.replace_layer(replace(item, weight_policy=policy))
                rows.append(DesignTrial(replace(base, mapping=mapping, evaluation_report=None), 'retention', (lid,), dict(weight_policy=policy)))
        if self.inner:
            changed = tuple(replace(m, dataflow=bank.inner_flow(m.layer_id, m.partition.as_tuple())) for m in base.mapping.layer_mappings)
            rows.append(DesignTrial(replace(base, mapping=replace(base.mapping, layer_mappings=changed), evaluation_report=None),
                                    'inner_dataflow', tuple(self.w.layer_map), dict(proxy_is_full_ppa=False)))
        if self.fusion:
            rows.extend(self._fusion_beam(base, rng))
        if self.resource_projection:
            rows.extend(self._projected_groups(base, rng, report))
        targets = [c for c in self.candidates.values() if c.id != candidate.id]
        rng.shuffle(targets)
        for c in targets[:min(8, len(targets))]:
            for legacy in (False, True):
                try:
                    rows.append(self.architecture_trial(base, c, legacy=legacy))
                except ValueError:
                    pass
        seen = set(excluded) | {decision_hash(base)}
        valid, rejects = [], Counter()
        for row in rows:
            if row.fingerprint in seen:
                continue
            try:
                validate_state(self.w, row.state, self.candidates)
            except ValueError as error:
                rejects[str(error)] += 1
                continue
            seen.add(row.fingerprint)
            valid.append(row)
        # Family round robin prevents 50 architecture choices from suppressing
        # Fusion or independent A2/A5/A8 actions. Remainder is seeded random.
        buckets = defaultdict(list)
        rng.shuffle(valid)
        for row in valid:
            buckets[row.family].append(row)
        selected = []
        while buckets and len(selected) < self.shortlist:
            keys = list(buckets)
            rng.shuffle(keys)
            for key in keys:
                if len(selected) >= self.shortlist:
                    break
                selected.append(buckets[key].pop())
                if not buckets[key]:
                    del buckets[key]
        stats = dict(calls=self.calls, generation_wall_s=time.perf_counter() - started,
                     v3=stats, rejected=dict(rejects), selected_families=dict(Counter(t.family for t in selected)),
                     candidate_count=len(selected), full_ppa=0)
        self.history = [stats]
        if len(bank.memo) > 2048:
            bank.memo.clear()
        return tuple(selected), stats

    def features(self, base, report, reference, trials, progress=0.):
        """Physical state and explicit proposed decisions; never oracle labels."""
        all_work = max(1, sum(l.macs_per_sample for l in self.w.layers))

        def design(state):
            h = self.candidates[state.architecture_candidate_id].hardware
            ms = state.mapping.layer_mappings
            n, slots = max(1, len(ms)), len(h.compute_slots)
            values = [h.compute_count / 16, h.rows / 16, h.cols / 16, h.io_count / 32,
                      len(state.mapping.groups) / n, sum(g.mode == 'temporal' for g in state.mapping.groups) / max(1, len(state.mapping.groups)),
                      len(state.mapping.fusion_stacks) / n,
                      max((len(s.layer_ids) for s in state.mapping.fusion_stacks), default=0) / n]
            qs = [len(m.chiplets) / slots for m in ms]
            values += [min(qs), sum(qs) / n, max(qs)]
            for axis in range(4):
                values.append(sum(math.log2(m.partition.as_tuple()[axis]) for m in ms) / (n * 8))
            for flow in ('output_stationary', 'weight_reuse', 'activation_reuse'):
                values.append(sum(self.w.layer_map[m.layer_id].macs_per_sample for m in ms if m.dataflow == flow) / all_work)
            values.append(sum(m.weight_policy == 'persistent' for m in ms) / n)
            for field in ('input_dram', 'weight_dram', 'output_dram'):
                values.extend((sum(getattr(m, field) == 'interleave' for m in ms) / n,
                               sum((int(getattr(m, field).split(':')[1]) + 1) / h.io_count if getattr(m, field).startswith('memory:') else 0 for m in ms) / n))
            for mode in ('recompute', 'horizontal_cache', 'full_cache'):
                values.append(sum(s.overlap == mode for s in state.mapping.fusion_stacks) / n)
            values += [sum(math.log2(s.tile_hw[a]) for s in state.mapping.fusion_stacks) / (n * 8) for a in (0, 1)]
            # Physical order/placement signatures encode assignment changes
            # that have the same PE quota. They are not performance scores.
            for field in range(4):
                values.append(sum(sum((j + 1) * (int(p.split('/')[0].split(':')[1]) * 16 + int(p.split(':')[-1]) + 1)
                                      for j, p in enumerate(m.chiplets)) * math.sin((i + 1) * (field + 1))
                                  / (len(m.chiplets) ** 2 * 256) for i, m in enumerate(ms)) / n)
            return np.asarray(values)

        before = design(base)
        ppa = [math.log(max(report[k], 1e-30) / max(reference[k], 1e-30)) for k in ('latency_s', 'energy_j', 'power_w', 'area_mm2')]
        evidence = {r['layer_id']: r for r in report.get('diagnostics', {}).get('layers', ())}
        order = {l.id: i / max(1, len(self.w.layers) - 1) for i, l in enumerate(self.w.layers)}
        result = []
        boundary_before = self.boundaries.sketch(base) if self.boundary_features else None
        for trial in trials:
            after = design(trial.state)
            targets = tuple(sorted(set(trial.targets) & set(self.w.layer_map), key=order.get))
            local = [len(targets) / len(self.w.layers), sum(self.w.layer_map[l].macs_per_sample for l in targets) / all_work]
            for key in ('criticality', 'memory_pressure', 'pe_utilization'):
                local.append(max((evidence.get(l, {}).get(key, 0) or 0 for l in targets), default=0) if self.guidance else 0.)
            # Distinguish equal-shape layers at different DAG positions even
            # with bottleneck guidance disabled; aggregate quotas alias them.
            positions = [order[l] for l in targets]
            count = max(1, len(targets))
            local.extend((sum(positions) / count, min(positions, default=0.), max(positions, default=0.)))
            for axis in (1, 2, 3):
                local.append(sum(math.log2(self.w.tensor_map[self.w.layer_map[l].output].shape[axis]) for l in targets) / (count * 16))
            local.extend((sum(len(self.w.layer_map[l].inputs) for l in targets) / (count * 4),
                          sum(len(self.w.consumers[self.w.layer_map[l].output]) for l in targets) / (count * 4),
                          sum(bool(self.w.layer_map[l].weight_shape) for l in targets) / count,
                          sum(base.mapping.group_map[l].mode == 'spatial' for l in targets) / count))
            vector = [1., *ppa, progress, *[float(trial.family == f) for f in FAMILIES], *before, *after, *(after - before), *local]
            if self.boundary_features:
                vector.extend(self.boundaries.encode(base, trial.state, before_sketch=boundary_before,
                                                      after_key=trial.fingerprint))
                vector.extend(self.boundaries.observed_edit(base, report, trial.state, trial.targets))
                vector.extend(self.locality.encode(base, report, trial.state))
            result.append(np.clip(vector, -8, 8))
        return np.asarray(result, dtype=np.float32)
