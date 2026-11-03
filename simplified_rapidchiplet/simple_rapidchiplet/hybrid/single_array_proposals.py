"""Bounded joint mapping/fusion candidate generator for schema v2 only.

The Cartesian search is a proposal mechanism. Structural legality is checked
here; exact tiled capacity, communication and PPA remain the global oracle's
decision. No proxy improvement is fabricated or added to the reward.
"""
from collections import Counter
from dataclasses import replace
from itertools import product
import time

from .actions import partition_options
from .depth_first import legal_stack
from .mapping import LayerGroup, Partition
from .rl_actions import _group_dict, _has_overlap, apply_transaction
from .rl_schema import ActionEdit, make_action_plan, stable_hash


class SingleArrayJointProposals:
    version = 'single_array_joint_window_v1'

    def __init__(self, workload, candidates, *, shortlist=48, depth_cap=4, transition_cap=36,
                 timeout_s=2., fusion_enabled=True, generator_guidance=True):
        self.generator_guidance = generator_guidance
        self.workload = workload
        self.candidates = {c.id: c for c in candidates}
        self.shortlist, self.depth_cap, self.transition_cap = shortlist, depth_cap, transition_cap
        self.timeout_s, self.fusion_enabled = timeout_s, fusion_enabled
        self.calls, self.history, self.deadline = 0, [], None

    def state_dict(self):
        return {'calls': self.calls, 'history': self.history}

    def load_state_dict(self, data):
        self.calls, self.history = data['calls'], data['history']

    def _regroup(self, mapping, ids, mode):
        members, groups = set(ids), []
        inserted = False
        for group in mapping.groups:
            before, after = [], []
            crossed = False
            for lid in group.layer_ids:
                if lid in members:
                    crossed = True
                else:
                    (after if crossed or inserted else before).append(lid)
            if before:
                groups.append(LayerGroup(group.id + '_before', tuple(before), group.mode))
            if members.intersection(group.layer_ids) and not inserted:
                groups.append(LayerGroup('joint_' + stable_hash(ids)[:8], ids, mode))
                inserted = True
            if after:
                groups.append(LayerGroup(group.id + '_after', tuple(after), group.mode))
        return tuple(groups)

    def __call__(self, state, report, plans, stats):
        if state.version < 2:
            raise ValueError('v0.2 joint proposals require design schema 2')
        self.calls += 1
        started = time.monotonic()
        deadline = min(self.deadline or float('inf'), started + self.timeout_s)
        order = [layer.id for layer in self.workload.layers]
        fused = {lid for stack in state.mapping.fusion_stacks for lid in stack.layer_ids}
        windows = [tuple(order[i:i + depth]) for depth in range(2, min(self.depth_cap, len(order)) + 1)
                   for i in range(len(order) - depth + 1)
                   if not fused.intersection(order[i:i + depth]) and legal_stack(self.workload, order[i:i + depth])]
        window = windows[(self.calls - 1) % len(windows)] if windows else ()
        window_reason = 'generic deterministic window coverage'
        layer_evidence = {row['layer_id']: row for row in report.get('diagnostics', {}).get('layers', ())} if self.generator_guidance else {}
        if windows and layer_evidence and self.calls % 2:
            def pressure(ids):
                return sum((row.get('criticality', 0.) or 0.) * (
                    (row.get('resource_wait_s', 0.) or 0.) + (row.get('transfer_queue_wait_s', 0.) or 0.) +
                    (row.get('compute_busy_s', 0.) or 0.) + (row.get('vector_busy_s', 0.) or 0.))
                    for row in (layer_evidence.get(lid, {}) for lid in ids))
            ranked = sorted(windows, key=lambda ids: (-pressure(ids), ids))
            if pressure(ranked[0]) > 0:
                window = ranked[((self.calls - 1) // 2) % min(3, len(ranked))]
                window_reason = 'partial critical-calendar busy/queue evidence; proposal heuristic only, overlapping waits not added as latency'
        extra, attempts, rejected = [], [], Counter()
        hardware = self.candidates[state.architecture_candidate_id].hardware
        slots = hardware.compute_slots
        if window:
            combinations = list(product(('temporal', 'spatial'), (1, 4, 16), ('K', 'P'), (0, 1),
                                        ((2, 2), (4, 4), (8, 8)), ('recompute', 'horizontal_cache', 'full_cache')))
            # Rotate the bounded joint domain between calls. Rotation affects
            # coverage, never policy features or an evaluator's cost.
            offset = ((self.calls - 1) * self.transition_cap) % len(combinations)
            combinations = combinations[offset:] + combinations[:offset]
            # Interleave both group modes and partition directions in each
            # bounded tranche, rather than exhausting one product prefix.
            combinations.sort(key=lambda row: stable_hash({'call': self.calls, 'choice': row}))
            for mode, requested, direction, placement, tile, overlap in combinations:
                if len(attempts) >= self.transition_cap or time.monotonic() >= deadline:
                    break
                evidence = {'depth': len(window), 'mode': mode, 'requested_quota': requested,
                            'partition_direction': direction, 'placement': placement, 'tile_hw': tile, 'cache': overlap}
                attempts.append(evidence)
                if overlap != 'recompute' and not _has_overlap(self.workload, window):
                    evidence['status'] = 'hard_rejected_no_overlap'
                    rejected[evidence['status']] += 1
                    continue
                groups = self._regroup(state.mapping, window, mode)
                allocated, cursor = {}, 0
                rotated = slots[placement:] + slots[:placement]
                for lid in window:
                    quota = min(requested, len(slots) // len(window) if mode == 'spatial' else len(slots))
                    options = ()
                    while quota:
                        options = partition_options(self.workload, lid, quota, 1)
                        if options:
                            break
                        quota -= 1
                    if not options:
                        break
                    preferred = [p for p in options if p[1] > 1 and p[2] * p[3] == 1] if direction == 'K' else [p for p in options if p[1] == 1]
                    part = (preferred or list(options))[0]
                    selected = rotated[cursor:cursor + quota] if mode == 'spatial' else rotated[:quota]
                    cursor += quota if mode == 'spatial' else 0
                    allocated[lid] = {'cores': list(selected), 'partition': list(part)}
                if len(allocated) != len(window):
                    evidence['status'] = 'hard_rejected_integer_partition'
                    rejected[evidence['status']] += 1
                    continue
                edits = [ActionEdit('A10', {'groups': _group_dict(groups)}), ActionEdit('A4', {'allocations': allocated})]
                pair = stable_hash({'groups': _group_dict(groups), 'allocations': allocated, 'window': window})[:16]
                off = make_action_plan(state, 'C9', edits, targets=window,
                    reason='paired joint mapping counterfactual with fusion off', sources=(self.version, 'pair:' + pair, 'DF_off'))
                h, v = overlap != 'recompute', overlap == 'full_cache'
                on = make_action_plan(state, 'C6', [*edits, ActionEdit('A1', {'operation': 'merge', 'layer_ids': list(window),
                    'tile_hw': list(tile), 'cache_h': h, 'cache_v': v})], targets=window,
                    reason='bounded group+quota+placement+partition+depth+tile+cache joint proposal; full PPA decides',
                    sources=(self.version, 'pair:' + pair, 'DF_on'))
                try:
                    apply_transaction(self.workload, state, off, self.candidates)
                    if self.fusion_enabled:
                        apply_transaction(self.workload, state, on, self.candidates)
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    evidence['status'] = 'structurally_rejected'
                    evidence['reason'] = str(error)
                    rejected[str(error)] += 1
                    continue
                evidence['status'] = 'structurally_legal_physical_check_pending'
                extra.extend((off, on) if self.fusion_enabled else (off,))
        memory = []
        if state.version >= 3 and self.generator_guidance and self.fusion_enabled:
            from .target_bottleneck import memory_actions
            for candidate in memory_actions(self.workload, state, report):
                try:
                    apply_transaction(self.workload, state, candidate, self.candidates)
                    memory.append(candidate)
                except ValueError as error:
                    rejected['memory_transaction:' + str(error)] += 1
        # Preserve architecture jumps, generic atomic and joint proposals in
        # the same policy pool. A single sampled ActionPlan commits one state.
        jumps = [p for p in plans if any(e.action_id == 'A9' for e in p.ordered_edits)]
        if state.version >= 3:
            jumps = jumps[:min(12, self.shortlist // 4)]
        atoms = [p for p in plans if p.family_id.startswith('A') and p not in jumps]
        joint = extra[:min(8, max(0, self.shortlist - len(jumps) - 4))]
        flows = [p for p in atoms if p.family_id == 'A12'][:3] if state.version >= 3 else []
        pool = (*memory[:4], *flows, *jumps, *atoms[:4], *joint, *plans, *extra)
        valid, seen, kept_extra = [], set(), set()
        excluded = set(stats.get('excluded_state_hashes', ())) | set(stats.get('budget_excluded_state_hashes', ()))
        for plan in pool:
            trial = apply_transaction(self.workload, state, plan, self.candidates)
            if trial.fingerprint == state.fingerprint or trial.fingerprint in seen or trial.fingerprint in excluded:
                continue
            seen.add(trial.fingerprint)
            valid.append(plan)
            if plan in extra:
                kept_extra.add(plan.id)
            if len(valid) >= self.shortlist:
                break
        evidence = {'window': window, 'window_reason': window_reason, 'attempts': attempts, 'rejected': dict(rejected), 'generated': len(extra),
                    'memory_pressure_proposals': len(memory), 'shortlisted_joint': len(kept_extra), 'elapsed_s': time.monotonic() - started,
                    'full_evaluations': 0, 'physical_validity': 'global oracle pending',
                    'domain_truncated': bool(window) and len(attempts) < len(combinations)}
        self.history.append(evidence)
        stats['joint_window'] = evidence
        for plan in (*extra, *memory):
            legal = stats.setdefault('structurally_legal_by_family', {})
            legal[plan.family_id] = legal.get(plan.family_id, 0) + 1
            stats['proposed'][plan.family_id] = stats['proposed'].get(plan.family_id, 0) + 1
        stats['selected_families'] = dict(Counter(p.family_id for p in valid))
        return tuple(valid), stats
