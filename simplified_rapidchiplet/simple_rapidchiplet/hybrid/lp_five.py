"""Lazy Gemini-style five-operator LP comparison under the INMIN oracle.

This preserves OP1-OP5 transformation semantics, not the original executable,
cost model or its stochastic proposal schedule. Group/architecture/inner
decisions are held fixed in this explicitly restricted comparison arm only.
"""
from dataclasses import replace
import random

from .actions import MappingAction, apply_action, partition_options


def propose_five(workload, base, candidate, rng, targets=12):
    plan, w = base.mapping, workload
    ids = rng.sample(tuple(w.layer_map), min(targets, len(w.layers)))
    actions = []
    for lid in ids:
        m = plan.mapping_map[lid]
        parts = [p for p in partition_options(w, lid, len(m.chiplets), 1) if p != m.partition.as_tuple()]
        for p in rng.sample(parts, min(2, len(parts))):
            actions.append(MappingAction('OP1', lid, partition=p))
        if len(m.chiplets) > 1:
            for _ in range(2):
                a, b = rng.sample(range(len(m.chiplets)), 2)
                actions.append(MappingAction('OP2', lid, index=a, other_index=b))
        fields = []
        layer = w.layer_map[lid]
        if any(t in w.input_ids for t in layer.inputs):
            fields.append('input_dram')
        if layer.weight_bytes:
            fields.append('weight_dram')
        if layer.output in w.output_ids or any(plan.group_map[u].id != plan.group_map[lid].id for u in w.consumers[layer.output]):
            fields.append('output_dram')
        for field in fields:
            memories = [name for name in (*plan.memory_ids, 'interleave') if name != getattr(m, field)]
            for memory in rng.sample(memories, min(2, len(memories))):
                actions.append(MappingAction('OP5', lid, field=field, memory=memory))
    spatial = [g for g in plan.groups if g.mode == 'spatial' and len(g.layer_ids) > 1]
    for group in rng.sample(spatial, min(8, len(spatial))):
        left, right = rng.sample(group.layer_ids, 2)
        a, b = plan.mapping_map[left], plan.mapping_map[right]
        actions.append(MappingAction('OP3', left, right, rng.randrange(len(a.chiplets)), rng.randrange(len(b.chiplets))))
        for donor, recipient in ((left, right), (right, left)):
            dm, rm = plan.mapping_map[donor], plan.mapping_map[recipient]
            if len(dm.chiplets) <= 1:
                continue
            dp = partition_options(w, donor, len(dm.chiplets) - 1, 1)
            rp = partition_options(w, recipient, len(rm.chiplets) + 1, 1)
            if dp and rp:
                actions.append(MappingAction('OP4', donor, recipient, rng.randrange(len(dm.chiplets)),
                    rng.randrange(len(rm.chiplets) + 1), rng.choice(dp), rng.choice(rp)))
    family = dict(OP1='A2', OP2='A5', OP3='A6', OP4='A4', OP5='A8')
    rows = []
    for action in actions:
        mapping = apply_action(plan, action, core_ids=candidate.hardware.compute_slots)
        state = replace(base, mapping=mapping, evaluation_report=None, remaining_budget=None)
        rows.append((state, family[action.operator], tuple(x for x in (action.layer_id, action.other_layer_id) if x),
                     dict(mapping_action=action.to_dict(), semantics='Gemini-style OP1-OP5; same INMIN oracle; not original executable')))
    return rows
