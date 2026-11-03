"""Transactional atomic edits and diagnosis-guided, explicit macro proposals.

Only executable semantics produce a hard mask. Shortlisting is approximate;
generic candidates remain available independently of bottleneck diagnostics.
No proposal modifies the physical estimator or guarantees an improvement.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import replace
from functools import lru_cache
import math
import random

from .actions import (apply_action, legal_actions, MappingAction, partition_options,
                      partition_tuple)
from .depth_first import legal_stack
from .hardware import build_package
from .mapping import FusionStack, LayerGroup, Partition, validate_plan as validate_mapping
from .rl_schema import ActionEdit, CAPABILITIES, make_action_plan, stable_hash
from .search import plan_from_dict


OVERLAP_TO_FLAGS = {"recompute": (False, False), "horizontal_cache": (True, False),
                    "full_cache": (True, True)}
FLAGS_TO_OVERLAP = {flags: name for name, flags in OVERLAP_TO_FLAGS.items()}


def _has_overlap(workload, layer_ids):
    return any(any(dilation * (kernel - 1) + 1 > stride
                   for dilation, kernel, stride in zip(workload.layer_map[layer].dilation,
                                                       workload.layer_map[layer].kernel,
                                                       workload.layer_map[layer].stride))
               for layer in layer_ids)


@lru_cache(maxsize=128)
def _package(spec):
    if getattr(spec, 'profile_id', '').startswith('chiplet_single_pe_array_v0_'):
        from .single_array_hardware import build_single_array_package
        return build_single_array_package(spec)
    return build_package(spec)


def _candidate_map(candidates):
    return candidates if isinstance(candidates, dict) else {candidate.id: candidate for candidate in candidates}


def validate_state(workload, state, architecture_candidates):
    candidates = _candidate_map(architecture_candidates)
    if state.version not in (1, 2, 3) or state.model_hash != workload.fingerprint:
        raise ValueError("design schema/model identity differs")
    if state.architecture_candidate_id not in candidates:
        raise ValueError("unknown legal architecture candidate")
    candidate = candidates[state.architecture_candidate_id]
    single_array = getattr(candidate.hardware, 'profile_id', '').startswith('chiplet_single_pe_array_v0_')
    if single_array != (state.version >= 2) or (single_array and (state.version == 3) != (candidate.hardware.profile_id.endswith('_3'))):
        raise ValueError('hardware profile requires its matching design schema')
    if single_array and (state.hardware_template_version != candidate.hardware.profile_id or
                         workload.batch_size != 1 or state.mapping.microbatch_size != 1):
        raise ValueError('v0.2 requires its hardware identity and Batch/Microbatch=1')
    package = _package(candidate.hardware)
    for key in ("activation_bits", "weight_bits", "psum_bits"):
        if state.precision_config.get(key) != getattr(candidate.hardware, key):
            raise ValueError("precision changes are unsupported by these actions")
    if state.routing_config != {"algorithm": "template_deterministic"}:
        raise ValueError("arbitrary routing is unsupported")
    if state.scheduler_config != {"algorithm": "resource_calendar"}:
        raise ValueError("unsupported scheduler configuration")
    if tuple(state.mapping.memory_ids) != tuple(package.memory_ids):
        raise ValueError("mapping memory endpoints must match the architecture")
    validate_mapping(workload, state.mapping, package.mapping_core_ids)
    return True


def _stack_index(mapping, layer_ids):
    ids = tuple(layer_ids)
    for index, stack in enumerate(mapping.fusion_stacks):
        if tuple(stack.layer_ids) == ids:
            return index
    raise ValueError("unknown fused stack")


def _apply_edit(workload, state, edit, candidates):
    params, mapping = edit.parameters, state.mapping
    action = edit.action_id
    package = _package(candidates[state.architecture_candidate_id].hardware)
    if action == "A1":
        mode = params.get("operation", "merge")
        ids = tuple(params["layer_ids"])
        if mode in ("merge", "shift_boundary"):
            if not legal_stack(workload, ids):
                raise ValueError("fusion requires a closed contiguous single-exit subgraph")
            if mode == "shift_boundary":
                previous = tuple(params["previous_layer_ids"])
                _stack_index(mapping, previous)
            else:
                previous = ()
            stacks = tuple(stack for stack in mapping.fusion_stacks
                           if tuple(stack.layer_ids) != previous)
            if any(set(stack.layer_ids) & set(ids) for stack in stacks):
                raise ValueError("fusion stacks cannot overlap")
            tile = tuple(params.get("tile_hw", (8, 8)))
            cache_h, cache_v = bool(params.get("cache_h", False)), bool(params.get("cache_v", False))
            if (cache_h, cache_v) not in FLAGS_TO_OVERLAP:
                raise ValueError("vertical-only overlap cache is unsupported")
            if (cache_h or cache_v) and not _has_overlap(workload, ids):
                raise ValueError("overlap cache has no overlapping receptive fields")
            mapping = replace(mapping, fusion_stacks=(*stacks, FusionStack(ids, tile, FLAGS_TO_OVERLAP[cache_h, cache_v])))
        elif mode == "split":
            index = _stack_index(mapping, ids)
            cut = params.get("boundary")
            stacks = list(mapping.fusion_stacks)
            old = stacks.pop(index)
            # Removing fusion retains every original DAG edge. A partial split
            # requires explicit A10 groups because of the backend stack scope.
            if cut is not None:
                if not 0 < int(cut) < len(ids):
                    raise ValueError("fusion split boundary must be internal")
                for part in (ids[:int(cut)], ids[int(cut):]):
                    if len(part) > 1:
                        stacks.append(replace(old, layer_ids=part))
            mapping = replace(mapping, fusion_stacks=tuple(stacks))
        else:
            raise ValueError("unsupported fusion operation")
    elif action == "A2":
        layer_id = params["layer_id"]
        item = mapping.mapping_map[layer_id]
        mapping = mapping.replace_layer(replace(item, partition=Partition(*params["partition"])))
    elif action == "A3":
        index = _stack_index(mapping, params["layer_ids"])
        stacks = list(mapping.fusion_stacks)
        stacks[index] = replace(stacks[index], tile_hw=tuple(params["tile_hw"]))
        mapping = replace(mapping, fusion_stacks=tuple(stacks))
    elif action == "A4":
        if "mapping_action" in params:
            low = MappingAction.from_dict(params["mapping_action"])
            if low.operator != "OP4":
                raise ValueError("A4 requires a resource-transfer operator")
            mapping = apply_action(mapping, low, core_ids=package.mapping_core_ids)
        else:
            allocations = params["allocations"]
            if not allocations:
                raise ValueError("resource transfer needs explicit allocations")
            groups = {mapping.group_map[layer_id].id for layer_id in allocations}
            if len(groups) != 1 or (state.version == 1 and len(allocations) > 1 and mapping.group_map[next(iter(allocations))].mode != "spatial"):
                raise ValueError("resource transfer requires a common simultaneous resource pool")
            for layer_id, allocation in allocations.items():
                item = mapping.mapping_map[layer_id]
                mapping = mapping.replace_layer(replace(item, chiplets=tuple(allocation["cores"]),
                                                       partition=Partition(*allocation["partition"])))
    elif action == "A5":
        item = mapping.mapping_map[params["layer_id"]]
        cores = tuple(params["cores"])
        if len(cores) != len(item.chiplets):
            raise ValueError("within-layer placement must preserve quota; use A4")
        mapping = mapping.replace_layer(replace(item, chiplets=cores))
    elif action == "A6":
        left, right = params["left_layer_id"], params["right_layer_id"]
        if left == right:
            raise ValueError("cross-layer swap requires two layers")
        lg, rg = mapping.group_map[left], mapping.group_map[right]
        if lg.id != rg.id or lg.mode != "spatial":
            raise ValueError("cross-layer swap requires one spatial group")
        li, ri = tuple(params["left_indices"]), tuple(params["right_indices"])
        if not li or len(li) != len(ri) or len(set(li)) != len(li) or len(set(ri)) != len(ri):
            raise ValueError("cross-layer swap requires distinct equal-size sets")
        litem, ritem = mapping.mapping_map[left], mapping.mapping_map[right]
        lc, rc = list(litem.chiplets), list(ritem.chiplets)
        if any(type(i) is not int or not 0 <= i < len(lc) for i in li) or any(type(i) is not int or not 0 <= i < len(rc) for i in ri):
            raise ValueError("cross-layer index out of range")
        for a, b in zip(li, ri):
            lc[a], rc[b] = rc[b], lc[a]
        mapping = mapping.replace_layer(replace(litem, chiplets=tuple(lc))).replace_layer(replace(ritem, chiplets=tuple(rc)))
    elif action == "A7":
        flags = params["cache_h"], params["cache_v"]
        if any(type(value) is not bool for value in flags) or flags not in FLAGS_TO_OVERLAP:
            raise ValueError("unsupported overlap booleans; vertical-only is unavailable")
        if any(flags) and not _has_overlap(workload, params["layer_ids"]):
            raise ValueError("overlap cache has no overlapping receptive fields")
        index = _stack_index(mapping, params["layer_ids"])
        stacks = list(mapping.fusion_stacks)
        stacks[index] = replace(stacks[index], overlap=FLAGS_TO_OVERLAP[flags])
        mapping = replace(mapping, fusion_stacks=tuple(stacks))
    elif action == "A12":
        if state.version < 3:
            raise ValueError("A12 dataflow switching requires schema3")
        if params['dataflow'] not in ('output_stationary', 'weight_reuse', 'activation_reuse'):
            raise ValueError("unpriced dataflow")
        item = mapping.mapping_map[params['layer_id']]
        if not workload.layer_map[item.layer_id].weight_shape:
            raise ValueError("dataflow switching requires a MAC layer")
        mapping = mapping.replace_layer(replace(item, dataflow=params['dataflow']))
    elif action == "A8":
        tensor_id, memory = params["tensor_id"], params["configuration_id"]
        if memory not in (*package.memory_ids, "interleave"):
            raise ValueError("unsupported DRAM configuration")
        if tensor_id.startswith("weight:"):
            layer_id = tensor_id[len("weight:"):]
            if not workload.layer_map[layer_id].weight_bytes:
                raise ValueError("layer has no external weights")
            item = mapping.mapping_map[layer_id]
            mapping = mapping.replace_layer(replace(item, weight_dram=memory))
        else:
            tensor = workload.tensor_map[tensor_id]
            if tensor.producer:
                item = mapping.mapping_map[tensor.producer]
                mapping = mapping.replace_layer(replace(item, output_dram=memory))
            for layer_id in workload.consumers[tensor_id]:
                item = mapping.mapping_map[layer_id]
                mapping = mapping.replace_layer(replace(item, input_dram=memory))
    elif action == "A9":
        target = params["architecture_candidate_id"]
        if target not in candidates:
            raise ValueError("unknown legal architecture candidate")
        # A complete target mapping is mandatory: there is no hidden free
        # allocation, migration, data replication, or unchanged dangling core.
        if "mapping" not in params:
            raise ValueError("architecture switch requires explicit evacuation/reallocation mapping")
        mapping = plan_from_dict(params["mapping"])
        return replace(state, architecture_candidate_id=target, mapping=mapping, evaluation_report=None)
    elif action == "A10":
        groups = tuple(LayerGroup(row["id"], tuple(row["layer_ids"]), row.get("mode", "spatial"))
                       for row in params["groups"])
        if "mapping" in params:
            mapping = plan_from_dict(params["mapping"])
        mapping = replace(mapping, groups=groups, microbatch_size=params.get("microbatch_size", mapping.microbatch_size))
    return replace(state, mapping=mapping, evaluation_report=None)


def apply_transaction(workload, state, plan, architecture_candidates):
    """All edits run on a shadow state, with one final semantic validation."""
    if plan.base_state_hash != state.fingerprint:
        raise ValueError("stale action plan: base state identity differs")
    candidates = _candidate_map(architecture_candidates)
    if state.version >= 2 and plan.family_id.startswith('C'):
        actions = [edit.action_id for edit in plan.ordered_edits]
        required = {'C1': {'A4'}, 'C2': {'A9'}, 'C3': {'A9'}, 'C4': {'A2'},
                    'C5': {'A8'}, 'C6': {'A10', 'A4', 'A1'}, 'C7': {'A3', 'A7'},
                    'C8': {'A3', 'A7'}, 'C9': {'A10', 'A4'}}
        if plan.family_id not in required or not required[plan.family_id].issubset(actions):
            raise ValueError('unsupported or incomplete v0.2 composite transaction semantics')
        if plan.family_id == 'C4' and len({edit.parameters.get('layer_id') for edit in plan.ordered_edits if edit.action_id == 'A2'}) < 2:
            raise ValueError('C4 requires at least two linked partition edits')
        if plan.family_id == 'C5' and not {'A5', 'A6'}.intersection(actions):
            raise ValueError('C5 requires placement and external storage edits')
    # All edits invalidate the derived report; copying it costs CPU but cannot
    # affect any explicit mapping decision or the transaction's identity.
    trial = copy.deepcopy(replace(state,evaluation_report=None))
    for edit in plan.ordered_edits:
        trial = _apply_edit(workload, trial, edit, candidates)
    validate_state(workload, trial, candidates)
    if state.version >= 2 and plan.family_id in ('C2', 'C3'):
        source = candidates[state.architecture_candidate_id].hardware.compute_count
        target = candidates[trial.architecture_candidate_id].hardware.compute_count
        if (plan.family_id == 'C2' and target <= source) or (plan.family_id == 'C3' and target >= source):
            raise ValueError('C2/C3 must respectively increase/decrease physical compute count')
    return trial


def validate_plan(workload, state, plan, architecture_candidates):
    try:
        trial = apply_transaction(workload, state, plan, architecture_candidates)
        return {"valid": True, "reason": None, "no_op": trial.fingerprint == state.fingerprint}
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return {"valid": False, "reason": str(error), "no_op": False}


def diagnose(state, report):
    """Use measured wait/criticality; absent evidence stays unknown."""
    diagnostics = report.get("diagnostics", {})
    layers = diagnostics.get("layers", [])
    if isinstance(layers, dict):
        layers = [{"layer_id": key, **value} for key, value in layers.items()]
    compute = [row for row in layers if row.get("bottleneck") == "compute" and row.get("has_parallelism", True) and
               (row.get("criticality", 0) or row.get("slack_s") == 0)]
    donors = [row for row in layers if isinstance(row.get("slack_s"), (int, float)) and row["slack_s"] > 0]
    networks = diagnostics.get("network", {})
    windows = networks.get("time_windows", networks.get("windows", [])) if isinstance(networks, dict) else [
        {**window, "source": link.get("source"), "destination": link.get("destination")}
        for link in networks if isinstance(link, dict) for window in link.get("time_windows", [])] if isinstance(networks, list) else []
    hotspots = [row for row in windows if (row.get("queue_wait_s") or row.get("wait_s") or 0) > 0]
    memory = diagnostics.get("memory", {})
    buffers = memory.get("buffers", []) if isinstance(memory, dict) else memory if isinstance(memory, list) else []
    pressure = [row for row in buffers if (row.get("spill_bytes") or 0) > 0 or
                (isinstance(row.get("peak_live_bytes"), (int, float)) and
                 isinstance(row.get("capacity_bytes"), (int, float)) and
                 row["peak_live_bytes"] > .9 * row["capacity_bytes"])]
    return {"compute_targets": [row["layer_id"] for row in compute],
            "donor_candidates": [row["layer_id"] for row in donors],
            "hotspots": hotspots, "buffer_pressure": pressure,
            "layers": layers, "flows": diagnostics.get("flows", []),
            "df_stacks": diagnostics.get("df_stacks", []),
            "evidence": "measured" if layers else "unknown"}


def _translate(state, low):
    if low.operator == "OP1":
        return "A2", {"layer_id": low.layer_id, "partition": list(low.partition)}
    if low.operator == "OP2":
        cores = list(state.mapping.mapping_map[low.layer_id].chiplets)
        cores[low.index], cores[low.other_index] = cores[low.other_index], cores[low.index]
        return "A5", {"layer_id": low.layer_id, "cores": cores}
    if low.operator == "OP3":
        return "A6", {"left_layer_id": low.layer_id, "right_layer_id": low.other_layer_id,
                      "left_indices": [low.index], "right_indices": [low.other_index]}
    if low.operator == "OP4":
        return "A4", {"mapping_action": low.to_dict()}
    item = state.mapping.mapping_map[low.layer_id]
    if low.field == "weight_dram":
        tensor = "weight:" + low.layer_id
    elif low.field == "output_dram":
        tensor = None  # filled by the caller using the workload DAG
    else:
        tensor = None
    return "A8", {"tensor_id": tensor, "configuration_id": low.memory, "_field": low.field, "_layer": item.layer_id}


def _part(workload, layer_id, count, microbatch, previous=None):
    options = partition_options(workload, layer_id, count, microbatch)
    if not options:
        return None
    if previous is None:
        return options[0]
    return min(options, key=lambda option: sum(abs(math.log(a / b)) for a, b in zip(option, previous)))


def resized_mapping(workload, state, target):
    """Explicit integer allocation proposal; no physical cost is invented.

    Preserve groups and mapping options; allocate proportionally to current
    shares and then redistribute idle slots only to exact legal partitions.
    All endpoint/storage updates are returned in the action parameters.
    """
    base, package = state.mapping, _package(target.hardware)
    cores, memories = tuple(package.mapping_core_ids), tuple(package.memory_ids)
    mappings = dict(base.mapping_map)
    for group in base.groups:
        if group.mode == "spatial" and len(group.layer_ids) > len(cores):
            raise ValueError("smaller architecture requires an explicit pipeline split")
        counts = {}
        if group.mode == "spatial":
            old_total = sum(len(mappings[layer].chiplets) for layer in group.layer_ids)
            remaining = len(cores)
            for index, layer in enumerate(group.layer_ids):
                count = max(1, round(len(base.mapping_map[layer].chiplets) * len(cores) / old_total))
                count = min(count, remaining - len(group.layer_ids) + index + 1)
                while count > 1 and _part(workload, layer, count, base.microbatch_size) is None:
                    count -= 1
                counts[layer] = count
                remaining -= count
            while remaining:
                choices = []
                for layer in group.layer_ids:
                    for extra in range(1, remaining + 1):
                        if _part(workload, layer, counts[layer] + extra, base.microbatch_size) is not None:
                            demand = max(workload.layer_map[layer].macs_per_sample, 1) / counts[layer]
                            choices.append((-demand, extra, layer))
                            break
                if not choices:
                    break
                _, extra, layer = min(choices)
                counts[layer] += extra
                remaining -= extra
        else:
            for layer in group.layer_ids:
                count = len(cores)
                while count > 1 and _part(workload, layer, count, base.microbatch_size) is None:
                    count -= 1
                counts[layer] = count
        offset = 0
        for layer in group.layer_ids:
            item = base.mapping_map[layer]
            count = counts[layer]
            selection = cores[offset:offset + count] if group.mode == "spatial" else cores[:count]
            offset += count
            fields = {name: getattr(item, name) if getattr(item, name) in (*memories, "interleave", "direct") else memories[0]
                      for name in ("input_dram", "weight_dram", "output_dram")}
            mappings[layer] = replace(item, chiplets=selection,
                                      partition=Partition(*_part(workload, layer, count, base.microbatch_size, partition_tuple(item.partition))),
                                      **fields)
    return replace(base, layer_mappings=tuple(mappings[item.layer_id] for item in base.layer_mappings), memory_ids=memories)


def _group_dict(groups):
    return [{"id": group.id, "layer_ids": list(group.layer_ids), "mode": group.mode} for group in groups]


def _hotspot_storage_owners(workload, state, diagnosis):
    """Join actual queued time-window flows to their executed unit owners.

    A compact/truncated report may omit a flow. Such an unidentifiable flow
    cannot justify a targeted C5; generic placement candidates still survive.
    Only paid external/materialized storage participates in a DRAM edit.
    """
    flow_ids = {row.get("flow_id") for row in diagnosis["hotspots"] if row.get("flow_id")}
    owners = {}
    for flow in diagnosis["flows"]:
        flow_id = flow.get("flow_id", flow.get("id"))
        if flow_id not in flow_ids:
            continue
        unit_ids = (*flow.get("producer_units", ()), *flow.get("consumer_units", ()))
        layer_ids = {layer.id for layer in workload.layers for unit in unit_ids
                     if unit.startswith(layer.id + ":") or unit.startswith("df:" + layer.id + ":")}
        tensor_id = flow.get("tensor_id", flow.get("tensor"))
        storage_id = None
        if isinstance(tensor_id, str) and tensor_id.startswith("weights:"):
            layer_id = tensor_id[len("weights:"):]
            if layer_id in workload.layer_map:
                layer_ids.add(layer_id)
                storage_id = "weight:" + layer_id
        elif tensor_id in workload.tensor_map:
            tensor = workload.tensor_map[tensor_id]
            if tensor_id in workload.input_ids:
                storage_id = tensor_id
            elif tensor.producer:
                source_group = state.mapping.group_map[tensor.producer].id
                if tensor_id in workload.output_ids or any(state.mapping.group_map[user].id != source_group
                                                           for user in workload.consumers[tensor_id]):
                    storage_id = tensor_id
        if storage_id is not None:
            for layer_id in layer_ids:
                owners.setdefault(layer_id, set()).add(storage_id)
    return owners, flow_ids


def _storage_selector(workload, mapping, layer_id, tensor_id):
    if tensor_id.startswith("weight:"):
        return mapping.mapping_map[tensor_id[len("weight:"):]].weight_dram
    tensor = workload.tensor_map[tensor_id]
    if tensor.producer:
        return mapping.mapping_map[tensor.producer].output_dram
    return mapping.mapping_map[layer_id].input_dram


def _realization_edits(workload, mapping, layer_ids, *, storage=False):
    """Expose all linked partition/placement/storage decisions in a macro.

    Allocation edits already maintain a valid partition as a necessary atomic
    postcondition. Repeating that partition here documents its exact realized
    parameters; it does not run a mapper or earn another reward/evaluation.
    """
    edits, stored = [], set()
    for layer_id in layer_ids:
        item, layer = mapping.mapping_map[layer_id], workload.layer_map[layer_id]
        edits.extend((ActionEdit("A2", {"layer_id": layer_id, "partition": list(item.partition.as_tuple())}),
                      ActionEdit("A5", {"layer_id": layer_id, "cores": list(item.chiplets)})))
        if storage:
            assignments = []
            if layer.weight_bytes:
                assignments.append(("weight:" + layer_id, item.weight_dram))
            if item.output_dram != "direct":
                assignments.append((layer.output, item.output_dram))
            assignments.extend((tensor, item.input_dram) for tensor in layer.inputs
                               if tensor in workload.input_ids and item.input_dram != "direct")
            for tensor, memory in assignments:
                if tensor not in stored:
                    edits.append(ActionEdit("A8", {"tensor_id": tensor, "configuration_id": memory}))
                    stored.add(tensor)
    return edits


def generate_candidates(workload, state, report, architecture_candidates, *, rng=None,
                        shortlist=48, random_quota=12, max_per_family=12,
                        proposal="composite", fusion_enabled=True, max_fusion_depth=4,
                        tile_sizes=((8, 8), (16, 16)), excluded_state_hashes=(), budget_excluded_state_hashes=(),
                        search_space="joint", generator_guidance=True):
    """Return plans and three-stage proposal accounting, without oracle calls."""
    if proposal not in {"atomic", "guided_atomic", "composite"}:
        raise ValueError("proposal must be atomic, guided_atomic, or composite")
    if min(shortlist, random_quota, max_per_family) < 1:
        raise ValueError("candidate bounds and generic quota must be positive")
    if proposal == "composite" and shortlist < 2:
        raise ValueError("composite shortlist must be at least 2 to reserve both generic and macro candidates")
    rng = rng or random.Random(0)
    excluded_state_hashes = set(excluded_state_hashes)
    budget_excluded_state_hashes = set(budget_excluded_state_hashes)
    candidates = _candidate_map(architecture_candidates)
    current = candidates[state.architecture_candidate_id]
    package = _package(current.hardware)
    diagnosis = diagnose(state, report if generator_guidance else {})
    if not generator_guidance:
        diagnosis["compute_targets"] = []
        diagnosis["donor_candidates"] = []
    hotspot_owners, hotspot_flow_ids = _hotspot_storage_owners(workload, state, diagnosis) if generator_guidance else ({}, ())
    diagnosis["hotspot_layer_ids"] = sorted(hotspot_owners)
    diagnosis["hotspot_flow_ids"] = sorted(hotspot_flow_ids)
    generic, guided = [], []
    stats = {"proposed": {}, "hard_rejected": {}, "deduplicated": 0, "shortlist_dropped": 0,
             "proxy_evaluations": 0, "full_evaluations": 0, "diagnosis": diagnosis,
             "structurally_legal_by_family": {}, "structurally_rejected_by_family": {}}

    def add(family, edits, targets=(), reason="generic exploration", sources=(), *, guide=False):
        row = make_action_plan(state, family, edits, targets=targets, reason=reason, sources=sources)
        (guided if guide else generic).append(row)
        stats["proposed"][family] = stats["proposed"].get(family, 0) + 1
        return row

    if state.version >= 3:
        for layer in workload.layers:
            if layer.weight_shape:
                for flow in ('output_stationary', 'weight_reuse', 'activation_reuse'):
                    if flow != state.mapping.mapping_map[layer.id].dataflow:
                        add('A12', [('A12', {'layer_id': layer.id, 'dataflow': flow})], (layer.id,),
                            'priced loop-order dataflow switch; MAC count preserved, L1/psum traffic recomputed')
    low_actions = legal_actions(workload, state.mapping, package.memory_ids,
                                core_ids=package.mapping_core_ids, max_per_operator=max_per_family, rng=rng)
    for low in low_actions:
        family, params = _translate(state, low)
        if family == "A8":
            field, layer_id = params.pop("_field"), params.pop("_layer")
            layer = workload.layer_map[layer_id]
            if field == "output_dram":
                params["tensor_id"] = layer.output
            elif field == "input_dram":
                external = [tensor for tensor in layer.inputs if tensor in workload.input_ids]
                if not external:
                    continue
                params["tensor_id"] = external[0]
        plan = add(family, [(family, params)], (low.layer_id, low.other_layer_id) if low.other_layer_id else (low.layer_id,))
        if proposal == "composite" and family == "A4":
            receiver = low.other_layer_id or low.layer_id
            if receiver in diagnosis["compute_targets"] and (not low.other_layer_id or low.layer_id in diagnosis["donor_candidates"]):
                realized = apply_action(state.mapping, low, core_ids=package.mapping_core_ids)
                involved = tuple(dict.fromkeys(layer for layer in plan.target_ids if layer))
                add("C1", (*plan.ordered_edits, *_realization_edits(workload, realized, involved)), plan.target_ids,
                    "critical compute stage: test explicit transfer and legal repartition", ("Gemini", "engineering_proposal"), guide=True)
        affected_hotspot_layers = [layer_id for layer_id in plan.target_ids if layer_id in hotspot_owners]
        if proposal == "composite" and family in {"A5", "A6", "A8"} and affected_hotspot_layers:
            layer_id = affected_hotspot_layers[0]
            item = state.mapping.mapping_map[layer_id]
            coupled = []
            if family in {"A5", "A6"}:
                tensor_id = sorted(hotspot_owners[layer_id])[0]
                previous = _storage_selector(workload, state.mapping, layer_id, tensor_id)
                memories = [memory for memory in (*package.memory_ids, "interleave") if memory != previous]
                if memories:
                    coupled = [ActionEdit("A8", {"tensor_id": tensor_id, "configuration_id": memories[0]})]
            elif params["tensor_id"] in hotspot_owners[layer_id] and len(item.chiplets) > 1:
                coupled = [ActionEdit("A5", {"layer_id": layer_id, "cores": list(reversed(item.chiplets))})]
            if coupled:
                add("C5", (*plan.ordered_edits, *coupled), plan.target_ids,
                    "measured competing time-window load: jointly test work placement and DRAM entry redistribution",
                    ("Gemini", "INDM", "engineering_proposal"), guide=True)

    # Independent exploration of adjacent legal scales/shapes always exists.
    architectures = sorted(candidates.values(), key=lambda c: (len(_package(c.hardware).mapping_core_ids), c.id))
    index = next(i for i, candidate in enumerate(architectures) if candidate.id == current.id)
    neighbors = [architectures[i] for i in (index - 1, index + 1) if 0 <= i < len(architectures)]
    if state.version >= 2:
        neighbors = [target for target in architectures if target.id != current.id]
    for target in neighbors:
        try:
            mapping = resized_mapping(workload, state, target)
        except ValueError as error:
            stats["hard_rejected"][str(error)] = stats["hard_rejected"].get(str(error), 0) + 1
            continue
        edit = ActionEdit("A9", {"architecture_candidate_id": target.id, "mapping": mapping.to_dict()})
        add("A9", [edit], (target.id,), "any legal v0.2 architecture jump" if state.version >= 2 else "neighboring legal hardware template")
        if proposal == "composite":
            target_core_count = len(_package(target.hardware).mapping_core_ids)
            if target_core_count == len(package.mapping_core_ids):
                # Same-count mesh/IO shape exploration is A9, never a false
                # expansion/consolidation label in the family statistics.
                continue
            growing = target_core_count > len(package.mapping_core_ids)
            family = "C2" if growing else "C3"
            realized = _realization_edits(workload, mapping, tuple(item.layer_id for item in mapping.layer_mappings), storage=True)
            add(family, [edit, *realized], (target.id,),
                "test expansion with explicit workload allocation" if growing else "test consolidation with complete task/data evacuation",
                ("Gemini", "engineering_proposal"), guide=True)

    for stack in state.mapping.fusion_stacks if fusion_enabled else ():
        ids = stack.layer_ids
        add("A1", [("A1", {"operation": "split", "layer_ids": list(ids)})], ids)
        sink = workload.tensor_map[workload.layer_map[ids[-1]].output]
        overlap_exists = _has_overlap(workload, ids)
        for axis in (0, 1):
            for factor in (.5, 2):
                tile = list(stack.tile_hw)
                tile[axis] = max(1, min(sink.shape[axis + 2], int(tile[axis] * factor)))
                if tuple(tile) == tuple(stack.tile_hw):
                    continue
                edit = ActionEdit("A3", {"layer_ids": list(ids), "tile_hw": tile})
                add("A3", [edit], ids)
                if proposal == "composite" and overlap_exists:
                    cache = ActionEdit("A7", {"layer_ids": list(ids), "cache_h": factor > 1, "cache_v": factor > 1})
                    add("C7" if factor < 1 else "C8", [edit, cache], ids,
                        "test DF memory/recompute tradeoff; physical evaluator decides benefit", ("DeFiNES", "engineering_proposal"), guide=True)
        if overlap_exists:
            for mode, flags in OVERLAP_TO_FLAGS.items():
                if mode != stack.overlap:
                    edit = ActionEdit("A7", {"layer_ids": list(ids), "cache_h": flags[0], "cache_v": flags[1]})
                    add("A7", [edit], ids)
                    if proposal == "composite":
                        tile = [max(1, min(extent, value * 2 if any(flags) else max(1, value // 2)))
                                for value, extent in zip(stack.tile_hw, sink.shape[2:])]
                        if tuple(tile) != tuple(stack.tile_hw):
                            add("C8", [ActionEdit("A3", {"layer_ids": list(ids), "tile_hw": tile}), edit], ids,
                                "jointly test DF tile and overlap reuse against paid cache lifetime/cost",
                                ("DeFiNES", "engineering_proposal"), guide=True)
        elif stack.overlap != "recompute":
            add("A7", [("A7", {"layer_ids": list(ids), "cache_h": False, "cache_v": False})], ids,
                "remove redundant overlap policy from a no-overlap stack")

    fused = {layer for stack in state.mapping.fusion_stacks for layer in stack.layer_ids}
    for group in state.mapping.groups:
        if fusion_enabled and len(group.layer_ids) <= max_fusion_depth and not fused.intersection(group.layer_ids) and legal_stack(workload, group.layer_ids):
            tile = [min(value, extent) for value, extent in zip(tile_sizes[0], workload.tensor_map[workload.layer_map[group.layer_ids[-1]].output].shape[2:])]
            edit = ActionEdit("A1", {"operation": "merge", "layer_ids": list(group.layer_ids), "tile_hw": tile})
            add("A1", [edit], group.layer_ids)
            if proposal == "composite" and state.version == 1:
                overlap_exists = _has_overlap(workload, group.layer_ids)
                macro_tile = [max(1, value // 2) for value in tile]
                if overlap_exists or macro_tile != tile:
                    add("C6", [edit, ActionEdit("A3", {"layer_ids": list(group.layer_ids), "tile_hw": macro_tile}),
                               ActionEdit("A7", {"layer_ids": list(group.layer_ids), "cache_h": overlap_exists, "cache_v": overlap_exists})],
                        group.layer_ids, "closed subgraph: jointly test fusion, sink tile and paid overlap cache",
                        ("DeFiNES", "engineering_proposal"), guide=True)

    # A10 does not imply fusion and never changes requested batch/workload.
    groups = state.mapping.groups
    for index, group in enumerate(groups):
        if len(group.layer_ids) > 1 and not fused.intersection(group.layer_ids):
            cut = len(group.layer_ids) // 2
            changed = (*groups[:index], LayerGroup(group.id + "_l", group.layer_ids[:cut], group.mode),
                       LayerGroup(group.id + "_r", group.layer_ids[cut:], group.mode), *groups[index + 1:])
            edit = ActionEdit("A10", {"groups": _group_dict(changed)})
            add("A10", [edit], group.layer_ids)
            if proposal == "composite":
                regrouped = replace(state.mapping, groups=tuple(changed))
                shadow = replace(state, mapping=regrouped)
                remapped = resized_mapping(workload, shadow, current)
                allocations = []
                for new_group in changed[index:index + 2]:
                    allocations.append(ActionEdit("A4", {"allocations": {
                        layer_id: {"cores": list(remapped.mapping_map[layer_id].chiplets),
                                   "partition": list(remapped.mapping_map[layer_id].partition.as_tuple())}
                        for layer_id in new_group.layer_ids}}))
                add("C9", [edit, *allocations, *_realization_edits(workload, remapped, group.layer_ids)],
                    group.layer_ids, "jointly test pipeline boundary, resource quotas and integer partitions",
                    ("Gemini", "engineering_proposal"), guide=True)
        if index + 1 < len(groups):
            other = groups[index + 1]
            ids = (*group.layer_ids, *other.layer_ids)
            if not fused.intersection(ids):
                occupied = [core for layer in ids for core in state.mapping.mapping_map[layer].chiplets]
                mode = "spatial" if len(occupied) == len(set(occupied)) else "temporal"
                changed = (*groups[:index], LayerGroup(group.id + "_" + other.id, ids, mode), *groups[index + 2:])
                add("A10", [("A10", {"groups": _group_dict(changed)})], ids)
    divisors = [value for value in range(1, workload.batch_size + 1) if workload.batch_size % value == 0]
    for batch in divisors:
        if batch != state.mapping.microbatch_size and all(item.partition.b <= batch for item in state.mapping.layer_mappings):
            add("A10", [("A10", {"groups": _group_dict(groups), "microbatch_size": batch})], (), "fixed-workload pipeline batch unit")

    # C4 changes adjacent partition choices together, holding all other
    # boundaries/quotas fixed; all branch consumers are rebuilt by the oracle.
    if proposal == "composite":
        for consumer in workload.layers:
            for tensor_id in consumer.inputs:
                producer = workload.tensor_map[tensor_id].producer
                if not producer:
                    continue
                options = []
                for layer_id in (producer, consumer.id):
                    item = state.mapping.mapping_map[layer_id]
                    parts = [part for part in partition_options(workload, layer_id, len(item.chiplets), state.mapping.microbatch_size)
                             if part != partition_tuple(item.partition)]
                    options.append(parts[:2])
                if all(options):
                    for a, b in zip(*options):
                        add("C4", [("A2", {"layer_id": producer, "partition": list(a)}),
                                   ("A2", {"layer_id": consumer.id, "partition": list(b)})],
                            (producer, consumer.id), "test coordinated adjacent partitions with fixed external boundaries",
                            ("INDM", "Gemini", "engineering_proposal"), guide=True)

    def clean(rows):
        output, seen = [], set()
        for plan in rows:
            from .search_space import check_transition
            try:
                check_transition(state, apply_transaction(workload, state, plan, candidates), search_space, plan)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                rejected = stats["structurally_rejected_by_family"]
                rejected[plan.family_id] = rejected.get(plan.family_id, 0) + 1
                reason = str(error)
                stats["hard_rejected"][reason] = stats["hard_rejected"].get(reason, 0) + 1
                continue
            verdict = validate_plan(workload, state, plan, candidates)
            if not verdict["valid"]:
                rejected = stats["structurally_rejected_by_family"]
                rejected[plan.family_id] = rejected.get(plan.family_id, 0) + 1
                reason = verdict["reason"]
                stats["hard_rejected"][reason] = stats["hard_rejected"].get(reason, 0) + 1
                continue
            legal = stats["structurally_legal_by_family"]
            legal[plan.family_id] = legal.get(plan.family_id, 0) + 1
            trial = apply_transaction(workload, state, plan, candidates)
            key = trial.fingerprint
            if key in budget_excluded_state_hashes:
                stats['budget_pruned_timeout_states'] = stats.get('budget_pruned_timeout_states', 0) + 1
                continue
            if key in excluded_state_hashes:
                reason = "cached full-system execution/memory validity failure"
                stats["hard_rejected"][reason] = stats["hard_rejected"].get(reason, 0) + 1
                continue
            if verdict["no_op"] or key in seen:
                stats["deduplicated"] += 1
                continue
            seen.add(key)
            output.append(plan)
        return output

    generic, guided = clean(generic), clean(guided)
    # Prefer the explicit macro label when it proposes the same transition;
    # duplicate encodings must not silently multiply sampling probability.
    if proposal == "composite":
        macro_states = {apply_transaction(workload, state, plan, candidates).fingerprint for plan in guided}
        kept = []
        for plan in generic:
            if apply_transaction(workload, state, plan, candidates).fingerprint in macro_states:
                stats["deduplicated"] += 1
            else:
                kept.append(plan)
        generic = kept
    # Keep multiple family/target directions. This is a bounded shortlist,
    # not a safe pruning theorem or a policy-logit prior.
    rng.shuffle(generic)
    rng.shuffle(guided)
    if proposal == "atomic":
        guided = []
    if proposal == "guided_atomic":
        targets = set(diagnosis["compute_targets"])
        generic.sort(key=lambda plan: not bool(targets.intersection(plan.target_ids)))
    # User random quota is a maximum reservation. A legal macro receives a
    # slot even when random_quota >= shortlist; a legal generic receives at
    # least one as well. Equivalent final designs were already deduplicated.
    generic_slots = min(random_quota, shortlist - int(bool(guided)))
    reserved = generic[:max(1, generic_slots)] if generic else []
    selected = list(reserved)
    if state.version >= 2:
        # Reserve direct architecture reachability independently of diagnosis
        # and family caps. The remaining pool retains generic and fusion edits.
        jumps = {}
        for plan in (*guided, *generic):
            target = next((edit.parameters['architecture_candidate_id'] for edit in plan.ordered_edits if edit.action_id == 'A9'), None)
            if target is not None:
                jumps.setdefault(target, plan)
        jump_rows = list(jumps.values())[:max(1, min(shortlist // 4, 12)) if state.version >= 3 else max(0, shortlist - 2)]
        selected = list(dict((plan.id, plan) for plan in (*jump_rows, *reserved[:max(1, shortlist - len(jump_rows) - int(bool(guided)))])).values())
    seen = {plan.id for plan in selected}
    family_counts = Counter()
    for pool in (guided, generic[len(reserved):]):
        for plan in pool:
            if len(selected) >= shortlist:
                break
            if plan.id not in seen and family_counts[plan.family_id] < max_per_family:
                selected.append(plan)
                seen.add(plan.id)
                family_counts[plan.family_id] += 1
    # Fallback is independent of diagnosis and always consumes spare budget.
    if not selected and generic:
        selected = [generic[0]]
    stats["shortlist_dropped"] = len(generic) + len(guided) - len(selected)
    stats["selected_families"] = dict(Counter(plan.family_id for plan in selected))
    stats["capabilities"] = CAPABILITIES
    return tuple(selected), stats
