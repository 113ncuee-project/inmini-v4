"""Schema3 target-local evidence, never a predicted PPA or reward bonus."""
import numpy as np
from .rl_schema import ActionEdit, make_action_plan


def memory_actions(workload, state, report):
    """Respond to measured shared-pool pressure; validity/PPA remain separate."""
    pools = list(report.get('diagnostics', {}).get('memory', ()))
    result = []
    for stack in state.mapping.fusion_stacks:
        owners = {owner for lid in stack.layer_ids for slot in state.mapping.mapping_map[lid].chiplets
                  for owner in (slot, slot.split('/')[0])}
        relevant = [row for row in pools if row['pool_id'].rsplit(':', 1)[0] in owners]
        if not relevant:
            continue
        pressure = max(row['peak_live_bytes'] / max(row['capacity_bytes'], 1) for row in relevant)
        if pressure < .9:
            from .rl_actions import _has_overlap
            if pressure < .65 and report.get('recompute_macs', 0) > 0 and _has_overlap(workload, stack.layer_ids):
                sink = workload.tensor_map[workload.layer_map[stack.layer_ids[-1]].output]
                tile = tuple(min(extent, value * 2) for extent, value in zip(sink.shape[2:], stack.tile_hw))
                result.append(make_action_plan(state, 'C8', [
                    ActionEdit('A3', {'layer_ids': list(stack.layer_ids), 'tile_hw': list(tile)}),
                    ActionEdit('A7', {'layer_ids': list(stack.layer_ids), 'cache_h': True, 'cache_v': True})],
                    targets=stack.layer_ids, reason='measured SRAM headroom and recompute; test larger tile/cache with full PPA',
                    sources=('measured_memory_headroom',)))
            continue
        tile = tuple(max(1, value // 2) for value in stack.tile_hw)
        if tile == stack.tile_hw and stack.overlap == 'recompute':
            continue
        edits = [ActionEdit('A3', {'layer_ids': list(stack.layer_ids), 'tile_hw': list(tile)}),
                 ActionEdit('A7', {'layer_ids': list(stack.layer_ids), 'cache_h': False, 'cache_v': False})]
        pressure = max(row['peak_live_bytes'] / row['capacity_bytes'] for row in relevant)
        result.append(make_action_plan(state, 'C7', edits, targets=stack.layer_ids,
            reason=f'measured SRAM pool pressure {pressure:.6f}; shrink tile and release overlap cache; PPA pending',
            sources=('measured_memory_pressure', *(row['pool_id'] for row in relevant))))
    return result


def target_features(state, plan, report, workload):
    latency = max(report.get('latency_s', 0.), 1e-30)
    diagnostics = report.get('diagnostics', {})
    targets = set(plan.target_ids).intersection(workload.layer_map)
    for edit in plan.ordered_edits:
        if edit.action_id == 'A9':
            targets.update(workload.layer_map)
        targets.update(edit.parameters.get('layer_ids', ()))
        if edit.parameters.get('layer_id'):
            targets.add(edit.parameters['layer_id'])
        targets.update(edit.parameters.get('allocations', {}))
    rows = [row for row in diagnostics.get('layers', ()) if row['layer_id'] in targets]
    maximum = lambda key: max((row.get(key, 0.) or 0. for row in rows), default=0.)
    total = lambda key: sum(row.get(key, 0.) or 0. for row in rows)
    slots = {slot for lid in targets if lid in state.mapping.mapping_map for slot in state.mapping.mapping_map[lid].chiplets}
    pes, dies = max(len(slots), 1), max(len({slot.split('/')[0] for slot in slots}), 1)
    transfer_wait = max((row.get('max_transfer_queue_wait_s', (row.get('transfer_queue_wait_s', 0.) or 0.) / max(row.get('transfer_count', 1), 1)) for row in rows), default=0.)
    job_wait = max((row.get('max_job_queue_wait_s', (row.get('resource_wait_s', 0.) or 0.) / max(len(row.get('pe_allocations', ())), 1)) for row in rows), default=0.)
    return np.clip(np.asarray([
        maximum('criticality'), total('mac_service_s') / (latency * pes),
        total('l1_service_s') / (latency * pes), total('vector_busy_s') / (latency * dies),
        transfer_wait / latency, job_wait / latency,
        maximum('memory_pressure'), min((row.get('slack_s') or 0. for row in rows), default=0.) / latency,
        sum(workload.layer_map[lid].macs_per_sample for lid in targets if lid in workload.layer_map) / max(workload.total_macs, 1),
        len(targets) / max(len(workload.layers), 1),
        maximum('pe_utilization'), float(not rows),
    ], dtype=np.float64), -8, 8)
