"""Auditable transaction constraints, checked before any full oracle."""
SPACES = ('mapping_only', 'mapping_fusion', 'joint')

def check_transition(before, after, space='joint', plan=None):
    if space not in SPACES:
        raise ValueError('unknown search space: ' + space)
    if space != 'joint':
        if plan and any(e.action_id == 'A9' for e in plan.ordered_edits):
            raise ValueError('search_space: architecture edit forbidden (entire transaction rejected)')
        if before.architecture_candidate_id != after.architecture_candidate_id:
            raise ValueError('search_space: complete architecture must remain fixed')
    if space == 'mapping_only' and before.mapping.fusion_stacks != after.mapping.fusion_stacks:
        raise ValueError('search_space: fusion stack/tile/cache must remain fixed')
    return after
