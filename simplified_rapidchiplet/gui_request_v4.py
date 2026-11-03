"""UI schema4: 50 layouts, independent seed/search budgets, local bottleneck actor."""
from gui_request_v3 import normalize_request_v3

SEARCH_PROFILES_V4 = {
 'fast': {'budget':32, 'max_wall_time_s':600., 'seed_wall_time_s':600., 'evaluation_timeout_s':300.},
 'standard': {'budget':128, 'max_wall_time_s':7200., 'seed_wall_time_s':7200., 'evaluation_timeout_s':1800.},
 'deep': {'budget':256, 'max_wall_time_s':14400., 'seed_wall_time_s':14400., 'evaluation_timeout_s':3600.},
}

def normalize_request_v4(raw, validate_v1):
 if raw.get('ui_schema_version') != 4 or type(raw.get('ui_schema_version')) is not int:
  raise ValueError('schema4 required')
 config, provenance = normalize_request_v3({**raw, 'ui_schema_version':3}, validate_v1)
 budget = config['budget']
 config.update(SEARCH_PROFILES_V4[config['search_mode']])
 if raw.get('budget_mode') == 'manual': config['budget']=budget
 config.update(engine_profile='gui04_v4',hardware_profile='chiplet_single_pe_array_v0_3',ui_schema_version=4,
  seed_queries=16,seed_dp_seconds=120.,architecture_templates=[f'single_array_c{n}_{r}x{n//r}'
    for n in range(1,17) for r in range(1,n+1) if n%r==0],
  suite_wall_time_s=(config['seed_wall_time_s']+config['max_wall_time_s'])*len(config['models']),
  budget_semantics='separate seed and learning phases; common seed cost remains charged',
  seed_method='capacity_screened_segment_beam_v3',candidate_feature_schema=3)
 provenance.update(ui_schema_version=4,effective_engine_config=config,resolver_version='gui04_v4_single_array_r1')
 return config,provenance
