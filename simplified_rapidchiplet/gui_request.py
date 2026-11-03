"""Pure UI-intent resolution. v1 requests never receive Auto defaults."""
from copy import deepcopy

# Provisional bounded profiles. Calibration evidence is recorded separately;
# these ceilings are not latency-quality or search-speed promises.
SEARCH_PROFILES = {
    'fast':dict(budget=12,max_wall_time_s=90.,seed_budget=3,seed_architecture_target=2,
                evaluation_timeout_s=45.,report_reserve_s=12.,local_dp_timeout_s=3.),
    'standard':dict(budget=48,max_wall_time_s=480.,seed_budget=8,seed_architecture_target=3,
                    evaluation_timeout_s=400.,report_reserve_s=45.,local_dp_timeout_s=8.),
    'deep':dict(budget=128,max_wall_time_s=1200.,seed_budget=16,seed_architecture_target=5,
                evaluation_timeout_s=360.,report_reserve_s=90.,local_dp_timeout_s=15.),
}
V2_KEYS={'ui_schema_version','search_mode','workload_mode','budget_mode','fusion_mode',
         'microbatch_mode','architecture_mode','seed','budget','batch','models','preference',
         'max_power','max_area','max_latency_ms','strict_power','strict_area','strict_latency'}

def normalize_request(raw,validate_v1):
    if not isinstance(raw,dict):raise ValueError('設定必須是 JSON object')
    if raw.get('ui_schema_version') == 4:
        from gui_request_v4 import normalize_request_v4
        return normalize_request_v4(raw, validate_v1)
    if raw.get('ui_schema_version') == 3:
        from gui_request_v3 import normalize_request_v3
        return normalize_request_v3(raw, validate_v1)
    if 'ui_schema_version' not in raw:
        return validate_v1(raw), {'ui_schema_version':1,'auto_resolved':False}
    if type(raw['ui_schema_version']) is not int or raw['ui_schema_version']!=2:
        raise ValueError('不支援的 ui_schema_version')
    unknown=set(raw)-V2_KEYS
    if unknown:raise ValueError('未知 UI v2 欄位：'+', '.join(sorted(unknown)))
    mode=raw.get('search_mode','standard')
    if mode not in SEARCH_PROFILES:raise ValueError('無效搜尋模式')
    for key,value in [('workload_mode','single'),('fusion_mode','auto'),
                      ('microbatch_mode','auto'),('architecture_mode','auto')]:
        if raw.get(key,value)!=value:raise ValueError('本版僅支援 '+key+'='+value)
    if type(raw.get('batch',1)) is not int or raw.get('batch',1)!=1:
        raise ValueError('single 工作模式必須為 Batch=1')
    budget_mode=raw.get('budget_mode','auto')
    if budget_mode not in ('auto','manual'):raise ValueError('無效 budget_mode')
    profile=deepcopy(SEARCH_PROFILES[mode])
    if budget_mode=='manual':
        budget=raw.get('budget')
        if type(budget) is not int or not 1<=budget<=512:raise ValueError('手動完整評估預算必須為 1–512 整數')
        profile['budget']=budget
    elif 'budget' in raw:
        raise ValueError('Auto 預算不可同時指定數值 budget；請選擇 manual')
    base={k:deepcopy(v) for k,v in raw.items() if k in {
        'models','preference','max_power','max_area','max_latency_ms',
        'strict_power','strict_area','strict_latency','seed'}}
    # Invoke the original validator before adding validated engine-only bounds.
    config=validate_v1({**base,'batch':1,'budget':profile['budget'],'fusion':'dp','fusion_depth':4})
    config.update(profile,engine_profile='gui04_v2',search_mode=mode,budget_mode=budget_mode,
                  auto_profile_version='gui04_r2_eval400',
                  input_size=224,fusion_depth=4,fusion_depth_max=8,adaptive_depth=True,
                  local_fusion_dp=True,horizon=4,checkpoint_every=1,
                  suite_wall_time_s=profile['max_wall_time_s']*len(config['models']),
                  model_shapes={m:[1,8,8,8] if m=='toy' else [1,3,224,224] for m in config['models']},
                  architecture_templates=[f'gemini_c{n}_{r}x{n//r}' for n in (1,2,4,8,16)
                                          for r in (1,2,4,8,16) if n%r==0 and r<=n])
    return config,{'ui_schema_version':2,'auto_resolved':True,'requested':deepcopy(raw),
        'effective_engine_config':deepcopy(config),'resolver_version':'gui04_v2_20261101_r2',
        'profile_status':'bounded heuristic ceilings; quality and wall time require workload-specific measurement',
        'microbatch_candidates':[1],'fusion_depth_is_selected_per_stack':True,
        'capabilities':{'arbitrary_floorplan':False,'arbitrary_routing':False,'C10_local_partition_DP':False}}
