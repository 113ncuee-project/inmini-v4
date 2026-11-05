"""Versioned user PPA preferences; average power and energy stay distinct.

The four presets match V3 rl_dse.PPA_PREFERENCES. Every normalization uses
one fixed measured common reference, never a new reference per architecture.
An experiment trains one cost objective; checkpoints are not relabeled later.
"""
from copy import deepcopy
import math

PPA_WEIGHTS={
    'balanced':dict(latency=1/3,area=1/3,power=1/3),
    'low_latency':dict(latency=.8,area=.1,power=.1),
    'low_power':dict(latency=.1,area=.1,power=.8),
    'small_area':dict(latency=.1,area=.8,power=.1),
}
OBJECTIVES=(*PPA_WEIGHTS,'edp','latency','energy')
LEGACY_OBJECTIVES=('balanced','edp','latency','energy')


def resolve_target(objective=None,preference=None,*,default='edp'):
    if preference is not None and preference not in PPA_WEIGHTS:raise ValueError('Unknown user PPA preference')
    if objective is not None and objective not in OBJECTIVES:raise ValueError('Unknown research objective')
    if objective and preference and objective!=preference:raise ValueError('preference and objective disagree; choose one user target')
    return preference or objective or default


def verify_objective_context(saved,expected,*,allow_reference_upgrade=False):
    prior=dict(saved);current=dict(expected)
    if allow_reference_upgrade and 'fixed_reference' not in prior:current.pop('fixed_reference',None)
    if prior!=current:raise ValueError('Saved PPA preference, limits or reference differ; start a new experiment')


def objective_contract(name):
    if name not in OBJECTIVES:raise ValueError('Unknown PPA preference/objective: '+str(name))
    return dict(version='ppa_preference_fixed_reference_v1',name=name,
        weights=deepcopy(PPA_WEIGHTS.get(name)),
        formula='weighted normalized latency/area/average power' if name in PPA_WEIGHTS else
                'normalized energy times normalized latency' if name=='edp' else 'normalized '+name,
        normalization='fixed common measured reference for every architecture',
        power_definition='modeled average inference power; distinct from energy')


def scalar_cost(metrics,reference,name):
    if name not in OBJECTIVES:raise ValueError('Unknown PPA preference/objective: '+str(name))
    if not metrics.get('feasible'):return None
    if name=='edp':return (metrics['energy_j']/reference['energy_j'])*(metrics['latency_s']/reference['latency_s'])
    if name=='latency':return metrics['latency_s']/reference['latency_s']
    if name=='energy':return metrics['energy_j']/reference['energy_j']
    # Preserve the historical full-engine Balanced operation order exactly.
    if name=='balanced':return sum(metrics[k]/reference[k] for k in ('latency_s','area_mm2','power_w'))/3
    weights=PPA_WEIGHTS[name]
    value=sum(weights[w]*metrics[m]/reference[m] for w,m in
              (('latency','latency_s'),('area','area_mm2'),('power','power_w')))
    if not math.isfinite(value) or value<=0:raise ValueError('PPA cost must be finite and positive')
    return value


def run_objective_context(args,reference=None):
    result=dict(contract=objective_contract(args.objective),
        hard_limits={key:getattr(args,key,None) for key in ('max_area','max_power','max_latency_ms')},
        policy_binding='one frozen preference and limits per run; change preference in a new experiment')
    if reference is not None:
        result['fixed_reference']={key:reference[key] for key in ('latency_s','area_mm2','power_w','energy_j')}
    return result
