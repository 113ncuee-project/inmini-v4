"""User PPA priority, physical metrics, fixed references and policy isolation."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace
import unittest

from ppa_preferences import (PPA_WEIGHTS,objective_contract,run_objective_context,resolve_target,
                             scalar_cost,verify_objective_context)
from full_system_dse import objective_cost,physical_violation
from rl_dse import PPA_PREFERENCES,make_objective
from simple_rapidchiplet.hybrid.mapping import initial_mapping
from simple_rapidchiplet.hybrid.single_array_evaluator import SingleArrayEvaluator
from simple_rapidchiplet.hybrid.single_array_hardware import SingleArrayHardware
from simple_rapidchiplet.hybrid.workload import make_toy_workload


class UserPPAPreferenceTests(unittest.TestCase):
    def test_presets_match_V3_and_choose_the_requested_physical_priority(self):
        reference=dict(latency_s=1.,area_mm2=1.,power_w=1.,energy_j=1.)
        points={
            'balanced':dict(latency_s=.9,area_mm2=.9,power_w=.9),
            'low_latency':dict(latency_s=.5,area_mm2=2.,power_w=2.),
            'low_power':dict(latency_s=2.,area_mm2=2.,power_w=.5),
            'small_area':dict(latency_s=2.,area_mm2=.5,power_w=2.),
        }
        for metrics in points.values():metrics.update(feasible=True,energy_j=metrics['latency_s']*metrics['power_w'])
        for name,weights in PPA_WEIGHTS.items():
            self.assertEqual(weights,{k:PPA_PREFERENCES[name][k] for k in weights})
            expected=make_objective(dict(preference=name))
            for metrics in points.values():self.assertAlmostEqual(scalar_cost(metrics,reference,name),expected.cost(metrics,reference),places=12)
            selected=min(points,key=lambda k:scalar_cost(points[k],reference,name))
            self.assertEqual(selected,name)
        self.assertNotEqual(min(points,key=lambda k:scalar_cost(points[k],reference,'energy')),'low_power')

    def test_all_four_legacy_formulas_are_bitwise_preserved(self):
        m=dict(feasible=True,latency_s=.57,area_mm2=2.31,power_w=.82,energy_j=.47)
        r=dict(latency_s=1.79,area_mm2=1.14,power_w=1.27,energy_j=2.17)
        expected=dict(edp=(m['energy_j']/r['energy_j'])*(m['latency_s']/r['latency_s']),
            energy=m['energy_j']/r['energy_j'],latency=m['latency_s']/r['latency_s'],
            balanced=sum(m[k]/r[k] for k in ('latency_s','area_mm2','power_w'))/3)
        for name,cost in expected.items():self.assertEqual(scalar_cost(m,r,name),cost)

    def test_preference_aliases_and_conflicting_targets(self):
        for name in PPA_WEIGHTS:self.assertEqual(resolve_target(preference=name),name)
        self.assertEqual(resolve_target(), 'edp')
        self.assertEqual(resolve_target(default='balanced'),'balanced')
        with self.assertRaises(ValueError):resolve_target('edp','low_power')
        with self.assertRaises(ValueError):resolve_target(preference='unknown')

    def test_checkpoint_cannot_relabel_preference_limits_or_fixed_reference(self):
        args=SimpleNamespace(objective='low_power',max_power=2.,max_area=None,max_latency_ms=None)
        ref=dict(latency_s=1.,area_mm2=2.,power_w=3.,energy_j=3.)
        context=run_objective_context(args,ref);saved=json.loads(json.dumps(context))
        verify_objective_context(saved,context)
        for changed in (SimpleNamespace(**{**vars(args),'objective':'small_area'}),
                        SimpleNamespace(**{**vars(args),'max_power':3.})):
            with self.assertRaises(ValueError):verify_objective_context(saved,run_objective_context(changed,ref))
        with self.assertRaises(ValueError):verify_objective_context(saved,run_objective_context(args,{**ref,'area_mm2':3.}))
        verify_objective_context(run_objective_context(args),context,allow_reference_upgrade=True)
        with self.assertRaises(ValueError):verify_objective_context(run_objective_context(args),context)

    def test_hard_constraints_are_independent_of_the_weighted_score(self):
        m=dict(feasible=True,latency_s=.5,area_mm2=2.,power_w=2.,energy_j=1.)
        ref=dict(latency_s=1.,area_mm2=1.,power_w=1.,energy_j=1.)
        for name in PPA_WEIGHTS:
            args=SimpleNamespace(objective=name,max_area=1.,max_power=None,max_latency_ms=None)
            self.assertEqual(physical_violation(m,args),1.)
            self.assertEqual(objective_cost(m,ref,args),scalar_cost(m,ref,name)+10.)

    def test_actual_physical_PPA_drives_all_preference_costs_and_rewards(self):
        workload=make_toy_workload(batch=1,channels=3,spatial=5)
        metrics=[]
        for rows,cols in ((1,1),(1,2)):
            hardware=SingleArrayHardware(rows,cols)
            plan=initial_mapping(workload,hardware.compute_slots,tuple('memory:'+str(i) for i in range(hardware.io_count)),max_group_layers=3)
            result=SingleArrayEvaluator(workload,hardware)(plan)
            self.assertTrue(result.metrics['feasible'])
            metrics.append(result.metrics)
        ref=metrics[0]
        for name in PPA_WEIGHTS:
            expected=make_objective(dict(preference=name))
            before,after=(scalar_cost(m,ref,name) for m in metrics)
            self.assertAlmostEqual(after,expected.cost(metrics[1],ref),places=12)
            self.assertTrue(math.isfinite(10*math.log(before/after)))
        self.assertNotEqual(scalar_cost(metrics[1],ref,'low_power'),scalar_cost(metrics[1],ref,'small_area'))


if __name__=='__main__':unittest.main()
