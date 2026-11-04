"""Seed-shift invariance, exact Bellman offsets, and physical-domain checks."""
from dataclasses import replace
from pathlib import Path
import math
import random
import tempfile
import unittest

import numpy as np
import torch

from simple_rapidchiplet.hybrid.cost_value_q import CostValueQ,potential
from simple_rapidchiplet.hybrid.full_space import FullDesignSpace,DesignTrial
from simple_rapidchiplet.hybrid.mapping import initial_mapping,LayerGroup
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.rl_actions import validate_state
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.workload import make_toy_workload


class CostValueTests(unittest.TestCase):
    def setUp(self):
        self.w=make_toy_workload(batch=1,channels=8,spatial=7)
        self.cs=enumerate_single_array_architectures_v3()
        self.c=next(c for c in self.cs if c.id=='single_array_c2_2x1')
        m=initial_mapping(self.w,self.c.hardware.compute_slots,self.c.package.memory_ids,max_group_layers=2)
        self.s=DesignState(self.c.id,m,self.w.fingerprint,self.c.hardware.profile_id,version=3)

    def test_absolute_after_features_invariant_to_before_family_and_progress(self):
        for graph in (False,True):
            sp=FullDesignSpace(self.w,self.cs,boundary_features=graph)
            c=next(c for c in self.cs if c.id=='single_array_c4_2x2')
            changed=sp.architecture_trial(self.s,c).state
            metrics=dict(latency_s=1.,energy_j=2.,power_w=2.,area_mm2=1.)
            other=dict(latency_s=.1,energy_j=.8,power_w=8.,area_mm2=3.)
            a=sp.features(self.s,metrics,metrics,[DesignTrial(changed,'A2',('conv1',))],0.)
            b=sp.features(changed,other,metrics,[DesignTrial(changed,'architecture_mapping')],.9)
            indices=sp.after_design_indices
            self.assertEqual(len(indices),65 if graph else 34)
            self.assertLess(max(indices),a.shape[1])
            np.testing.assert_array_equal(a[0,list(indices)],b[0,list(indices)])
            self.assertFalse(np.array_equal(a,b))

    def test_warm_cost_head_leaves_future_zero_and_changes_gain_exactly(self):
        model=CostValueQ(4,[2,3],seed=7)
        a=np.array([1.,2.,.2,.3],np.float32)
        b=np.array([7.,-5.,.2,.3],np.float32)
        future={k:v.clone() for k,v in model.model.future.state_dict().items()}
        for _ in range(5):model.pretrain(a,.25,updates=24)
        self.assertEqual(len(model.replay),0)
        self.assertEqual(model.updates,120)
        for k,v in model.model.future.state_dict().items():torch.testing.assert_close(v,future[k],rtol=0,atol=0)
        x=model.predict_reward([a],1.)[0];y=model.predict_reward([b],.1)[0]
        self.assertAlmostEqual(float(y-x),potential(.1),places=5)
        self.assertAlmostEqual(float(model.predict([a],1.)[0]),float(x),places=5)

    def test_online_bellman_uses_observed_costs_and_exact_checkpoint_resume(self):
        model=CostValueQ(4,[2,3],seed=7)
        a=np.array([1.,0.,0.,1.],np.float32);b=np.array([0.,1.,1.,0.],np.float32)
        for _ in range(3):
            model.pretrain(a,.4,updates=16);model.pretrain(b,.2,updates=16)
        for _ in range(3):
            model.observe(a,10*math.log(.8/.4),[a,b],False,16,before_cost=.8,next_cost=.4)
            model.observe(b,10*math.log(.4/.2),[],True,16,before_cost=.4,next_cost=.2)
        self.assertTrue(any(not r[3] for r in model.replay))
        self.assertTrue(any(r[3] for r in model.replay))
        self.assertGreater(model.last_gradient_norm,0)
        self.assertGreater(float(model.model.future[-1].weight.abs().sum()),0)
        with self.assertRaises(ValueError):model.observe(a,99,[],True,1,before_cost=.8,next_cost=.4)
        tmp=Path(__file__).resolve().parents[2]/'tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as folder:
            path=Path(folder)/'q.pt';model.save(path)
            restored=CostValueQ(4,[2,3]);restored.load(path)
            np.testing.assert_array_equal(model.predict([a,b],.5),restored.predict([a,b],.5))
            for q in (model,restored):q.observe(a,10*math.log(.5/.2),[a,b],False,8,before_cost=.5,next_cost=.2)
            np.testing.assert_array_equal(model.predict([a,b],.5),restored.predict([a,b],.5))

    def test_proportional_architecture_jump_keeps_idle_and_real_target_resources(self):
        sp=FullDesignSpace(self.w,self.cs)
        # A whole Temporal group can share a strict subset of physical PEs.
        m=sp.bank(self.s).build((4,'balanced','spread','interleave','auto')).state.mapping
        m=replace(m,groups=(LayerGroup('serial',tuple(self.w.layer_map),'temporal'),))
        source=replace(self.s,mapping=m)
        target=next(c for c in self.cs if c.id=='single_array_c4_2x2')
        t=sp.architecture_trial(source,target,legacy=True)
        self.assertTrue(validate_state(self.w,t.state,self.cs))
        self.assertEqual(t.action['allocation_mode'],'proportional_spread')
        self.assertTrue(all(len(item.chiplets)<=8 for item in t.state.mapping.layer_mappings))
        self.assertEqual(len(t.state.mapping.memory_ids),4)
        self.assertTrue(all(len(item.chiplets)<len(target.hardware.compute_slots) for item in t.state.mapping.layer_mappings))
        for c in self.cs:
            self.assertTrue(validate_state(self.w,sp.architecture_trial(source,c,legacy=True).state,self.cs))

    def test_zero_discount_has_no_successor_value_leakage(self):
        a=np.array([1.,0.,.2,.4],np.float32)
        left=CostValueQ(4,[2,3],seed=7,gamma=0.)
        right=CostValueQ(4,[2,3],seed=7,gamma=0.)
        reward=10*math.log(.5/.2)
        left.observe(a,reward,[a,a*4],False,8,before_cost=.5,next_cost=.2)
        right.observe(a,reward,[a*-.5],False,8,before_cost=.5,next_cost=.2)
        np.testing.assert_array_equal(left.predict([a],.5),right.predict([a],.5))


if __name__=='__main__':unittest.main()
