"""Measured-label residual fitting, frozen prior and real Bellman mechanics."""
from pathlib import Path
import math
import tempfile
import unittest

import numpy as np
import torch

from simple_rapidchiplet.hybrid.cost_value_q import potential
from simple_rapidchiplet.hybrid.residual_cost_q import PhysicalResidual, ResidualCostQ
from simple_rapidchiplet.hybrid.full_space import FullDesignSpace, DesignTrial
from simple_rapidchiplet.hybrid.mapping import initial_mapping
from simple_rapidchiplet.hybrid.rl_schema import DesignState
from simple_rapidchiplet.hybrid.single_array_hardware import enumerate_single_array_architectures_v3
from simple_rapidchiplet.hybrid.workload import make_toy_workload


class ResidualCostTests(unittest.TestCase):
    def test_training_only_LOO_equals_actual_leave_one_out(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(12,4)).astype(np.float32)
        y = x[:,0]*.7-x[:,2]*.2+rng.normal(size=12)*.1
        residual = PhysicalResidual(4,[2,3],[0,1],basis='full')
        for feature,target in zip(x,y):
            residual.observe(feature,target)
        scale = np.maximum(np.sqrt(np.square(x.astype(float)).mean(axis=0)),.1)
        kernel = (x/scale)@(x/scale).T
        for row in residual.selection['candidates']:
            errors = []
            for i in range(len(x)):
                indices = [j for j in range(len(x)) if j != i]
                alpha = np.linalg.solve(kernel[np.ix_(indices,indices)]+row['ridge']*np.eye(len(indices)),y[indices])
                errors.append((y[i]-kernel[i,indices]@alpha)**2)
            self.assertAlmostEqual(row['LOO_MSE'],float(np.mean(errors)),places=7)

    def test_bounded_deterministic_history_and_disabled_ablation(self):
        enabled = PhysicalResidual(4,[2,3],[0,1],capacity=8)
        disabled = PhysicalResidual(4,[2,3],[0,1],capacity=8,enabled=False)
        for i in range(20):
            feature = np.asarray([1.,i/20,math.sin(i),math.cos(i)],np.float32)
            for model in (enabled,disabled):
                model.observe(feature,float(feature[1]-.2*feature[2]))
        self.assertEqual(enabled.selection['fitted_observations'],8)
        self.assertEqual(enabled.selection['available_own_physical_observations'],20)
        self.assertGreater(enabled.fit_count,0)
        self.assertEqual(disabled.fit_count,0)
        np.testing.assert_array_equal(disabled.predict([[1,2,3,4]]),[0.])

    def model(self,seed=7,gamma=.9):
        q = ResidualCostQ(4,[2,3],[0,1,2],seed=seed,gamma=gamma)
        for i in range(4):
            q.pretrain(np.asarray([1.,i/4,.2,.3],np.float32),.2+i/20,updates=8)
        return q

    def transition(self,q,i,updates=4):
        x = np.asarray([1.,i/20,math.sin(i),math.cos(i)],np.float32)
        c0,c1 = .5+i/200,.2+(i%5)/100
        q.observe(x,10*math.log(c0/c1),[x,x*.9],bool(i%4==3),updates,
                  before_cost=c0,next_cost=c1)

    def test_prior_stays_bitwise_frozen_while_residual_and_continuation_learn(self):
        q = self.model()
        prior = {k:v.clone() for k,v in q.model.value.state_dict().items()}
        for i in range(12):
            self.transition(q,i)
        self.assertTrue(q.prior_frozen)
        self.assertGreater(q.residual.fit_count,0)
        self.assertTrue(any(not r[3] for r in q.replay))
        self.assertGreater(float(q.model.future[-1].weight.abs().sum()),0.)
        for key,value in prior.items():
            torch.testing.assert_close(value,q.model.value.state_dict()[key],rtol=0,atol=0)
        with self.assertRaises(ValueError):
            q.observe(np.ones(4),99,[],True,1,before_cost=.5,next_cost=.2)
        with self.assertRaises(ValueError):
            q.pretrain(np.ones(4),.2,1)

    def test_checkpoint_restores_residual_target_optimizer_rng_and_future_updates_exactly(self):
        q = self.model()
        for i in range(12):
            self.transition(q,i)
        directory = Path(__file__).resolve().parents[2]/'tmp'
        directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as folder:
            path = Path(folder)/'q.pt'
            q.save(path)
            restored = ResidualCostQ(4,[2,3],[0,1,2])
            restored.load(path)
            x = np.asarray([[1.,.1,.5,.3],[1.,.3,.2,.8]],np.float32)
            np.testing.assert_array_equal(q.predict(x,.5),restored.predict(x,.5))
            self.assertEqual(q.residual.selection,restored.residual.selection)
            for i in range(12,15):
                for model in (q,restored):
                    self.transition(model,i,updates=8)
            np.testing.assert_array_equal(q.predict(x,.5),restored.predict(x,.5))
            self.assertEqual(q.updates,restored.updates)

    def test_zero_discount_cannot_leak_unobserved_successor_values(self):
        left,right = self.model(gamma=0.),self.model(gamma=0.)
        x = np.asarray([1.,.2,.4,.3],np.float32)
        for q,next_features in ((left,[x,x*4]),(right,[x*-.5])):
            q.observe(x,10*math.log(.5/.2),next_features,False,8,before_cost=.5,next_cost=.2)
        np.testing.assert_array_equal(left.predict([x],.5),right.predict([x],.5))

    def test_nonterminal_Bellman_value_changes_the_actual_gradient(self):
        current = np.asarray([0.,1.,.2,.3],np.float32)
        successor = np.asarray([2.,1.,.2,.3],np.float32)
        zero = ResidualCostQ(4,[2,3],[0,1,2],gamma=0.)
        future = ResidualCostQ(4,[2,3],[0,1,2],gamma=.9)
        for q in (zero,future):
            with torch.no_grad():
                for parameter in q.model.parameters():
                    parameter.zero_()
                q.model.future[0].weight[0,0] = 1.
                q.model.future[1].weight.fill_(1.)
                q.model.future[4].weight.fill_(1.)
                q.model.future[3].weight[0,0] = 1.
                q.model.future[6].weight[0,0] = 2.
            # Next cost1 has exact zero potential: the immediate residual
            # target is0. Only the successor's positive H can cause learning.
            q.observe(current,potential(.5),[successor],False,1,before_cost=.5,next_cost=1.)
        self.assertEqual(zero.last_gradient_norm,0.)
        self.assertGreater(future.last_gradient_norm,0.)
        self.assertGreater(float(future.predict([current],.5)[0]),float(zero.predict([current],.5)[0]))

    def test_local_basis_respects_optional_feature_layouts(self):
        workload = make_toy_workload(batch=1,channels=8,spatial=7)
        candidates = enumerate_single_array_architectures_v3()
        candidate = next(c for c in candidates if c.id=='single_array_c2_2x1')
        plan = initial_mapping(workload,candidate.hardware.compute_slots,candidate.package.memory_ids,max_group_layers=2)
        state = DesignState(candidate.id,plan,workload.fingerprint,candidate.hardware.profile_id,version=3)
        metrics = dict(latency_s=1.,energy_j=2.,power_w=2.,area_mm2=1.)
        for boundary in (False,True):
            for service in (False,True):
                space = FullDesignSpace(workload,candidates,boundary_features=boundary,service_features=service)
                x = space.features(state,metrics,metrics,[DesignTrial(state,'A2')])
                self.assertLess(max(space.residual_local_indices),x.shape[1])
                PhysicalResidual(x.shape[1],space.after_design_indices,space.residual_local_indices)

    def test_shared_prior_training_does_not_age_the_untrained_TD_optimizer(self):
        q=self.model()
        self.assertTrue(all(parameter not in q.optimizer.state for parameter in q.model.future.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in q.model.future.parameters()))
        self.assertTrue(all(parameter in q.optimizer.state for parameter in q.model.value.parameters()))
        self.transition(q,0,1)
        self.assertTrue(all(int(q.optimizer.state[parameter]['step'])==1 for parameter in q.model.future.parameters()))

    def test_TD_retains_distinct_action_values_after_many_updates(self):
        q=ResidualCostQ(4,[2,3],[0,1,2],gamma=0.,correction_enabled=False)
        with torch.no_grad():
            for parameter in q.model.value.parameters():parameter.zero_()
        points=[np.asarray([1.,-1.,.2,.3],np.float32),np.asarray([1.,1.,.2,.3],np.float32)]
        costs=[1.,4.]
        for i in range(32):
            j=i%2
            q.observe(points[j],-potential(costs[j]),[],True,24,before_cost=1.,next_cost=costs[j])
        with torch.no_grad():h=q.model.future(torch.as_tensor(np.stack(points)))[:,0].numpy()
        self.assertGreater(float(np.std(h)),2.)
        self.assertLess(float(np.abs(h-np.asarray([0.,-potential(4.)])).mean()),1.)


if __name__ == '__main__':
    unittest.main()
