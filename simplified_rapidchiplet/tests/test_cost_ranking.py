"""Optional physical auxiliary ordering; no synthetic-search success claim."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from simple_rapidchiplet.hybrid.cost_value_q import CostValueQ


class CostRankingTests(unittest.TestCase):
    def test_nearby_physical_order_has_gradient_and_absolute_anchor_is_retained(self):
        q = CostValueQ(3,(1,2),ranking_weight=1.)
        prediction=torch.tensor([0.,0.],requires_grad=True)
        target=torch.tensor([.05,-.05])
        base=torch.nn.functional.smooth_l1_loss(prediction,target,beta=1.)
        base.backward();gradient=prediction.grad.clone();prediction.grad.zero_()
        full=q._physical_loss(prediction,target);full.backward()
        self.assertGreater(float(full),float(base))
        self.assertGreater(float(prediction.grad.abs().sum()),float(gradient.abs().sum()))
        self.assertLess(float(prediction.grad[0]),0.)
        self.assertEqual(float(q._physical_loss(target,target)),0.)
        q.ranking_weight=0.
        self.assertEqual(float(q._physical_loss(prediction,target)),float(base))

    def test_rank_training_checkpoint_continues_exactly_with_true_double_q_reward(self):
        q = CostValueQ(3,(1,2),ranking_weight=1.)
        f=np.asarray([1.,.3,.7],np.float32)
        q.pretrain(f,.9,3);q.pretrain(f+[0.,.2,-.1],1.1,3)
        task_tmp=Path(__file__).resolve().parents[2]/'tmp'
        task_tmp.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=task_tmp) as name:
            path=Path(name)/'own.pt';q.save(path)
            resumed=CostValueQ(3,(1,2));resumed.load(path)
            self.assertEqual(resumed.ranking_weight,1.)
            successors=np.asarray([f,f+[0.,.1,.1]],np.float32)
            for model in (q,resumed):
                model.observe(f,10*math.log(1/.9),successors,updates=3,before_cost=1.,next_cost=.9)
            for key,value in q.model.state_dict().items():
                np.testing.assert_array_equal(value.numpy(),resumed.model.state_dict()[key].numpy())
            self.assertEqual(q.updates,resumed.updates)
            self.assertFalse(q.replay[-1][3])


if __name__=='__main__':unittest.main()
