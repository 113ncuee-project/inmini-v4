"""Cost-factored Double Q with an invariant, measured after-state cost head.

For the unchanged physical reward r=phi(before)-phi(after), phi=10*log(C),
Q(before,action)=phi(before)+V(after)+H(before,action). V learns -phi(after)
from real completed feasible PPA only. H learns the discounted continuation
and V's residual error through the original Double-Q Bellman target.

Warm initialization only trains V; it cannot invent a future trajectory.
The same after-design has the same V input under any before-state. Thus
changing a paid shared starting point cannot silently reuse an old gain.
"""
import copy
import math
import random

import numpy as np
import torch
from torch import nn


def potential(cost):
    if cost is None or not math.isfinite(float(cost)) or cost <= 0:
        raise ValueError('cost potential requires a completed positive physical cost')
    return 10*math.log(float(cost))


def potentials(cost):
    values=np.asarray(cost)
    return potential(float(values)) if values.ndim==0 else np.asarray([potential(v) for v in values])


class CostValueNetwork(nn.Module):
    def __init__(self,size,after_indices):
        super().__init__()
        self.register_buffer('after_indices',torch.tensor(after_indices,dtype=torch.long))
        self.value=nn.Sequential(nn.Linear(len(after_indices),96),nn.Tanh(),nn.Linear(96,48),nn.Tanh(),nn.Linear(48,1))
        self.future=nn.Sequential(nn.Linear(size,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.future[-1].weight)
        nn.init.zeros_(self.future[-1].bias)

    def forward(self,features):
        v=self.value(features.index_select(1,self.after_indices))[:,0]
        h=self.future(features)[:,0]
        # Full-Q loss trains continuation; the auxiliary physical-cost loss is
        # the only writer to the after-state cost estimator.
        return torch.stack((v.detach()+h,v),dim=1)


class CostValueQ:
    representation='physical_cost_factored_double_q_v1'

    def __init__(self,size,after_indices,seed=7,gamma=.9,learning_rate=.001,**ignored):
        torch.set_num_threads(1);torch.manual_seed(seed)
        self.size,self.after_indices,self.gamma=size,tuple(after_indices),gamma
        self.model=CostValueNetwork(size,self.after_indices)
        self.target=copy.deepcopy(self.model)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        self.rng=random.Random(seed);self.replay=[];self.cost_replay=[]
        self.updates=0;self.losses=[];self.last_gradient_norm=0.

    def _tensor(self,features):
        return torch.as_tensor(np.asarray(features),dtype=torch.float32)

    def predict(self,features,before_cost=1.):
        with torch.no_grad():
            return self.model(self._tensor(features))[:,0].numpy()+potentials(before_cost)

    def predict_reward(self,features,before_cost=1.):
        with torch.no_grad():
            return self.model(self._tensor(features))[:,1].numpy()+potentials(before_cost)

    def probabilities(self,features,epsilon=.15,before_cost=1.):
        values=self.predict(features,before_cost)
        best=np.flatnonzero(values>=values.max()-1e-7)
        probabilities=np.full(len(values),epsilon/len(values))
        probabilities[best]+=(1-epsilon)/len(best)
        return probabilities,values

    def _step(self,loss):
        self.optimizer.zero_grad();loss.backward()
        self.last_gradient_norm=float(nn.utils.clip_grad_norm_(self.model.parameters(),10.))
        self.optimizer.step();self.updates+=1;self.losses.append(float(loss.detach()))
        self.losses=self.losses[-256:]
        if self.updates%64==0:self.target.load_state_dict(self.model.state_dict())

    def pretrain(self,feature,after_cost,updates=24):
        self.cost_replay.append((np.asarray(feature,np.float32),-potential(after_cost)))
        self.cost_replay=self.cost_replay[-4096:]
        for _ in range(updates):
            batch=self.rng.sample(self.cost_replay,min(32,len(self.cost_replay)))
            x=torch.tensor(np.stack([row[0] for row in batch]))
            y=torch.tensor([row[1] for row in batch],dtype=torch.float32)
            self._step(nn.functional.smooth_l1_loss(self.model(x)[:,1],y,beta=1.))

    def observe(self,feature,reward,next_features,terminal=False,updates=24,*,before_cost,next_cost):
        p,pnext=potential(before_cost),potential(next_cost)
        if not math.isclose(float(reward),p-pnext,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('cost-factored target does not equal the measured physical reward')
        self.replay.append((np.asarray(feature,np.float32),float(reward),np.asarray(next_features,np.float32),bool(terminal),p,pnext))
        self.replay=self.replay[-4096:]
        self.cost_replay.append((np.asarray(feature,np.float32),-pnext));self.cost_replay=self.cost_replay[-4096:]
        for _ in range(updates):
            batch=self.rng.sample(self.replay,min(32,len(self.replay)))
            x=torch.tensor(np.stack([row[0] for row in batch]))
            with torch.no_grad():
                targets=[]
                for _,r,successors,done,p0,p1 in batch:
                    future=0.
                    if not done and len(successors):
                        successors=torch.as_tensor(successors)
                        index=int(self.model(successors)[:,0].argmax())
                        future=p1+float(self.target(successors)[index,0])
                    targets.append(r-p0+self.gamma*future)
                target=torch.tensor(targets,dtype=torch.float32)
            predictions=self.model(x)
            qloss=nn.functional.smooth_l1_loss(predictions[:,0],target,beta=1.)
            physical=self.rng.sample(self.cost_replay,min(32,len(self.cost_replay)))
            px=torch.tensor(np.stack([row[0] for row in physical]))
            py=torch.tensor([row[1] for row in physical],dtype=torch.float32)
            vloss=nn.functional.smooth_l1_loss(self.model(px)[:,1],py,beta=1.)
            self._step(qloss+.5*vloss)

    def save(self,path):
        torch.save(dict(representation=self.representation,size=self.size,after_indices=self.after_indices,
            model=self.model.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
            replay=self.replay,cost_replay=self.cost_replay,updates=self.updates,losses=self.losses,
            rng=self.rng.getstate(),gamma=self.gamma,gradient=self.last_gradient_norm),path)

    def load(self,path):
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data.get('representation')!=self.representation or data['size']!=self.size or tuple(data['after_indices'])!=self.after_indices:
            raise ValueError('incompatible cost-value representation')
        self.model.load_state_dict(data['model']);self.target.load_state_dict(data['target'])
        self.optimizer.load_state_dict(data['optimizer']);self.replay=data['replay'];self.cost_replay=data['cost_replay']
        self.updates,self.losses=data['updates'],data['losses'];self.rng.setstate(data['rng'])
        self.gamma,self.last_gradient_norm=data['gamma'],data['gradient']
