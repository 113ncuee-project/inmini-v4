"""Frozen physical-cost prior, measured residual adaptation and Double Q.

Only paid pre-search observations train the invariant global cost prior. It is
then frozen, avoiding forgetting of architectures while learning a small local
neighborhood. Own completed PPA observations fit a regularized correction;
basis/ridge strength are selected by exact TRAINING-ONLY leave-one-out error.
The continuation head still uses actual nonterminal Double-Q Bellman targets.
The correction is an auxiliary physical-cost estimator, not a new PPA sample.
"""
import copy
import math

import numpy as np
import torch
from torch import nn

from .cost_value_q import CostValueQ, potential, potentials


class PhysicalResidual:
    def __init__(self, size, after_indices, local_indices, *, enabled=True, capacity=128, basis='auto'):
        if capacity < 8 or capacity > 512 or basis not in ('auto','full','local','after'):
            raise ValueError('Residual capacity must be8..512 and basis auto/full/local/after')
        self.size = size
        self.indices = dict(full=tuple(range(size)),local=tuple(local_indices),after=tuple(after_indices))
        if any(not v or min(v)<0 or max(v)>=size for v in self.indices.values()):
            raise ValueError('Residual feature basis outside declared inputs')
        self.enabled, self.capacity, self.basis = enabled, capacity, basis
        self.observations = []
        self.model = None
        self.fit_count = 0
        self.linear_candidates_fitted = 0
        self.selection = None

    def predict(self, features):
        x = np.asarray(features,dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.size:
            raise ValueError('Wrong residual feature shape')
        if self.model is None:
            return np.zeros(len(x),dtype=np.float64)
        indices, scale, coefficient = self.model
        return (x[:,indices]/scale)@coefficient

    def observe(self, feature, correction):
        if not math.isfinite(correction):
            raise ValueError('Residual target must be an observed finite physical cost correction')
        self.observations.append((np.asarray(feature,dtype=np.float32),float(correction)))
        self.observations = self.observations[-4096:]
        if not self.enabled or len(self.observations) < 8:
            return
        count = len(self.observations)
        if count <= self.capacity:
            selected = self.observations
        else:
            # Deterministic temporal coverage, including every newest32. No
            # RNG perturbation of the physical proposal/acceptance process.
            recent = min(32,self.capacity//2)
            indices = np.linspace(0,count-recent-1,self.capacity-recent,dtype=int)
            selected = [self.observations[i] for i in indices]+self.observations[-recent:]
        x = np.asarray([r[0] for r in selected],dtype=np.float64)
        y = np.asarray([r[1] for r in selected],dtype=np.float64)
        candidates = []
        for name in (self.indices if self.basis=='auto' else (self.basis,)):
            indices = self.indices[name]
            values = x[:,indices]
            scale = np.maximum(np.sqrt(np.square(values).mean(axis=0)),.1)
            values = values/scale
            kernel = values@values.T
            eigenvalues,eigenvectors = np.linalg.eigh(kernel)
            eigenvalues = np.maximum(0.,eigenvalues)
            rotated = eigenvectors.T@y
            for lam in (.01,.1,1.,10.,100.,1000.):
                inverse = 1/(eigenvalues+lam)
                alpha = eigenvectors@(rotated*inverse)
                diagonal = np.square(eigenvectors)@inverse
                loo_error = alpha/diagonal
                error = float(np.square(loo_error).mean())
                coefficient = values.T@alpha
                candidates.append((error,name,-lam,(indices,scale,coefficient)))
        error,name,negative_lambda,model = min(candidates,key=lambda r:r[:3])
        self.model = model
        self.fit_count += 1
        self.linear_candidates_fitted += len(candidates)
        self.selection = dict(basis=name,ridge=-negative_lambda,training_LOO_MSE=error,
            fitted_observations=len(selected),available_own_physical_observations=count,
            candidates=[dict(LOO_MSE=r[0],basis=r[1],ridge=-r[2]) for r in candidates])

    def diagnostics(self):
        return dict(enabled=self.enabled,observations=len(self.observations),fits=self.fit_count,
            linear_candidates_fitted=self.linear_candidates_fitted,capacity=self.capacity,
            selected_basis=self.selection['basis'] if self.selection else None,
            selected_ridge=self.selection['ridge'] if self.selection else None,
            training_LOO_MSE=self.selection['training_LOO_MSE'] if self.selection else None,
            scope='own measured labels only; training-only LOO, no control labels or added PPA')


class ResidualCostQ(CostValueQ):
    representation = 'frozen_prior_residual_normalized_continuation_q_v2'

    def __init__(self,size,after_indices,local_indices,seed=7,gamma=.9,learning_rate=.001,
                 *,correction_enabled=True,capacity=128,basis='auto'):
        super().__init__(size,after_indices,seed=seed,gamma=gamma,learning_rate=learning_rate)
        # A Tanh continuation saturated to almost the same value for every
        # measured action. Layer normalization keeps hidden scales controlled;
        # SiLU retains a usable gradient beyond Tanh's narrow active range.
        # The measured reward and Q values stay in the original10log units.
        self.model.future=nn.Sequential(nn.Linear(size,128),nn.LayerNorm(128),nn.SiLU(),
            nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.model.future[-1].weight)
        nn.init.zeros_(self.model.future[-1].bias)
        self.target=copy.deepcopy(self.model)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        self.continuation_scale=10.
        self.local_indices = tuple(local_indices)
        self.residual = PhysicalResidual(size,after_indices,local_indices,enabled=correction_enabled,
                                         capacity=capacity,basis=basis)
        self.target_residual = copy.deepcopy(self.residual)
        self.prior_frozen = False
        self.prior_warm_updates = None

    def predict(self,features,before_cost=1.):
        return super().predict(features,before_cost)+self.residual.predict(features)

    def predict_reward(self,features,before_cost=1.):
        return super().predict_reward(features,before_cost)+self.residual.predict(features)

    def pretrain(self,feature,after_cost,updates=24):
        if self.prior_frozen:
            raise ValueError('Global prior is frozen after the first own physical transition')
        self.cost_replay.append((np.asarray(feature,np.float32),-potential(after_cost)))
        self.cost_replay=self.cost_replay[-4096:]
        for _ in range(updates):
            batch=self.rng.sample(self.cost_replay,min(32,len(self.cost_replay)))
            x=self._tensor(np.stack([row[0] for row in batch]))
            y=torch.tensor([row[1] for row in batch],dtype=torch.float32)
            # Do not route this loss through stack(V+H,V). A zero H gradient
            # still advances Adam's step counter during shared warmup, making
            # its first real TD update use an artificially aged optimizer.
            prediction=self.model.value(x.index_select(1,self.model.after_indices))[:,0]
            self._step(self._physical_loss(prediction,y))

    def _step(self,loss):
        super()._step(loss)
        if self.updates%64 == 0:
            self.target_residual = copy.deepcopy(self.residual)

    def observe(self,feature,reward,next_features,terminal=False,updates=24,*,before_cost,next_cost):
        p,pnext = potential(before_cost),potential(next_cost)
        if not math.isclose(float(reward),p-pnext,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('Residual Q requires the exact completed physical reward')
        if not self.prior_frozen:
            self.prior_frozen = True
            self.prior_warm_updates = self.updates
            # A detached branch can still receive a zero gradient through
            # stack(), letting Adam's old warmup momentum change its weights.
            # Explicitly disable gradients and clear them before freezing.
            for parameter in self.model.value.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            self.target.load_state_dict(self.model.state_dict())
        feature = np.asarray(feature,dtype=np.float32)
        successors = np.asarray(next_features,dtype=np.float32)
        self.replay.append((feature,float(reward),successors,bool(terminal),p,pnext))
        self.replay = self.replay[-4096:]
        self.cost_replay.append((feature,-pnext))
        self.cost_replay = self.cost_replay[-4096:]
        with torch.no_grad():
            prior = float(self.model(self._tensor([feature]))[0,1])
        self.residual.observe(feature,-pnext-prior)
        for _ in range(updates):
            batch = self.rng.sample(self.replay,min(32,len(self.replay)))
            x = np.stack([r[0] for r in batch])
            with torch.no_grad():
                targets = []
                for _,r,nxt,done,p0,p1 in batch:
                    future = 0.
                    if not done and len(nxt):
                        tensor = self._tensor(nxt)
                        online = self.model(tensor)[:,0].numpy()+self.residual.predict(nxt)
                        selected = int(online.argmax())
                        future = p1+float(self.target(tensor)[selected,0])+float(self.target_residual.predict(nxt)[selected])
                    targets.append(r-p0+self.gamma*future)
                target = torch.tensor(targets,dtype=torch.float32)
            # V is detached by CostValueNetwork; the measured linear residual
            # is constant for SGD. Only H receives a Bellman gradient here.
            prediction = self.model(self._tensor(x))[:,0]+torch.as_tensor(self.residual.predict(x),dtype=torch.float32)
            # Dimensionless log-cost loss; this scales the optimization only,
            # never the physical reward, discount or reported full-Q values.
            loss = nn.functional.smooth_l1_loss(prediction/self.continuation_scale,
                                                target/self.continuation_scale,beta=1.)
            self._step(loss)

    def diagnostics(self):
        return dict(prior_frozen=self.prior_frozen,prior_warm_SGD=self.prior_warm_updates,
                    continuation='LayerNorm/SiLU; cold Adam during V-only warmup; TD loss in log-cost units',
                    residual=self.residual.diagnostics())

    def save(self,path):
        # One durable write; no duplicate checkpoint serialization per step.
        torch.save(dict(representation=self.representation,size=self.size,after_indices=self.after_indices,
            model=self.model.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
            replay=self.replay,cost_replay=self.cost_replay,updates=self.updates,losses=self.losses,
            rng=self.rng.getstate(),gamma=self.gamma,gradient=self.last_gradient_norm,ranking_weight=self.ranking_weight,
            local_indices=self.local_indices,residual=self.residual.__dict__,
            target_residual=self.target_residual.__dict__,prior_frozen=self.prior_frozen,
            prior_warm_updates=self.prior_warm_updates),path)

    def load(self,path):
        data = torch.load(path,map_location='cpu',weights_only=False)
        if tuple(data.get('local_indices',())) != self.local_indices:
            raise ValueError('Residual local feature basis differs')
        super().load(path)
        self.residual.__dict__ = data['residual']
        self.target_residual.__dict__ = data['target_residual']
        self.prior_frozen,self.prior_warm_updates = data['prior_frozen'],data['prior_warm_updates']
        if self.prior_frozen:
            for parameter in self.model.value.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
