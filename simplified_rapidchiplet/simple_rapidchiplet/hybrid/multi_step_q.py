"""Action-conditioned Double Q learning with a target network and replay.

Observed costs provide both a dense physical improvement reward and a
supervised immediate-reward auxiliary head. No proxy labels train either head.
Future values use the same legal successor candidate set available to policy.
"""
import copy
import random

import numpy as np
import torch
from torch import nn


class MultiStepQ:
    def __init__(self, size, seed=7, gamma=.9, learning_rate=.001):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.model = nn.Sequential(nn.Linear(size, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 2))
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.replay = []
        self.updates = 0
        self.losses = []
        self.last_gradient_norm = 0.

    def predict(self, features):
        with torch.no_grad():
            return self.model(torch.as_tensor(np.asarray(features), dtype=torch.float32))[:, 0].numpy()

    def predict_reward(self, features):
        with torch.no_grad():
            return self.model(torch.as_tensor(np.asarray(features), dtype=torch.float32))[:, 1].numpy()

    def probabilities(self, features, epsilon=.15):
        values = self.predict(features)
        # Epsilon greedy exploits the learned ordering even when small
        # physical rewards would leave a unit-temperature softmax uniform.
        best = np.flatnonzero(values >= values.max() - 1e-7)
        p = np.full(len(values), epsilon / len(values))
        p[best] += (1 - epsilon) / len(best)
        return p, values

    def observe(self, feature, reward, next_features, terminal=False, updates=24):
        self.replay.append((np.asarray(feature, np.float32), float(reward),
                            np.asarray(next_features, np.float32), bool(terminal)))
        self.replay = self.replay[-4096:]
        for _ in range(updates):
            batch = self.rng.sample(self.replay, min(32, len(self.replay)))
            x = torch.tensor(np.stack([r[0] for r in batch]))
            rewards = torch.tensor([r[1] for r in batch], dtype=torch.float32)
            with torch.no_grad():
                future = []
                for _, _, successors, done in batch:
                    if done or not len(successors):
                        future.append(0.)
                    else:
                        f = torch.as_tensor(successors)
                        index = int(self.model(f)[:, 0].argmax())
                        future.append(float(self.target(f)[index, 0]))
                expected = rewards + self.gamma * torch.tensor(future)
            predictions = self.model(x)
            loss = nn.functional.smooth_l1_loss(predictions[:, 0], expected, beta=1.)
            loss = loss + .5 * nn.functional.smooth_l1_loss(predictions[:, 1], rewards, beta=1.)
            self.optimizer.zero_grad()
            loss.backward()
            self.last_gradient_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), 10.))
            self.optimizer.step()
            self.updates += 1
            self.losses.append(float(loss.detach()))
            if self.updates % 64 == 0:
                self.target.load_state_dict(self.model.state_dict())
        self.losses = self.losses[-256:]

    def save(self, path):
        torch.save(dict(model=self.model.state_dict(), target=self.target.state_dict(), optimizer=self.optimizer.state_dict(),
                        replay=self.replay, updates=self.updates, losses=self.losses, rng=self.rng.getstate(),
                        gamma=self.gamma, gradient=self.last_gradient_norm), path)

    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(data['model'])
        self.target.load_state_dict(data['target'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.replay, self.updates, self.losses = data['replay'], data['updates'], data['losses']
        self.rng.setstate(data['rng'])
        self.gamma, self.last_gradient_norm = data['gamma'], data['gradient']
