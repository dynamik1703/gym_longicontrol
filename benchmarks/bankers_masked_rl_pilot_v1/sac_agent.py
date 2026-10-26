"""Native discrete adaptation of corrected Hybrid constrained SAC."""

import copy

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .masked_distribution import distribution, sample

OBS_DIM = 10
GAMMA_NATIVE = 0.99 ** 0.1


def network():
    return nn.Sequential(
        nn.Linear(OBS_DIM, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 21)
    )


def targets(reward, costs, discount, probability, logp, reward_q, cost_q, alpha):
    yr = reward + discount * (
        probability * (reward_q - alpha * logp)
    ).sum(-1, keepdim=True)
    yc = costs + discount * (
        probability.unsqueeze(1) * cost_q
    ).sum(-1)
    return yr, yc


class DiscreteSAC:
    def __init__(self, seed=11):
        self.seed = seed
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = network()
            self.q = nn.ModuleList([network() for _ in range(2)])
            self.c = nn.ModuleList([network() for _ in range(3)])
        self.tq = copy.deepcopy(self.q)
        self.tc = copy.deepcopy(self.c)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.multipliers = nn.Parameter(torch.zeros(2))
        self.optimizers = {
            'actor': Adam(self.actor.parameters(), lr=3e-4),
            'q': Adam(self.q.parameters(), lr=3e-4),
            'c': Adam(self.c.parameters(), lr=3e-4),
            'alpha': Adam([self.log_alpha], lr=3e-4),
            'lambda': Adam([self.multipliers], lr=1e-3),
        }
        self.updates = 0

    def act(self, observation, mask, generator, deterministic=False):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(observation, dtype=torch.float32))
            return int(sample(logits, torch.as_tensor(mask, dtype=torch.bool),
                              generator, deterministic).item())

    def update(self, batch):
        b = {k: torch.as_tensor(v) for k, v in batch.items()}
        x, nx = b['observation'], b['next_observation']
        action = b['action'].long().reshape(-1, 1)
        with torch.no_grad():
            p, lp, _, _ = distribution(self.actor(nx), b['next_mask'])
            yr, yc = targets(
                b['reward'], b['costs'], b['discount'], p, lp,
                torch.minimum(self.tq[0](nx), self.tq[1](nx)),
                torch.stack([c(nx) for c in self.tc], dim=1),
                self.log_alpha.exp(),
            )
        qloss = sum(nn.functional.mse_loss(q(x).gather(1, action), yr)
                    for q in self.q)
        closs = sum(nn.functional.mse_loss(c(x).gather(1, action), yc[:, i:i+1])
                    for i, c in enumerate(self.c))
        for key, loss in [('q', qloss), ('c', closs)]:
            self.optimizers[key].zero_grad(set_to_none=True)
            loss.backward()
            self.optimizers[key].step()
        # Categorical expectations allow frozen critic values; the actor
        # differentiates through probabilities, never through the discrete ID.
        with torch.no_grad():
            q = torch.minimum(self.q[0](x), self.q[1](x))
            cq = torch.stack([c(x) for c in self.c], dim=1)
            weights = torch.cat([self.multipliers.detach(), torch.tensor([0.05])])
            penalized = -q + (weights[None, :, None] * cq).sum(1)
        p, lp, entropy, target = distribution(self.actor(x), b['mask'])
        actor_loss = (p * (self.log_alpha.detach().exp() * lp + penalized)).sum(-1).mean()
        self.optimizers['actor'].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optimizers['actor'].step()
        # Positive discrete target H*. If H < H*, descent increases alpha.
        alpha_loss = (self.log_alpha * (entropy - target).detach()).mean()
        self.optimizers['alpha'].zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.optimizers['alpha'].step()
        expected_cost = (p.detach().unsqueeze(1) * cq).sum(-1).mean(0)
        lambda_loss = -(self.multipliers * expected_cost[:2]).sum()
        self.optimizers['lambda'].zero_grad(set_to_none=True)
        lambda_loss.backward()
        self.optimizers['lambda'].step()
        with torch.no_grad():
            self.multipliers.clamp_(min=0)
            for original, target_networks in [(self.q, self.tq), (self.c, self.tc)]:
                for s, t in zip(original.parameters(), target_networks.parameters(), strict=True):
                    t.mul_(0.995).add_(s, alpha=0.005)
        self.updates += 1
        values = dict(updates=self.updates, actor_loss=actor_loss.item(),
                      reward_loss=qloss.item(), cost_loss=closs.item(),
                      entropy=entropy.mean().item(), target_entropy=target.mean().item(),
                      alpha=self.log_alpha.exp().item(),
                      speed_multiplier=self.multipliers[0].item(),
                      terminal_multiplier=self.multipliers[1].item(),
                      q_mean=q.mean().item(), q_min=q.min().item(), q_max=q.max().item(),
                      cost_mean=cq.mean().item())
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError('nonfinite learner')
        if not all(torch.isfinite(p).all() for m in self.modules().values()
                   for p in m.parameters()):
            raise FloatingPointError('nonfinite parameter')
        return values

    def modules(self):
        return dict(actor=self.actor, q=self.q, c=self.c, tq=self.tq, tc=self.tc)

    def state_dict(self):
        return dict(seed=self.seed, updates=self.updates,
                    modules={k: v.state_dict() for k, v in self.modules().items()},
                    optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
                    log_alpha=self.log_alpha.detach().clone(),
                    multipliers=self.multipliers.detach().clone())

    def load_state_dict(self, value):
        if value['seed'] != self.seed:
            raise ValueError('agent seed mutation')
        for k, v in value['modules'].items():
            self.modules()[k].load_state_dict(v)
        for k, v in value['optimizers'].items():
            self.optimizers[k].load_state_dict(v)
        with torch.no_grad():
            self.log_alpha.copy_(value['log_alpha'])
            self.multipliers.copy_(value['multipliers'])
        self.updates = value['updates']
