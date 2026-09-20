"""Conditional FNF: shared-across-label sex flows, class-specific training priors."""
import numpy as np
import torch
from torch import nn
from experiments.rq6_fnf.fnf import Flow, GMMPrior, alternating_masks


def grid():
    from .common import SEED, PATHOLOGY
    return [dict(index=i, id=f'{i+1:02d}_conditional_fnf', method='conditional_fnf',
                 dim=8, gamma=g, seed=SEED, pathology=PATHOLOGY, condition='shift50')
            for i, g in enumerate([.05, .5, .95])]


def latent_log_density(z, prior, flow):
    x, inverse_logdet = flow.decode(z)
    return prior.log_prob(x) + inverse_logdet


def conditional_kl(flows, priors, n):
    """Equal label weights; half-symmetric KL per label. priors[y][a]."""
    losses = []
    for y in (0, 1):
        directional = []
        for a in (0, 1):
            x = priors[y][a].sample(n)
            z, ld = flows[a].encode(x)
            own = priors[y][a].log_prob(x) - ld
            other = latent_log_density(z, priors[y][1-a], flows[1-a])
            directional.append((own-other).mean())
        losses.append(sum(directional)/2)
    return sum(losses)/2, losses


def mmd(z, a):
    """RQ8 multiscale, median-bandwidth biased MMD estimator."""
    z = (z-z.mean(0))/z.std(0, unbiased=False).clamp_min(1e-5)
    dist = torch.cdist(z, z).square()
    vals = dist.detach()[dist.detach() > 1e-10]
    bw = vals.median() if len(vals) else dist.new_tensor(1.)
    k = sum(torch.exp(-dist/(2*bw.clamp_min(1e-6)*s)) for s in [.25,.5,1,2,4])/5
    u, v = a == 0, a == 1
    if min(int(u.sum()), int(v.sum())) < 2:
        raise ValueError('Conditional MMD requires both sex groups')
    return k[u][:,u].mean()+k[v][:,v].mean()-2*k[u][:,v].mean()


class ConditionalFNF(nn.Module):
    """Inference accepts x and a only. No label-specific routing exists."""
    def __init__(self, dim=8, seed=42):
        super().__init__()
        masks = alternating_masks(dim, 4, np.random.default_rng(seed))
        self.flows = nn.ModuleList([Flow(dim, [128,128], 4, masks) for _ in (0,1)])
        self.flows[1].load_state_dict(self.flows[0].state_dict())
        self.head = nn.Sequential(nn.Linear(dim,64), nn.ReLU(), nn.Linear(64,1))

    def forward(self, x, a):
        z = torch.empty_like(x)
        for g in (0,1):
            mask = a == g
            if mask.any():
                z[mask] = self.flows[g].encode(x[mask])[0]
        return z, self.head(z).flatten()
