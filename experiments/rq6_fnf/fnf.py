"""Fair Normalizing Flows (Balunovic, Ruoss & Vechev, ICLR 2022), two variants.

Continuous variant: one RealNVP flow per group, per-group density priors (GMM or
flow), symmetric KL between the two induced representation densities, and a
shared classifier. Categorical variant: one MADE density per group over one-hot
combinations, then a (randomised) rank-matching map from the mapped group onto
the reference group's support. Both follow https://github.com/eth-sri/fnf with
explicit seeding, saved histories and checkpoints. They are ports for this
study, not a bit-exact reproduction of the published numbers.
"""
import itertools
import math
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn import mixture
from sklearn.linear_model import LogisticRegression


# ----------------------------------------------------------------------------- continuous inputs
class Dequantizer:
    """Paper preprocessing: affine map to (alpha/2, 1-alpha/2), uniform dequantisation, logit."""

    def __init__(self, alpha=.05):
        self.alpha = alpha

    @staticmethod
    def _quantum(x):
        q = np.full(x.shape[1], 1.)
        for j in range(x.shape[1]):
            u = np.unique(x[:, j])
            if len(u) > 1:
                q[j] = float(np.min(np.diff(u)))
        return q

    def fit(self, x):
        x = np.asarray(x, dtype='float64')
        self.lo = x.min(0)
        self.hi = x.max(0) + self._quantum(x)
        self.q = self._quantum(self.scale(x))
        return self

    def scale(self, x):
        x = np.clip(np.asarray(x, dtype='float64'), self.lo, self.hi)
        x = (x - self.lo) / (self.hi - self.lo) - .5
        return ((1 - self.alpha) * x + .5).astype('float32')

    def state(self):
        return dict(alpha=self.alpha, lo=self.lo.tolist(), hi=self.hi.tolist(), q=self.q.tolist())


def dequantize(scaled, q, alpha, generator=None):
    """scaled: tensor in (0,1); adds U[0,q) noise, clamps and maps to logit space."""
    noise = torch.rand(scaled.shape, generator=generator, device=scaled.device if generator is None else 'cpu')
    noisy = scaled + q * noise.to(scaled.device)
    return torch.clamp(noisy, alpha / 2, 1 - alpha / 2).logit()


# ----------------------------------------------------------------------------- RealNVP blocks
def _mlp(d, hidden, tanh=False):
    sizes = [d] + list(hidden) + [d]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Coupling(nn.Module):
    def __init__(self, d, hidden, mask):
        super().__init__()
        self.s = _mlp(d, hidden, tanh=True)
        self.t = _mlp(d, hidden)
        self.register_buffer('mask', torch.as_tensor(mask, dtype=torch.float32))

    def decode(self, z):
        """z -> x with log|det dx/dz|."""
        s, t = self.s(self.mask * z), self.t(self.mask * z)
        x = self.mask * z + (1 - self.mask) * (z * torch.exp(s) + t)
        return x, torch.sum(s * (1 - self.mask), dim=1)

    def encode(self, x):
        """x -> z with log|det dz/dx|."""
        s, t = self.s(self.mask * x), self.t(self.mask * x)
        z = self.mask * x + (1 - self.mask) * ((x - t) * torch.exp(-s))
        return z, -torch.sum(s * (1 - self.mask), dim=1)


def alternating_masks(d, n_blocks, rng):
    masks = []
    for _ in range(math.ceil(n_blocks / 2)):
        m = np.array([j % 2 for j in range(d)])
        rng.shuffle(m)
        masks += [m, 1 - m]
    return masks[:n_blocks]


class Flow(nn.Module):
    """Stack of couplings. encode: data -> latent; decode: latent -> data."""

    def __init__(self, d, hidden, n_blocks, masks):
        super().__init__()
        self.d = d
        self.layers = nn.ModuleList([Coupling(d, hidden, masks[i]) for i in range(n_blocks)])

    def encode(self, x):
        total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer.encode(x)
            total = total + ld
        return x, total

    def decode(self, z):
        total = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            z, ld = layer.decode(z)
            total = total + ld
        return z, total


class FlowPrior(nn.Module):
    """Density model p(x) = N(u) |det du/dx| with u = encode(x); sampling by decoding Gaussian noise."""

    def __init__(self, d, hidden, n_blocks, masks):
        super().__init__()
        self.flow = Flow(d, hidden, n_blocks, masks)
        self.register_buffer('zero', torch.zeros(d))

    def log_prob(self, x):
        u, ld = self.flow.encode(x)
        return torch.distributions.Normal(0., 1.).log_prob(u).sum(1) + ld

    def sample(self, n):
        u = torch.randn(n, self.zero.shape[0], device=self.zero.device)
        return self.flow.decode(u)[0]


class GMMPrior(nn.Module):
    def __init__(self, gmm, device):
        super().__init__()
        weights = torch.from_numpy(gmm.weights_).float().to(device)
        means = torch.from_numpy(gmm.means_).float().to(device)
        covs = torch.from_numpy(gmm.covariances_).float().to(device)
        self.mix = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(weights), torch.distributions.MultivariateNormal(means, covs))

    def log_prob(self, x):
        return self.mix.log_prob(x)

    def sample(self, n):
        return self.mix.sample((n,))


def fit_gmm_prior(x_train, x_val, components, seed, device):
    gmm = mixture.GaussianMixture(n_components=components, covariance_type='full', reg_covar=1e-4,
                                  n_init=1, random_state=seed, max_iter=300).fit(x_train)
    info = dict(kind='gmm', components=components, converged=bool(gmm.converged_),
                train_nll=float(-gmm.score(x_train)), validation_nll=float(-gmm.score(x_val)))
    return gmm, GMMPrior(gmm, device), info


def fit_flow_prior(x_train, x_val, hidden, n_blocks, epochs, seed, device, batch=256, lr=1e-2, weight_decay=1e-4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = x_train.shape[1]
    prior = FlowPrior(d, hidden, n_blocks, alternating_masks(d, n_blocks, rng)).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [epochs // 3, 2 * epochs // 3], .1)
    dl = DataLoader(TensorDataset(torch.from_numpy(x_train)), batch_size=batch, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    xv = torch.from_numpy(x_val).to(device)
    best, state, history = float('inf'), None, []
    for epoch in range(1, epochs + 1):
        prior.train()
        total = 0.
        for (bx,) in dl:
            bx = bx.to(device)
            opt.zero_grad()
            loss = -prior.log_prob(bx).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(prior.parameters(), 10.)
            opt.step()
            total += float(loss) * len(bx)
        sched.step()
        prior.eval()
        with torch.no_grad():
            val = float(-prior.log_prob(xv).mean())
        history.append(dict(epoch=epoch, train_nll=total / len(x_train), validation_nll=val))
        if np.isfinite(val) and val < best:
            best, state = val, {k: v.detach().cpu().clone() for k, v in prior.state_dict().items()}
    assert state is not None, 'Flow prior diverged'
    prior.load_state_dict(state)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    info = dict(kind='flow', hidden=list(hidden), n_blocks=n_blocks, epochs=epochs, validation_nll=best,
                selected_epoch=int(np.argmin([h['validation_nll'] for h in history])) + 1)
    return prior, info, history


# ----------------------------------------------------------------------------- continuous FNF
def latent_log_density(z, prior, flow):
    x, ld = flow.decode(z)
    return prior.log_prob(x) + ld


def make_classifier(d, dims):
    layers, prev = [], d
    for h in dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 2))
    return nn.Sequential(*layers)


def symmetric_kl(flows, priors, n):
    """Paper estimator: samples from each prior mapped into the shared latent space."""
    kls, indicators = [], []
    for g in (0, 1):
        h = 1 - g
        x = priors[g].sample(n)
        z, ld = flows[g].encode(x)
        own = priors[g].log_prob(x) - ld  # log p_Z(z) = log p_X(x) - log|det dz/dx|
        other = latent_log_density(z, priors[h], flows[h])
        kls.append((own - other).mean())
        indicators.append((own > other).float().mean())
    return .5 * (kls[0] + kls[1]), indicators


@torch.no_grad()
def statistical_distance(flows, priors, n=20000, latents=None):
    """TV-style estimate |P_0(log p0 > log p1) - P_1(log p0 > log p1)|.

    With latents=None the paper's prior-sample estimator is used; otherwise the
    given real latent tensors per group give a data-based estimate."""
    hits = []
    for g in (0, 1):
        z = flows[g].encode(priors[g].sample(n))[0] if latents is None else latents[g]
        l0 = latent_log_density(z, priors[0], flows[0])
        l1 = latent_log_density(z, priors[1], flows[1])
        hits.append((l0 > l1).float().mean().item())
    return abs(hits[0] - hits[1]), hits


def group_loader(x, y, batch, seed, shuffle=True, drop_last=True):
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=shuffle, drop_last=drop_last,
                      generator=torch.Generator().manual_seed(seed))


def train_continuous_fnf(cfg, train, valid, priors, q, seed, device, log):
    """train/valid: dict group -> (scaled x tensor, y long tensor) on device. Returns flows, clf, history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = q.numel()
    masks = alternating_masks(d, cfg['n_blocks'], rng)
    flows = nn.ModuleList([Flow(d, cfg['flow_hidden'], cfg['n_blocks'], masks) for _ in range(2)]).to(device)
    flows[0].load_state_dict(flows[1].state_dict())  # paper: identical initialisation
    clf = make_classifier(d, cfg['clf_hidden']).to(device)
    opt = torch.optim.Adam(list(flows.parameters()) + list(clf.parameters()), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    epochs = cfg['epochs']
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [epochs // 3, 2 * epochs // 3], .1)
    loaders = [group_loader(train[g][0], train[g][1], cfg['batch_size'], seed + g) for g in (0, 1)]
    history, started = [], time.time()
    alpha = cfg['alpha']
    for epoch in range(1, epochs + 1):
        flows.train(); clf.train()
        if cfg['kl_end'] > cfg['kl_start']:
            gamma = 0. if epoch - 1 < cfg['kl_start'] else cfg['gamma'] * min((epoch - cfg['kl_start']) / (cfg['kl_end'] - cfg['kl_start']), 1.)
        else:
            gamma = cfg['gamma']
        sums = dict(kl=0., pred=0., correct=0., n=0, batches=0)
        for (x0, y0), (x1, y1) in zip(*loaders):
            opt.zero_grad()
            kl, _ = symmetric_kl(flows, priors, x0.shape[0])
            pred_losses, correct = [], 0.
            for g, (x, y) in enumerate([(x0, y0), (x1, y1)]):
                z = flows[g].encode(dequantize(x, q, alpha))[0]
                out = clf(z)
                pred_losses.append(F.cross_entropy(out, y))
                correct += (out.argmax(1) == y).float().sum().item()
            pred = .5 * (pred_losses[0] + pred_losses[1])
            loss = gamma * kl + (1 - gamma) * pred
            loss.backward()
            nn.utils.clip_grad_norm_(list(flows.parameters()) + list(clf.parameters()), cfg['grad_clip'])
            opt.step()
            sums['kl'] += float(kl); sums['pred'] += float(pred); sums['correct'] += correct
            sums['n'] += x0.shape[0] + x1.shape[0]; sums['batches'] += 1
        sched.step()
        assert np.isfinite(sums['kl']) and np.isfinite(sums['pred']), 'Training diverged'
        row = dict(epoch=epoch, gamma_effective=gamma, train_kl=sums['kl'] / max(sums['batches'], 1),
                   train_pred=sums['pred'] / max(sums['batches'], 1), train_accuracy=sums['correct'] / max(sums['n'], 1),
                   elapsed_seconds=time.time() - started)
        if epoch % cfg['log_every'] == 0 or epoch == epochs:
            row.update(evaluate_continuous(flows, clf, priors, valid, q, alpha, seed + 7))
        history.append(row)
        log(f"epoch {epoch} gamma {gamma:.3f} kl {row['train_kl']:.4f} pred {row['train_pred']:.4f}"
            + (f" val_bce {row['validation_bce']:.4f} val_stat_dist {row['validation_stat_dist_data']:.4f}" if 'validation_bce' in row else ''))
    flows.eval(); clf.eval()
    return flows, clf, history


@torch.no_grad()
def encode_continuous(flows, clf, x_scaled, a, q, alpha, seed, device, batch=4096):
    """Fixed-seed dequantisation, group-specific encoding; returns z and P(y=1)."""
    gen = torch.Generator().manual_seed(seed)
    zs, ps = np.zeros((len(x_scaled), q.numel()), "float32"), np.zeros(len(x_scaled), "float32")
    x_all = torch.from_numpy(x_scaled)
    noise_all = torch.rand(x_all.shape, generator=gen)
    for start in range(0, len(x_scaled), batch):
        x = x_all[start:start + batch].to(device)
        noise = noise_all[start:start + batch].to(device)
        aa = torch.from_numpy(np.asarray(a[start:start + batch])).to(device)
        xin = torch.clamp(x + q * noise, alpha / 2, 1 - alpha / 2).logit()
        z = torch.empty_like(xin)
        for g in (0, 1):
            m = aa == g
            if m.any():
                z[m] = flows[g].encode(xin[m])[0]
        zs[start:start + batch] = z.cpu().numpy()
        ps[start:start + batch] = torch.softmax(clf(z), 1)[:, 1].cpu().numpy()
    return zs, ps


@torch.no_grad()
def evaluate_continuous(flows, clf, priors, valid, q, alpha, seed):
    flows.eval(); clf.eval()
    gen = torch.Generator().manual_seed(seed)
    out, latents = {}, []
    losses, correct, n, positives = [], 0., 0, []
    for g in (0, 1):
        x, y = valid[g]
        z = flows[g].encode(dequantize(x, q, alpha, gen))[0]
        logits = clf(z)
        losses.append(F.cross_entropy(logits, y).item())
        correct += (logits.argmax(1) == y).float().sum().item(); n += len(y)
        positives.append((logits.argmax(1) == 1).float().mean().item())
        latents.append(z)
    out['validation_bce'] = .5 * (losses[0] + losses[1])
    out['validation_accuracy'] = correct / n
    out['validation_dp_argmax'] = abs(positives[0] - positives[1])
    out['validation_stat_dist_prior'], _ = statistical_distance(flows, priors, n=4000)
    out['validation_stat_dist_data'], _ = statistical_distance(flows, priors, latents=latents)
    flows.train(); clf.train()
    return out


# ----------------------------------------------------------------------------- categorical FNF
class MaskedLinear(nn.Linear):
    def __init__(self, i, o):
        super().__init__(i, o)
        self.register_buffer('mask', torch.ones(o, i))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """Masked autoencoder for binary vectors; log_prob sums Bernoulli log-likelihoods."""

    def __init__(self, d, hidden, rng):
        super().__init__()
        sizes = [d] + list(hidden) + [d]
        self.net = nn.ModuleList([MaskedLinear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])
        degrees = [np.arange(d)]
        for h in hidden:
            degrees.append(rng.integers(degrees[-1].min(), d - 1, size=h))
        for i, layer in enumerate(self.net[:-1]):
            layer.mask.copy_(torch.from_numpy((degrees[i][:, None] <= degrees[i + 1][None, :]).T.astype('float32')))
        self.net[-1].mask.copy_(torch.from_numpy((degrees[-1][:, None] < degrees[0][None, :]).T.astype('float32')))
        self.direct = MaskedLinear(d, d)
        # Output i may see input j directly only when deg(j) < deg(i); mask is indexed [out, in].
        self.direct.mask.copy_(torch.from_numpy((degrees[0][:, None] > degrees[0][None, :]).astype('float32')))

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.net):
            h = layer(h)
            if i < len(self.net) - 1:
                h = F.relu(h)
        return h + self.direct(x)

    def log_prob(self, x):
        return -F.binary_cross_entropy_with_logits(self.forward(x), x, reduction='none').sum(-1)


def train_made(x_train, x_val, hidden, epochs, seed, device, lr=1e-2, weight_decay=1e-4, batch=64):
    torch.manual_seed(seed)
    made = MADE(x_train.shape[1], hidden, np.random.default_rng(seed)).to(device)
    opt = torch.optim.Adam(made.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 2, 1), gamma=.1)
    dl = DataLoader(TensorDataset(torch.from_numpy(x_train)), batch_size=batch, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    xv = torch.from_numpy(x_val).to(device)
    best, state, history = float('inf'), None, []
    for epoch in range(1, epochs + 1):
        made.train()
        total = 0.
        for (bx,) in dl:
            bx = bx.to(device)
            opt.zero_grad()
            loss = -made.log_prob(bx).mean()
            loss.backward()
            opt.step()
            total += float(loss) * len(bx)
        sched.step()
        made.eval()
        with torch.no_grad():
            val = float(-made.log_prob(xv).mean())
        history.append(dict(epoch=epoch, train_nll=total / len(x_train), validation_nll=val))
        if val < best:
            best, state = val, {k: v.detach().cpu().clone() for k, v in made.state_dict().items()}
    made.load_state_dict(state)
    made.eval()
    return made, dict(kind='made', hidden=list(hidden), epochs=epochs, validation_nll=best), history


def combination_index(codes, dims):
    """Mixed-radix index of per-feature category codes (n, k) -> (n,)."""
    idx = np.zeros(len(codes), dtype=np.int64)
    for j, k in enumerate(dims):
        idx = idx * k + codes[:, j]
    return idx


def index_to_codes(idx, dims):
    codes = np.zeros((len(idx), len(dims)), dtype=np.int64)
    for j in range(len(dims) - 1, -1, -1):
        codes[:, j] = idx % dims[j]
        idx = idx // dims[j]
    return codes


def one_hot(codes, dims):
    offsets = np.concatenate([[0], np.cumsum(dims)[:-1]])
    out = np.zeros((len(codes), int(sum(dims))), dtype='float32')
    rows = np.repeat(np.arange(len(codes)), len(dims))
    out[rows, (codes + offsets[None, :]).ravel()] = 1.
    return out


@torch.no_grad()
def enumerate_log_probs(made, dims, device, batch=131072):
    """Normalised log-probability of every category combination under a MADE."""
    total = int(np.prod(dims))
    logp = np.zeros(total, dtype='float64')
    for start in range(0, total, batch):
        idx = np.arange(start, min(start + batch, total))
        x = torch.from_numpy(one_hot(index_to_codes(idx, dims), dims)).to(device)
        logp[start:start + len(idx)] = made.log_prob(x).double().cpu().numpy()
    # MADE puts mass on non-one-hot vectors; renormalise over valid combinations (paper NOTE).
    m = logp.max()
    return logp - (m + np.log(np.exp(logp - m).sum()))


def rank_matching_maps(logp_ref, logp_map, predicted_class, gamma):
    """Paper mapping: mapped-group combos -> reference-group combos by probability rank.

    Returns perm_group (within predicted class), perm_all (global), the implied
    reference/mapped latent distributions and the exact statistical distance."""
    total = len(logp_ref)
    perm_group = np.empty(total, dtype=np.int64)
    for g in np.unique(predicted_class):
        members = np.flatnonzero(predicted_class == g)
        ref_order = members[np.argsort(logp_ref[members], kind='stable')]
        map_order = members[np.argsort(logp_map[members], kind='stable')]
        perm_group[map_order] = ref_order
    perm_all = np.empty(total, dtype=np.int64)
    perm_all[np.argsort(logp_map, kind='stable')] = np.arange(total)[np.argsort(logp_ref, kind='stable')]
    p_ref, p_map = np.exp(logp_ref), np.exp(logp_map)
    pushed_group = np.zeros(total); pushed_group[perm_group] = p_map
    pushed_all = np.zeros(total); pushed_all[perm_all] = p_map
    p_z_map = (1 - gamma) * pushed_group + gamma * pushed_all
    diff = p_ref - p_z_map
    stat_dist = max(abs(diff[diff > 0].sum()), abs(diff[diff < 0].sum()))
    return perm_group, perm_all, float(stat_dist)


def encode_categorical(codes, a, dims, perm_group, perm_all, gamma, reference_group, seed):
    """Reference group unchanged; the other group is mapped, using perm_all with probability gamma."""
    idx = combination_index(codes, dims)
    rng = np.random.default_rng(seed)
    use_all = rng.random(len(idx)) < gamma
    mapped = np.where(use_all, perm_all[idx], perm_group[idx])
    out = np.where(np.asarray(a) == reference_group, idx, mapped)
    return one_hot(index_to_codes(out, dims), dims), out


def train_group_balanced_classifier(z, y, a, dims, epochs, seed, device, lr=1e-3, weight_decay=0., batch=128, balanced=True):
    """Paper classifier: mean of the two per-group cross-entropies each batch (balanced=False: plain ERM)."""
    torch.manual_seed(seed)
    clf = make_classifier(z.shape[1], dims).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    dl = DataLoader(TensorDataset(torch.from_numpy(z), torch.from_numpy(np.asarray(y)).long(),
                                  torch.from_numpy(np.asarray(a)).long()), batch_size=batch, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    history = []
    for epoch in range(1, epochs + 1):
        clf.train()
        total, n = 0., 0
        for bz, by, ba in dl:
            bz, by, ba = bz.to(device), by.to(device), ba.to(device)
            opt.zero_grad()
            out = clf(bz)
            if balanced:
                losses = [F.cross_entropy(out[ba == g], by[ba == g]) for g in (0, 1) if (ba == g).any()]
                loss = sum(losses) / len(losses)
            else:
                loss = F.cross_entropy(out, by)
            loss.backward()
            opt.step()
            total += float(loss) * len(by); n += len(by)
        history.append(dict(epoch=epoch, train_loss=total / n))
    clf.eval()
    return clf, history


@torch.no_grad()
def predict_proba(clf, z, device, batch=8192):
    out = []
    for start in range(0, len(z), batch):
        out.append(torch.softmax(clf(torch.from_numpy(z[start:start + batch]).to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype('float32')


# ----------------------------------------------------------------------------- paper adversary
def paper_adversary(z_train, a_train, z_val, a_val, tests, hidden, epochs, lr, weight_decay, seed, device, batch=256):
    """The paper's post-hoc adversary: fixed architecture, group-balanced CE, final-epoch balanced accuracy.

    Independent probes in evaluate.py are the primary decodability measure;
    this reproduces the paper's own reporting convention."""
    torch.manual_seed(seed)
    adv = make_classifier(z_train.shape[1], hidden).to(device)
    opt = torch.optim.Adam(adv.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [epochs // 3, 2 * epochs // 3], .1)
    dl = DataLoader(TensorDataset(torch.from_numpy(z_train), torch.from_numpy(np.asarray(a_train)).long()),
                    batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        adv.train()
        for bz, ba in dl:
            bz, ba = bz.to(device), ba.to(device)
            opt.zero_grad()
            out = adv(bz)
            losses = [F.cross_entropy(out[ba == g], ba[ba == g]) for g in (0, 1) if (ba == g).any()]
            (sum(losses) / len(losses)).backward()
            opt.step()
        sched.step()
    adv.eval()

    def balanced(z, a):
        a = np.asarray(a)
        pred = predict_proba(adv, z, device) >= .5
        return float(np.mean([np.mean(pred[a == g] == g) for g in (0, 1) if (a == g).any()]))

    result = dict(validation_balanced_accuracy=balanced(z_val, a_val))
    for name, (z, a) in tests.items():
        result[f'{name}_balanced_accuracy'] = balanced(z, a)
    return result
