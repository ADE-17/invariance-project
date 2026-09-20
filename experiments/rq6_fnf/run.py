"""One Slurm array shard: dataset x seed x prevalence condition; all FNF configurations plus ERM."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rq5_sweep.data import sha
from experiments.rq5_sweep.run import write_json
from experiments.rq6_fnf.data import BASE, DATASETS, CONDITIONS, SOURCE_SPLITS, load, apply_condition
from experiments.rq6_fnf import fnf

SEEDS = [42, 123, 456]
SHARDS = [dict(dataset=d, seed=s, condition=c) for d in DATASETS for s in SEEDS for c in CONDITIONS]
CONTINUOUS_GAMMAS = [0., .05, .1, .5, .95]
CATEGORICAL_GAMMAS = [0., .02, .1, .2, .5, .9, 1.]
PRIORS = ['gmm', 'flow']
N_BLOCKS = [4, 6]
MADE_HIDDEN = [50, 100]
REFERENCE_GROUP = 1  # categorical variant keeps group 1 (the higher-prevalence group in every dataset) fixed and maps group 0 onto it.

CONTINUOUS = dict(epochs=80, batch_size=256, lr=1e-3, weight_decay=0., flow_hidden=[100, 100, 50],
                  clf_hidden=[100, 50, 20], kl_start=0, kl_end=10, alpha=.05, grad_clip=10., log_every=10,
                  prior_flow_hidden=[50, 50], prior_flow_blocks=4, prior_epochs=80, gmm_components=4,
                  adversary_hidden=[50, 50], adversary_epochs=80, adversary_lr=1e-2, adversary_weight_decay=0.)
CATEGORICAL = dict(made_epochs=100, clf_epochs=100, clf_hidden=[20], clf_lr=1e-3, clf_weight_decay=1e-4,
                   age_bins=4, adversary_hidden=[30, 30], adversary_epochs=60, adversary_lr=1e-2,
                   adversary_weight_decay=1e-3)
SMOKE = dict(epochs=3, prior_epochs=3, made_epochs=3, clf_epochs=3, adversary_epochs=3, log_every=1)


# Class-conditional trial (Z independent of A within each stratum of the conditioning variable):
#   condition_on='y'    oracle: true label used to pick the per-stratum map on every split, including test (theoretical check)
#   condition_on='yhat' deployable: the logistic-regression predicted class picks the map (no label needed at test time)
CONDITIONAL_TRIAL = dict(adult=dict(feature_set='paper6_age', made_hidden=50), dutch=dict(feature_set='core6_hh', made_hidden=50),
                         crime=dict(feature_set='socio4', made_hidden=50), credit=dict(feature_set='core5', made_hidden=50),
                         compas=dict(feature_set='paper7', made_hidden=50))
CONDITIONAL_GAMMAS = [.5, 1.]
CONDITION_ON = ['y', 'yhat']


def conditional_configurations(dataset):
    base = CONDITIONAL_TRIAL[dataset]
    return [dict(method='fnf', variant='categorical_conditional', condition_on=c, gamma=g, **base) for c in CONDITION_ON for g in CONDITIONAL_GAMMAS]


def configurations(dataset, feature_sets):
    runs = [dict(method='erm', feature_set=f) for f in feature_sets]
    if dataset == 'health':
        runs += [dict(method='fnf', variant='continuous', feature_set=f, gamma=g, prior=p, n_blocks=b)
                 for f in feature_sets for g in CONTINUOUS_GAMMAS for p in PRIORS for b in N_BLOCKS]
    else:
        runs += [dict(method='fnf', variant='categorical', feature_set=f, gamma=g, made_hidden=h)
                 for f in feature_sets for g in CATEGORICAL_GAMMAS for h in MADE_HIDDEN]
    return runs


def run_name(cfg):
    if cfg['method'] == 'erm':
        return f"erm_{cfg['feature_set']}"
    if cfg['variant'] == 'continuous':
        return f"fnf_{cfg['feature_set']}_gamma{cfg['gamma']:g}_prior-{cfg['prior']}_blocks{cfg['n_blocks']}"
    if cfg['variant'] == 'categorical_conditional':
        return f"cfnf_{cfg['feature_set']}_cond-{cfg['condition_on']}_gamma{cfg['gamma']:g}_made{cfg['made_hidden']}"
    return f"fnf_{cfg['feature_set']}_gamma{cfg['gamma']:g}_made{cfg['made_hidden']}"


def code_hashes():
    here = Path(__file__).parent
    paths = sorted(here.glob('*.py'))
    from experiments.rq5_guarantees import prepare
    from experiments.rq5_sweep import data as rq5data, run as rq5run
    paths += [Path(m.__file__) for m in [prepare, rq5data, rq5run]]
    return {str(p): sha(p) for p in paths}


def source_sets(d, condition, smoke):
    sets = {s: apply_condition(d[d.split == s].copy(), condition, s) for s in SOURCE_SPLITS}
    sets['test'] = d[d.split == 'test'].copy()  # natural; variants are applied in evaluation
    if smoke:
        rng = np.random.default_rng(0)
        sets = {s: g.iloc[np.sort(rng.choice(len(g), min(len(g), 1500 if s == 'train' else 600), replace=False))]
                for s, g in sets.items()}
    return sets


class Log:
    def __init__(self, folder):
        self.path = folder / 'train.log'

    def __call__(self, message):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}'
        print(line, flush=True)
        with open(self.path, 'a') as f:
            f.write(line + '\n')


# ----------------------------------------------------------------------------- feature construction
def health_matrices(sets, columns):
    return {s: g[columns].to_numpy(dtype='float32') for s, g in sets.items()}


def categorical_codes(sets, columns, cfg, binned=None):
    """Category codes per feature; numeric features are quantile-binned on the training split.

    `binned` maps column -> requested bin count (manifest['binned_features']); Adult's age keeps cfg['age_bins'].
    Duplicate quantile edges (heavy ties) are dropped, so the realised bin count can be smaller than requested."""
    binned = dict(binned or {})
    if 'x_age' in columns and 'x_age' not in binned and pd.api.types.is_numeric_dtype(sets['train']['x_age']):
        binned['x_age'] = cfg['age_bins']
    train = sets['train']
    vocab, dims, edges = {}, [], {}
    for c in columns:
        if c in binned:
            e = np.unique(np.quantile(train[c].astype(float), np.linspace(0, 1, binned[c] + 1))[1:-1])
            edges[c] = e.tolist()
            dims.append(len(e) + 1)
        else:
            cats = sorted(pd.concat([g[c] for g in sets.values()]).astype(str).unique().tolist())
            vocab[c] = cats
            dims.append(len(cats))
    codes = {}
    for s, g in sets.items():
        cols = []
        for c in columns:
            if c in binned:
                cols.append(np.digitize(g[c].to_numpy(dtype=float), edges[c]))
            else:
                cols.append(pd.Categorical(g[c].astype(str), categories=vocab[c]).codes.astype(np.int64))
        codes[s] = np.stack(cols, 1)
        assert (codes[s] >= 0).all()
    return codes, dims, dict(vocabulary=vocab, bin_edges=edges, age_bins=edges.get('x_age'), dims=dims)


adult_codes = categorical_codes  # backwards-compatible name


# ----------------------------------------------------------------------------- shared priors
def prior_folder(shard, feature_set, name):
    return BASE / ('smoke_priors' if shard.get('smoke') else 'priors') / shard['dataset'] / f"seed{shard['seed']}" / shard['condition'] / f'{feature_set}_{name}'


def continuous_priors(shard, feature_set, kind, scaled, dq, sets, params, device, log):
    folder = prior_folder(shard, feature_set, kind)
    folder.mkdir(parents=True, exist_ok=True)
    q = torch.tensor(dq.q, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(shard['seed'] + 11)
    data = {}
    for split in ['train', 'validation']:
        x = fnf.dequantize(torch.from_numpy(scaled[split]), torch.tensor(dq.q, dtype=torch.float32), dq.alpha, gen).numpy()
        a = sets[split].a.to_numpy()
        data[split] = {g: x[a == g] for g in (0, 1)}
    priors, infos = [], []
    for g in (0, 1):
        if kind == 'gmm':
            path = folder / f'group{g}.joblib'
            if path.exists():
                gmm = joblib.load(path)
                prior = fnf.GMMPrior(gmm, device)
                info = json.loads((folder / f'group{g}.json').read_text())
            else:
                gmm, prior, info = fnf.fit_gmm_prior(data['train'][g], data['validation'][g], params['gmm_components'], shard['seed'] + g, device)
                joblib.dump(gmm, path)
                write_json(folder / f'group{g}.json', info)
        else:
            path = folder / f'group{g}.pt'
            rng = np.random.default_rng(shard['seed'] + 20 + g)
            if path.exists():
                saved = torch.load(path, map_location=device, weights_only=False)
                prior = fnf.FlowPrior(scaled['train'].shape[1], params['prior_flow_hidden'], params['prior_flow_blocks'], saved['masks']).to(device)
                prior.load_state_dict(saved['state']); prior.eval()
                for p in prior.parameters(): p.requires_grad_(False)
                info = json.loads((folder / f'group{g}.json').read_text())
            else:
                prior, info, history = fnf.fit_flow_prior(data['train'][g], data['validation'][g], params['prior_flow_hidden'],
                                                          params['prior_flow_blocks'], params['prior_epochs'], shard['seed'] + 20 + g, device)
                torch.save(dict(state=prior.state_dict(), masks=[m.tolist() for m in [l.mask.cpu().numpy().astype(int) for l in prior.flow.layers]]), path)
                pd.DataFrame(history).to_csv(folder / f'group{g}_history.csv', index=False)
                write_json(folder / f'group{g}.json', info)
        log(f'prior {kind} group{g}: {info}')
        priors.append(prior); infos.append(info)
    with torch.no_grad():
        # Cross-group log-likelihood diagnostics: how separable the groups are in input space.
        diag = {}
        for g in (0, 1):
            xv = torch.from_numpy(data['validation'][g]).to(device)
            diag[f'validation_group{g}_nll_own'] = float(-priors[g].log_prob(xv).mean())
            diag[f'validation_group{g}_nll_other'] = float(-priors[1 - g].log_prob(xv).mean())
    write_json(folder / 'diagnostics.json', diag)
    return priors, infos, diag


def made_priors(shard, feature_set, hidden, codes, dims, sets, params, device, log, mask=None, tag=''):
    """Per-group MADE densities; `mask` (split -> bool array) restricts fitting to one stratum, `tag` names its cache."""
    folder = prior_folder(shard, feature_set, f'made{hidden}{tag}')
    folder.mkdir(parents=True, exist_ok=True)
    mades, infos, logps = [], [], []
    for g in (0, 1):
        path = folder / f'group{g}.pt'
        if path.exists():
            made = fnf.MADE(int(sum(dims)), [hidden, hidden], np.random.default_rng(shard['seed'] + g)).to(device)
            made.load_state_dict(torch.load(path, map_location=device, weights_only=True)); made.eval()
            info = json.loads((folder / f'group{g}.json').read_text())
        else:
            mt = sets['train'].a.to_numpy() == g
            mv = sets['validation'].a.to_numpy() == g
            if mask is not None:
                mt, mv = mt & mask['train'], mv & mask['validation']
            xt = fnf.one_hot(codes['train'][mt], dims)
            xv = fnf.one_hot(codes['validation'][mv], dims)
            made, info, history = fnf.train_made(xt, xv, [hidden, hidden], params['made_epochs'], shard['seed'] + g, device)
            torch.save(made.state_dict(), path)
            pd.DataFrame(history).to_csv(folder / f'group{g}_history.csv', index=False)
            write_json(folder / f'group{g}.json', info)
        log(f'made{hidden} group{g}: {info}')
        cache = folder / f'group{g}_logp.npy'
        if cache.exists():
            logp = np.load(cache)
        else:
            logp = fnf.enumerate_log_probs(made, dims, device)
            np.save(cache, logp)
        mades.append(made); infos.append(info); logps.append(logp)
    return mades, infos, logps


# ----------------------------------------------------------------------------- one run
def save_representations(folder, sets, z, p):
    arrays = {}
    for s in sets:
        arrays[f'z_{s}'] = z[s].astype('float32')
        arrays[f'task_probability_{s}'] = p[s].astype('float32')
        arrays[f'row_id_{s}'] = sets[s].row_id.to_numpy()
    np.savez_compressed(folder / 'representations.npz', **arrays)


def adversaries(z, sets, params, seed, device):
    out = {}
    for label in [None, 0, 1]:
        def pick(s):
            m = np.ones(len(sets[s]), bool) if label is None else sets[s].y.to_numpy() == label
            return z[s][m], sets[s].a.to_numpy()[m]
        tr, va = pick('train'), pick('probe_validation')
        res = fnf.paper_adversary(tr[0], tr[1], va[0], va[1], {'test': pick('test')}, params['adversary_hidden'],
                                  params['adversary_epochs'], params['adversary_lr'], params['adversary_weight_decay'],
                                  seed + (0 if label is None else 100 * (label + 1)), device)
        suffix = 'marginal' if label is None else f'Y{label}'
        out.update({f'paper_adversary_{suffix}_{k}': v for k, v in res.items()})
    return out


def run_one(shard, cfg, sets, prepared, device, args):
    root = BASE / ('smoke' if args.smoke else 'runs') / shard['dataset'] / f"seed{shard['seed']}" / shard['condition']
    folder = root / run_name(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        print('Verified existing run', folder, flush=True)
        return
    log = Log(folder)
    params = dict(CONTINUOUS if shard['dataset'] == 'health' else CATEGORICAL)
    if args.smoke:
        params.update({k: v for k, v in SMOKE.items() if k in params})
    config = dict(**{k: v for k, v in shard.items() if k != 'smoke'}, **cfg, params=params, smoke=args.smoke, seed_policy='model seed = shard seed; splits fixed',
                  source_hashes=code_hashes(), reference_group=REFERENCE_GROUP if cfg.get('variant') == 'categorical' else None)
    write_json(folder / 'config.json', config)
    write_json(folder / 'runtime.json', dict(device=device, torch_version=str(torch.__version__), cuda=torch.version.cuda,
                                            gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                                            slurm_job_id=os.environ.get('SLURM_JOB_ID'), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    started = time.time()
    seed = shard['seed']
    summary = dict(**{k: v for k, v in shard.items() if k != 'smoke'}, **cfg, run=run_name(cfg))
    feature = prepared[cfg['feature_set']]
    a = {s: g.a.to_numpy() for s, g in sets.items()}
    y = {s: g.y.to_numpy() for s, g in sets.items()}
    z, p = {}, {}
    if cfg['method'] == 'erm':
        if shard['dataset'] == 'health':
            scaler = StandardScaler().fit(feature['raw']['train'])
            z = {s: scaler.transform(x).astype('float32') for s, x in feature['raw'].items()}
            joblib.dump(scaler, folder / 'scaler.joblib')
            epochs, hidden, lr, wd, batch = params['epochs'], params['clf_hidden'], params['lr'], params['weight_decay'], params['batch_size']
        else:
            z = {s: fnf.one_hot(c, feature['dims']) for s, c in feature['codes'].items()}
            epochs, hidden, lr, wd, batch = params['clf_epochs'], params['clf_hidden'], params['clf_lr'], params['clf_weight_decay'], 128
        clf, history = fnf.train_group_balanced_classifier(z['train'], y['train'], a['train'], hidden, epochs, seed, device, lr, wd, batch, balanced=False)
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        torch.save(clf.state_dict(), folder / 'classifier.pt')
        p = {s: fnf.predict_proba(clf, z[s], device) for s in sets}
        summary['representation'] = 'standardised input features' if shard['dataset'] == 'health' else 'one-hot input features'
    elif cfg['variant'] == 'continuous':
        dq, scaled = feature['dequantizer'], feature['scaled']
        priors, infos, diag = continuous_priors(shard, cfg['feature_set'], cfg['prior'], scaled, dq, sets, params, device, log)
        q = torch.tensor(dq.q, dtype=torch.float32, device=device)
        tensors = {}
        for split in ['train', 'validation']:
            xs = torch.from_numpy(scaled[split]).to(device)
            ys = torch.from_numpy(y[split]).long().to(device)
            tensors[split] = {g: (xs[a[split] == g], ys[a[split] == g]) for g in (0, 1)}
        train_cfg = dict(params, gamma=cfg['gamma'], n_blocks=cfg['n_blocks'])
        flows, clf, history = fnf.train_continuous_fnf(train_cfg, tensors['train'], tensors['validation'], priors, q, seed, device, log)
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        torch.save(dict(flows=flows.state_dict(), classifier=clf.state_dict(), masks=[l.mask.cpu().numpy().astype(int).tolist() for l in flows[0].layers],
                        dequantizer=dq.state()), folder / 'model.pt')
        for s in sets:
            z[s], p[s] = fnf.encode_continuous(flows, clf, scaled[s], a[s], q, dq.alpha, seed + 500 + SOURCE_SPLITS.index(s) if s in SOURCE_SPLITS else seed + 599, device)
        with torch.no_grad():
            latents = {s: [torch.from_numpy(z[s][a[s] == g]).to(device) for g in (0, 1)] for s in ['validation', 'test']}
            summary['stat_dist_prior_samples'], hits = fnf.statistical_distance(flows, priors, n=20000)
            summary['stat_dist_prior_hits'] = hits
            for s in latents:
                summary[f'stat_dist_data_{s}'], summary[f'stat_dist_data_{s}_hits'] = fnf.statistical_distance(flows, priors, latents=latents[s])
        summary['prior_info'] = infos
        summary['prior_diagnostics'] = diag
        summary['final_history'] = history[-1]
        summary['representation'] = 'group-specific RealNVP latent of dequantised inputs (sensitive attribute required at encoding)'
    elif cfg['variant'] == 'categorical_conditional':
        codes, dims = feature['codes'], feature['dims']
        total = int(np.prod(dims))
        xt = fnf.one_hot(codes['train'], dims)
        n1 = (a['train'] == REFERENCE_GROUP).sum() / len(a['train'])
        weights = np.where(a['train'] == REFERENCE_GROUP, 1 - n1, n1)
        logreg = LogisticRegression(max_iter=2000).fit(xt, y['train'], sample_weight=weights)
        joblib.dump(logreg, folder / 'mapping_logreg.joblib')
        predicted = np.concatenate([logreg.predict(fnf.one_hot(fnf.index_to_codes(np.arange(st, min(st + 200000, total)), dims), dims))
                                    for st in range(0, total, 200000)])
        idx = {s: fnf.combination_index(codes[s], dims) for s in sets}
        yhat = {s: predicted[idx[s]] for s in sets}
        strata = y if cfg['condition_on'] == 'y' else yhat
        ref, mapped = REFERENCE_GROUP, 1 - REFERENCE_GROUP
        maps, infos = {}, {}
        for v in (0, 1):
            mask = {s: strata[s] == v for s in sets}
            for s, minimum in (('train', 20), ('validation', 5)):
                for g in (0, 1):
                    # Sparse strata are a property of the deployable variant under shift (few predicted positives in the
                    # low-prevalence group); sizes are recorded in training_summary for interpretation.
                    assert (mask[s] & (a[s] == g)).sum() >= minimum, f'stratum {v} too small for group {g} on {s}'
            _, infos[v], logps = made_priors(shard, cfg['feature_set'], cfg['made_hidden'], codes, dims, sets, params, device, log,
                                              mask=mask, tag=f"_cond-{cfg['condition_on']}{v}")
            perm_group, perm_all, stat_dist = fnf.rank_matching_maps(logps[ref], logps[mapped], predicted, cfg['gamma'])
            maps[v] = dict(perm_group=perm_group, perm_all=perm_all)
            summary[f'stat_dist_exact_made_stratum{v}'] = stat_dist
            summary[f'train_rows_stratum{v}'] = {g: int((mask['train'] & (a['train'] == g)).sum()) for g in (0, 1)}
        np.savez_compressed(folder / 'mapping.npz', predicted_class=predicted, dims=np.array(dims),
                            **{f'{k}_stratum{v}': m[k] for v, m in maps.items() for k in m})
        summary['combinations'] = total
        summary['predicted_class_share'] = float(predicted.mean())
        summary['condition_on'] = cfg['condition_on']
        summary['yhat_accuracy'] = {s: float((yhat[s] == y[s]).mean()) for s in sets}
        encoded_idx = {}
        for i, s in enumerate(sets):
            rng = np.random.default_rng(seed + 700 + i)
            use_all = rng.random(len(idx[s])) < cfg['gamma']
            out = idx[s].copy()
            for v in (0, 1):
                sel = (a[s] != ref) & (strata[s] == v)
                out[sel] = np.where(use_all[sel], maps[v]['perm_all'][idx[s][sel]], maps[v]['perm_group'][idx[s][sel]])
            encoded_idx[s] = out
            z[s] = fnf.one_hot(fnf.index_to_codes(out, dims), dims)
        for s in ['validation', 'test']:
            counts = {g: np.bincount(encoded_idx[s][a[s] == g], minlength=total) / (a[s] == g).sum() for g in (0, 1)}
            summary[f'empirical_tv_{s}'] = float(.5 * np.abs(counts[0] - counts[1]).sum())
            for v in (0, 1):
                m = y[s] == v
                cv = {g: np.bincount(encoded_idx[s][m & (a[s] == g)], minlength=total) / max((m & (a[s] == g)).sum(), 1) for g in (0, 1)}
                summary[f'empirical_tv_{s}_withinY{v}'] = float(.5 * np.abs(cv[0] - cv[1]).sum())
        clf, history = fnf.train_group_balanced_classifier(z['train'], y['train'], a['train'], params['clf_hidden'], params['clf_epochs'],
                                                           seed, device, params['clf_lr'], params['clf_weight_decay'], 128)
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        torch.save(clf.state_dict(), folder / 'classifier.pt')
        p = {s: fnf.predict_proba(clf, z[s], device) for s in sets}
        summary['prior_info'] = infos
        summary['representation'] = ('one-hot of rank-matched combination, matched within strata of ' + ('the TRUE label (oracle; label required at encoding, including test)'
                                     if cfg['condition_on'] == 'y' else 'the logistic-regression predicted class (deployable; no label needed at test)'))
    else:
        codes, dims = feature['codes'], feature['dims']
        mades, infos, logps = made_priors(shard, cfg['feature_set'], cfg['made_hidden'], codes, dims, sets, params, device, log)
        xt = fnf.one_hot(codes['train'], dims)
        n1 = (a['train'] == REFERENCE_GROUP).sum() / len(a['train'])
        weights = np.where(a['train'] == REFERENCE_GROUP, 1 - n1, n1)
        logreg = LogisticRegression(max_iter=2000).fit(xt, y['train'], sample_weight=weights)
        joblib.dump(logreg, folder / 'mapping_logreg.joblib')
        total = int(np.prod(dims))
        predicted = np.concatenate([logreg.predict(fnf.one_hot(fnf.index_to_codes(np.arange(s, min(s + 200000, total)), dims), dims))
                                    for s in range(0, total, 200000)])
        ref, mapped = REFERENCE_GROUP, 1 - REFERENCE_GROUP
        perm_group, perm_all, stat_dist = fnf.rank_matching_maps(logps[ref], logps[mapped], predicted, cfg['gamma'])
        np.savez_compressed(folder / 'mapping.npz', perm_group=perm_group, perm_all=perm_all, predicted_class=predicted, dims=np.array(dims))
        summary['stat_dist_exact_made'] = stat_dist
        summary['combinations'] = total
        summary['predicted_class_share'] = float(predicted.mean())
        encoded_idx = {}
        for i, s in enumerate(sets):
            z[s], encoded_idx[s] = fnf.encode_categorical(codes[s], a[s], dims, perm_group, perm_all, cfg['gamma'], ref, seed + 700 + i)
        # Empirical TV of the encoded combination distributions (held-out splits).
        for s in ['validation', 'test']:
            counts = {g: np.bincount(encoded_idx[s][a[s] == g], minlength=total) / (a[s] == g).sum() for g in (0, 1)}
            summary[f'empirical_tv_{s}'] = float(.5 * np.abs(counts[0] - counts[1]).sum())
            summary[f'empirical_tv_{s}_input'] = float(.5 * np.abs(
                np.bincount(fnf.combination_index(codes[s], dims)[a[s] == 0], minlength=total) / (a[s] == 0).sum()
                - np.bincount(fnf.combination_index(codes[s], dims)[a[s] == 1], minlength=total) / (a[s] == 1).sum()).sum())
        clf, history = fnf.train_group_balanced_classifier(z['train'], y['train'], a['train'], params['clf_hidden'], params['clf_epochs'],
                                                           seed, device, params['clf_lr'], params['clf_weight_decay'], 128)
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        torch.save(clf.state_dict(), folder / 'classifier.pt')
        p = {s: fnf.predict_proba(clf, z[s], device) for s in sets}
        summary['prior_info'] = infos
        summary['representation'] = 'one-hot of rank-matched category combination (sensitive attribute required at encoding)'
    save_representations(folder, sets, z, p)
    summary.update(adversaries(z, sets, params, seed, device))
    for s in sets:
        yy, pp, aa = y[s], p[s], a[s]
        pred = pp >= .5
        summary[f'{s}_accuracy_05'] = float((pred == yy).mean())
        summary[f'{s}_balanced_group_accuracy_05'] = float(np.mean([(pred[aa == g] == yy[aa == g]).mean() for g in (0, 1)]))
        summary[f'{s}_dp_05'] = float(abs(pred[aa == 0].mean() - pred[aa == 1].mean()))
    summary['elapsed_seconds'] = time.time() - started
    write_json(folder / 'training_summary.json', summary)
    write_json(folder / 'complete.json', dict(splits=list(sets), elapsed_seconds=summary['elapsed_seconds'], smoke=args.smoke))
    log(f'COMPLETE {folder} in {summary["elapsed_seconds"]:.0f}s')


def prepare_features(shard, sets, manifest, args):
    prepared = {}
    params = dict(CATEGORICAL)
    for name, columns in manifest['feature_sets'].items():
        if shard['dataset'] == 'health':
            raw = health_matrices(sets, columns)
            dq = fnf.Dequantizer(CONTINUOUS['alpha']).fit(raw['train'])
            prepared[name] = dict(columns=columns, raw=raw, dequantizer=dq, scaled={s: dq.scale(x) for s, x in raw.items()})
        else:
            codes, dims, info = categorical_codes(sets, columns, params, manifest.get('binned_features'))
            prepared[name] = dict(columns=columns, codes=codes, dims=dims, info=info)
    return prepared


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int)
    p.add_argument('--dataset', choices=DATASETS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--condition', choices=CONDITIONS, default='natural')
    p.add_argument('--only', help='Run only configurations whose name contains this substring')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--conditional', action='store_true', help='Run the class-conditional FNF trial configurations instead of the main sweep')
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    shard = SHARDS[args.index] if args.index is not None else dict(dataset=args.dataset, seed=args.seed, condition=args.condition)
    if shard['dataset'] is None:
        p.error('Provide --index or --dataset')
    shard = dict(shard, smoke=args.smoke) if args.smoke else dict(shard)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.allow_cpu:
        assert device == 'cuda', 'GPU allocation required (use --allow-cpu for smoke tests)'
    d, manifest = load(shard['dataset'])
    sets = source_sets(d, shard['condition'], args.smoke)
    root = BASE / ('smoke' if args.smoke else 'runs') / shard['dataset'] / f"seed{shard['seed']}" / shard['condition']
    root.mkdir(parents=True, exist_ok=True)
    for s, g in sets.items():
        g[[c for c in g if not c.startswith('x_')]].to_csv(root / f'cohort_{s}.csv', index=False)
    prepared = prepare_features(shard, sets, manifest, args)
    write_json(root / 'features.json', {name: dict(columns=v['columns'], dequantizer=v['dequantizer'].state() if 'dequantizer' in v else None,
                                                    categorical=v.get('info')) for name, v in prepared.items()})
    failures = []
    configs = conditional_configurations(shard['dataset']) if args.conditional else configurations(shard['dataset'], list(manifest['feature_sets']))
    for cfg in configs:
        if args.only and args.only not in run_name(cfg):
            continue
        try:
            run_one(shard, cfg, sets, prepared, device, args)
        except Exception as error:  # record and continue; the shard must not lose later configurations
            import traceback
            folder = root / run_name(cfg)
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / 'failed.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                                    slurm_job_id=os.environ.get('SLURM_JOB_ID')))
            failures.append(run_name(cfg))
            print('FAILED', folder, repr(error), flush=True)
    write_json(root / 'shard_summary.json', dict(shard={k: v for k, v in shard.items() if k != 'smoke'}, failures=failures,
                                                completed=len(list(root.glob('*/complete.json')))))
    if failures:
        print('Shard finished with failures:', failures, flush=True)


if __name__ == '__main__':
    main()
