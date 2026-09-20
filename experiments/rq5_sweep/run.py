"""One Slurm array shard: dataset x seed x training condition, nine model fits."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rq5_sweep.data import BASE, DATASETS, sha, validate
from experiments.rq5_sweep.metrics import point_metrics
from experiments.rq5_sweep import probes
from experiments.rq4_ptbxl.trial import Encoder, balanced_bce, mlp
from src.models.gradient_reversal import GradientReversalFunction
from src.evaluation.task_metrics import select_threshold_on_validation

SEEDS = [42, 123, 456, 789, 1024]
LAMBDAS = [.1, 1., 5., 20.]
SHARDS = [dict(dataset=d, seed=s, condition=c) for d in DATASETS for s in SEEDS for c in ['natural', 'shifted']]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def shifted(d):
    return d.drop(d[(d.a == 0) & (d.y == 1)].sample(frac=.5, random_state=42).index)


def cap(d, n, seed=42):
    if len(d) <= n:
        return d
    # Stratified fixed cap for probe compute; does not change source cohort splits.
    chunks = []
    for _, g in d.groupby(['a', 'y']):
        chunks.append(g.sample(max(2, int(n * len(g) / len(d))), random_state=seed))
    return pd.concat(chunks).sort_index()


def config_hash():
    from experiments.rq4_ptbxl import trial
    from src.models import gradient_reversal
    from src.evaluation import task_metrics, calibration
    paths = sorted(Path(__file__).parent.glob('*.py'))
    paths += [Path(m.__file__) for m in [trial, gradient_reversal, task_metrics, calibration]]
    return {str(p): sha(p) for p in paths}


def source_data(name, condition, smoke):
    folder = BASE / 'prepared' / name
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['cohort_sha256'] == sha(folder / 'cohort.csv')
    d = pd.read_csv(folder / 'cohort.csv', dtype={'unit': str, **{c: str for c in manifest['categorical_features']}})
    validate(d)
    sets = {s: g.copy() for s, g in d.groupby('split')}
    if smoke:
        sets = {s: cap(g, 600 if s.endswith('test') else (1200 if s == 'train' else 400)) for s, g in sets.items()}
    if condition == 'shifted':
        sets['train'] = shifted(sets['train'])
    sets['prevalence_test'] = shifted(sets['test'])
    return sets, manifest


def transform(sets, manifest, folder):
    train = sets['train']
    if manifest['signal_path']:
        all_x = np.load(manifest['signal_path'], mmap_mode='r')
        ids = train.row_index.to_numpy()
        count = 0
        sums = np.zeros(12)
        squares = np.zeros(12)
        for start in range(0, len(ids), 256):
            chunk = np.asarray(all_x[ids[start:start+256]], dtype='float64')
            count += chunk.shape[0] * chunk.shape[2]
            sums += chunk.sum(axis=(0, 2))
            squares += (chunk**2).sum(axis=(0, 2))
        mean = (sums / count).astype('float32')[None, :, None]
        std = np.sqrt(np.maximum(squares / count - (sums / count)**2, 1e-12)).astype('float32')[None, :, None]
        np.savez(folder / 'normalization.npz', mean=mean, std=std)
        return {s: (np.asarray(all_x[g.row_index]).copy() - mean) / std for s, g in sets.items()}
    cols = manifest['features']
    num = train[cols].select_dtypes(include='number').columns.tolist()
    cat = [c for c in cols if c not in num]
    pp = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
        ('categorical', make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'),
                                      OneHotEncoder(handle_unknown='ignore', max_categories=128, sparse_output=False)), cat)])
    pp.fit(train[cols])
    joblib.dump(pp, folder / 'preprocessor.joblib')
    return {s: pp.transform(g[cols]).astype('float32') for s, g in sets.items()}


def loader(x, d, seed, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.tensor(d.y.to_numpy(), dtype=torch.float32),
                                   torch.tensor(d.a.to_numpy(), dtype=torch.long)), batch_size=256,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def extract(enc, head, x, device):
    enc.eval()
    head.eval()
    zs, ps = [], []
    for i in range(0, len(x), 512):
        z = enc(torch.from_numpy(x[i:i+512]).to(device))
        zs.append(z.cpu().numpy())
        ps.append(torch.sigmoid(head(z).squeeze(-1)).cpu().numpy())
    return np.concatenate(zs), np.concatenate(ps)


def fit_task(folder, method, lam, seed, sets, xs, signal, epochs, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    enc = Encoder() if signal else nn.Sequential(nn.Linear(xs['train'].shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
    enc = enc.to(device)
    head = nn.Linear(128, 1).to(device)
    adv = nn.ModuleList([mlp() for _ in range(2 if method == 'cdann' else 1)]).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=.001, weight_decay=.0001)
    aopt = torch.optim.Adam(adv.parameters(), lr=.001, weight_decay=.0001)
    if (folder / 'training_complete.json').exists():
        return enc, head, adv
    dl = loader(xs['train'], sets['train'], seed, True)
    best = float('inf')
    history = []
    started = time.time()
    for epoch in range(1, epochs+1):
        enc.train(); head.train(); adv.train()
        alpha = 0. if method == 'erm' else lam * min((epoch-1)/5, 1)
        total, atotal, skipped, batches = 0., 0., 0, 0
        for x, y, a in dl:
            x, y, a = x.to(device), y.to(device), a.to(device)
            z = enc(x)
            task = nn.functional.binary_cross_entropy_with_logits(head(z).squeeze(-1), y)
            aloss = torch.zeros((), device=device)
            if method != 'erm':
                rev = GradientReversalFunction.apply(z, alpha)
                losses = []
                for cls, model in enumerate(adv):
                    mask = torch.ones_like(y, dtype=torch.bool) if method == 'dann' else y == cls
                    if a[mask].unique().numel() == 2:
                        losses.append(balanced_bce(model(rev[mask]), a[mask]))
                    else:
                        skipped += 1
                if losses:
                    aloss = sum(losses) / len(losses)
            opt.zero_grad(); aopt.zero_grad()
            (task + aloss).backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 5)
            opt.step()
            if method != 'erm': aopt.step()
            total += float(task.detach()) * len(y)
            atotal += float(aloss.detach()) * len(y)
            batches += 1
        _, p = extract(enc, head, xs['validation'], device)
        yy = sets['validation'].y.to_numpy()
        pclip = np.clip(p.astype('float64'), 1e-7, 1-1e-7)
        vl = float(-np.mean(yy*np.log(pclip) + (1-yy)*np.log(1-pclip)))
        assert np.isfinite(vl)
        state = dict(encoder=enc.state_dict(), head=head.state_dict(), adversary=adv.state_dict(), epoch=epoch, alpha=alpha)
        if epoch >= 6 and vl < best:
            best = vl
            torch.save(state, folder / 'best.pt')
        torch.save(state, folder / 'last.pt')
        history.append(dict(epoch=epoch, alpha=alpha, train_bce=total/len(sets['train']),
                            adversary_bce=atotal/len(sets['train']), validation_bce=vl,
                            validation_auroc=float(roc_auc_score(yy, p)), skipped_adversary_cells=skipped,
                            batches=batches, elapsed_seconds=time.time()-started))
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        print(folder.name, 'epoch', epoch, 'validation_bce', round(vl, 5), flush=True)
    write_json(folder / 'training_complete.json', dict(epochs=epochs, elapsed_seconds=time.time()-started))
    return enc, head, adv


def audit_checkpoint(folder, checkpoint, enc, head, adv, sets, xs, config, device, smoke):
    out = folder / checkpoint
    out.mkdir(exist_ok=True)
    if (out / 'complete.json').exists():
        return
    saved = torch.load(folder / f'{checkpoint}.pt', map_location=device, weights_only=True)
    enc.load_state_dict(saved['encoder']); head.load_state_dict(saved['head']); adv.load_state_dict(saved['adversary']); adv.eval()
    before = {k: t.detach().cpu().clone() for k, t in enc.state_dict().items()}
    features = {s: extract(enc, head, x, device) for s, x in xs.items()}
    threshold = float(select_threshold_on_validation(sets['validation'].y.to_numpy(), features['validation'][1], criterion='youden'))
    write_json(out / 'checkpoint.json', dict(epoch=saved['epoch'], threshold=threshold, checkpoint_sha256=sha(folder / f'{checkpoint}.pt')))
    test_names = [s for s in sets if s.endswith('test')]
    predictions = {}
    for s in test_names:
        d = sets[s]
        pred = d[[c for c in ['row_id', 'unit', 'a', 'y', 'age', 'ecg_id'] if c in d]].copy()
        pred['task_probability'] = features[s][1]
        pred['constant_probe'] = .5
        pred['oracle_sensitive_control'] = d.a.to_numpy()
        predictions[s] = pred
    # Source probes evaluated everywhere; fresh target probes use disjoint target
    # audit train/val only. Their data NEVER updates task encoder/head/threshold.
    domains = [('source', 'train', 'probe_validation', test_names)]
    domains += [(prefix, prefix+'_probe_train', prefix+'_probe_validation', [prefix+'_test'])
                for prefix in ['geographic', 'temporal'] if prefix+'_test' in sets]
    for domain, tr_name, val_name, names in domains:
        tr = cap(sets[tr_name], 30000)
        val = cap(sets[val_name], 10000)
        tri = sets[tr_name].index.get_indexer(tr.index)
        vi = sets[val_name].index.get_indexer(val.index)
        ztr, zv = features[tr_name][0][tri], features[val_name][0][vi]
        # Task-score-only and true-label-only baselines quantify leakage explained
        # by the output/label; Y is used for audit only, not task inference.
        score_probe = LogisticRegression(class_weight='balanced', max_iter=1000).fit(features[tr_name][1][tri, None], tr.a)
        joblib.dump(score_probe, out / f'{domain}_score_only.joblib')
        label_rates = tr.groupby('y').a.mean().to_dict()
        for s in names:
            predictions[s][f'{domain}_score_only'] = score_probe.predict_proba(features[s][1][:, None])[:, 1]
            predictions[s][f'{domain}_label_only'] = sets[s].y.map(label_rates).to_numpy()
        for cls in [None, 0, 1]:
            tm = np.ones(len(tr), bool) if cls is None else tr.y.to_numpy() == cls
            vm = np.ones(len(val), bool) if cls is None else val.y.to_numpy() == cls
            masks = {s: np.ones(len(sets[s]), bool) if cls is None else sets[s].y.to_numpy() == cls for s in names}
            suffix = 'marginal' if cls is None else f'Y{cls}'
            scores = probes.fit(ztr[tm], tr.a.to_numpy()[tm], zv[vm], val.a.to_numpy()[vm],
                                {s: features[s][0][masks[s]] for s in names}, out / 'probes' / domain / suffix,
                                config['seed'] + (0 if cls is None else 100*(cls+1)), device, smoke)
            for s in names:
                for kind, p in scores[s].items():
                    col = f'{domain}_{kind}_{suffix}'
                    full = np.full(len(sets[s]), np.nan)
                    full[masks[s]] = p
                    predictions[s][col] = full
        pd.concat([tr.assign(audit_role='train'), val.assign(audit_role='validation')])[['row_id', 'unit', 'a', 'y', 'audit_role']].to_csv(out / f'{domain}_probe_cohort.csv', index=False)
    # The online adversary is diagnostic; independence is assessed by refit probes.
    if config['method'] != 'erm':
        with torch.no_grad():
            for s in test_names:
                z = torch.from_numpy(features[s][0]).to(device)
                for cls, model in enumerate(adv):
                    predictions[s][f'online_adversary_{cls}'] = torch.sigmoid(model(z)).cpu().numpy()
    metrics = []
    for s, pred in predictions.items():
        d = sets[s]
        row = dict(dataset=config['dataset'], seed=config['seed'], method=config['method'], lam=config['lam'],
                   condition=config['condition'], checkpoint=checkpoint, epoch=saved['epoch'], test=s,
                   threshold=threshold, feature_mean_std=float(features[s][0].std(axis=0).mean()),
                   **point_metrics(d.y.to_numpy(), features[s][1], d.a.to_numpy(), threshold))
        row.update({f'fixed05_{k}': v for k, v in point_metrics(d.y.to_numpy(), features[s][1], d.a.to_numpy(), .5).items() if k in ['dp', 'eopp', 'eodds', 'accuracy']})
        for c in pred:
            if c in ['row_id', 'unit', 'a', 'y', 'age', 'ecg_id', 'task_probability']: continue
            mask = pred[c].notna().to_numpy()
            row[c+'_auc'] = float(roc_auc_score(d.a.to_numpy()[mask], pred[c].to_numpy()[mask]))
            row[c+'_n'] = int(mask.sum())
        pred.to_csv(out / f'predictions_{s}.csv.gz', index=False, compression='gzip')
        metrics.append(row)
        curves = [dict(threshold=float(t), **{k: v for k, v in point_metrics(d.y.to_numpy(), features[s][1], d.a.to_numpy(), t).items() if k in ['dp', 'eopp', 'eodds', 'accuracy']}) for t in np.linspace(0, 1, 21)]
        pd.DataFrame(curves).to_csv(out / f'threshold_curve_{s}.csv', index=False)
        if 'age' in d:
            groups = pd.cut(d.age, [-np.inf, 40, 60, 80, 90, np.inf], right=False).astype(str)
            strata = []
            for (age, group), ids in d.assign(age_group=groups).groupby(['age_group', 'a']).groups.items():
                idx = d.index.get_indexer(ids)
                y, p = d.y.to_numpy()[idx], features[s][1][idx]
                strata.append(dict(age_group=age, group=int(group), n=len(idx), positive=int(y.sum()),
                                   auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                                   ppr=float((p >= threshold).mean()),
                                   tpr=float((p[y == 1] >= threshold).mean()) if (y == 1).any() else None,
                                   fpr=float((p[y == 0] >= threshold).mean()) if (y == 0).any() else None))
            pd.DataFrame(strata).to_csv(out / f'age_intersection_{s}.csv', index=False)
    assert all(torch.equal(before[k], t.detach().cpu()) for k, t in enc.state_dict().items()), 'Probe training modified encoder'
    write_json(out / 'metrics.json', metrics)
    write_json(out / 'complete.json', dict(tests=test_names, probe_does_not_modify_encoder=True, smoke=smoke))


def run_model(shard, method, lam, args):
    dataset, seed, condition = shard['dataset'], shard['seed'], shard['condition']
    folder = BASE / ('smoke' if args.smoke else 'runs') / dataset / f'seed{seed}' / condition / f'{method}_lambda{lam:g}'
    folder.mkdir(parents=True, exist_ok=True)
    sets, manifest = source_data(dataset, condition, args.smoke)
    config = dict(**shard, method=method, lam=lam, epochs=8 if args.smoke else 30, smoke=args.smoke,
                  input_sha256=manifest['cohort_sha256'], source_hashes=config_hash(),
                  probe_train_cap=30000, probe_validation_cap=10000, probe_restarts=3,
                  probe_epochs=60, group_mapping=manifest['mapping'])
    path = folder / 'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config, f'Configuration mismatch: {folder}'
    write_json(path, config)
    if (folder / 'complete.json').exists():
        print('Verified existing run', folder, flush=True)
        return
    for s, d in sets.items():
        d[[c for c in d if not c.startswith('x_')]].to_csv(folder / f'cohort_{s}.csv', index=False)
    xs = transform(sets, manifest, folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.allow_cpu: assert device == 'cuda', 'GPU allocation required'
    write_json(folder / 'runtime.json', dict(device=device, torch_version=str(torch.__version__),
                                            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                                            slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    enc, head, adv = fit_task(folder, method, lam, seed, sets, xs, bool(manifest['signal_path']), config['epochs'], device)
    for checkpoint in ['best', 'last']:
        audit_checkpoint(folder, checkpoint, enc, head, adv, sets, xs, config, device, args.smoke)
    write_json(folder / 'complete.json', dict(checkpoints=['best', 'last'], smoke=args.smoke))
    print('COMPLETE', folder, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int)
    p.add_argument('--dataset', choices=DATASETS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--condition', choices=['natural', 'shifted'], default='natural')
    p.add_argument('--method', choices=['erm', 'dann', 'cdann'])
    p.add_argument('--lambda-adv', type=float, default=1.)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    shard = SHARDS[args.index] if args.index is not None else dict(dataset=args.dataset, seed=args.seed, condition=args.condition)
    if shard['dataset'] is None: p.error('Provide --index or --dataset')
    methods = [('erm', 0.)] + [(m, l) for m in ['dann', 'cdann'] for l in LAMBDAS]
    if args.method: methods = [(args.method, 0. if args.method == 'erm' else args.lambda_adv)]
    for method, lam in methods:
        run_model(shard, method, lam, args)


if __name__ == '__main__':
    main()
