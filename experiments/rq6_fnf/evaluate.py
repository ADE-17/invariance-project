"""CPU evaluation of completed RQ6 runs: independent probes, fairness metrics, cluster bootstrap."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rq5_sweep import probes
from experiments.rq5_sweep.metrics import point_metrics, cluster_weights, interval, weighted_task
from experiments.rq5_sweep.run import cap, write_json
from experiments.rq5_removal.audit import finite_probability_threshold, FAMILIES
from experiments.rq4_ptbxl.probe_analysis import weighted_aucs
from experiments.rq6_fnf.data import BASE, CONDITIONS, load, apply_condition
from experiments.rq6_fnf.run import SHARDS, source_sets
from experiments.rq6_fnf.calibration import group_calibration

BOOTSTRAP = 1000
GATE = .55


def test_variants(natural_test):
    return {f'test_{c}': apply_condition(natural_test, c, 'test') for c in CONDITIONS}


def evaluate_run(folder, sets, tests, device, smoke):
    out = folder / 'evaluation'
    out.mkdir(exist_ok=True)
    if (out / 'complete.json').exists():
        return
    started = time.time()
    config = json.loads((folder / 'config.json').read_text())
    summary = json.loads((folder / 'training_summary.json').read_text())
    rep = np.load(folder / 'representations.npz')
    z, p = {}, {}
    for s in ['train', 'validation', 'probe_validation', 'test']:
        assert np.array_equal(rep[f'row_id_{s}'], sets[s].row_id.to_numpy()), f'row mismatch {s}'
        z[s], p[s] = rep[f'z_{s}'], rep[f'task_probability_{s}'].astype('float32')
    test_index = {name: sets['test'].index.get_indexer(t.index) for name, t in tests.items()}
    threshold = finite_probability_threshold(sets['validation'].y.to_numpy(), p['validation'])
    pred = {name: t[['row_id', 'unit', 'a', 'y']].copy().reset_index(drop=True) for name, t in tests.items()}
    for name in tests:
        pred[name]['task_probability'] = p['test'][test_index[name]]
    train = cap(sets['train'], 30000)
    val = cap(sets['probe_validation'], 10000)
    ti = sets['train'].index.get_indexer(train.index)
    vi = sets['probe_validation'].index.get_indexer(val.index)
    probe_audit = {}
    for label in [None, 0, 1]:
        tm = np.ones(len(train), bool) if label is None else train.y.to_numpy() == label
        vm = np.ones(len(val), bool) if label is None else val.y.to_numpy() == label
        masks = {name: np.ones(len(t), bool) if label is None else t.y.to_numpy() == label for name, t in tests.items()}
        suffix = 'marginal' if label is None else f'Y{label}'
        scores = probes.fit(z['train'][ti][tm], train.a.to_numpy()[tm], z['probe_validation'][vi][vm], val.a.to_numpy()[vm],
                            {name: z['test'][test_index[name]][masks[name]] for name in tests}, out / 'probes' / suffix,
                            config['seed'] + (0 if label is None else 100 * (label + 1)), device, smoke)
        probe_audit[suffix] = json.loads((out / 'probes' / suffix / 'audit.json').read_text())
        for name in tests:
            for kind, values in scores[name].items():
                full = np.full(len(tests[name]), np.nan)
                full[masks[name]] = values
                pred[name][f'{kind}_{suffix}'] = full
    rows = []
    for name, t in tests.items():
        aa, yy, pp = t.a.to_numpy(), t.y.to_numpy(), pred[name].task_probability.to_numpy()
        row = dict(dataset=config['dataset'], seed=config['seed'], condition=config['condition'], run=folder.name,
                   method=config['method'], variant=config.get('variant'), feature_set=config['feature_set'],
                   gamma=config.get('gamma'), prior=config.get('prior'), n_blocks=config.get('n_blocks'),
                   made_hidden=config.get('made_hidden'), condition_on=config.get('condition_on'), test=name, test_condition=name.removeprefix('test_'),
                   threshold=threshold, **point_metrics(yy, pp, aa, threshold))
        row.update(group_calibration(yy, pp, aa))  # Cox slope/intercept, overall and per group
        row.update({f'fixed05_{k}': v for k, v in point_metrics(yy, pp, aa, .5).items() if k in ['dp', 'eopp', 'eodds', 'fpr_gap', 'accuracy', 'group0_tpr', 'group1_tpr', 'group0_ppr', 'group1_ppr']})
        row['score_ks'] = float(ks_2samp(pp[aa == 0], pp[aa == 1]).statistic)
        row['score_conditional_ks'] = max(float(ks_2samp(pp[(aa == 0) & (yy == label)], pp[(aa == 1) & (yy == label)]).statistic) for label in (0, 1))
        columns = [c for c in pred[name] if any(c.startswith(f'{fam}_') for fam in FAMILIES + ['permuted_linear'])]
        for c in columns:
            m = pred[name][c].notna().to_numpy()
            row[f'{c}_auc'] = float(roc_auc_score(aa[m], pred[name][c].to_numpy()[m]))
            row[f'{c}_n'] = int(m.sum())
        draws = {c: [] for c in columns}
        task_draws = {}
        for weights in cluster_weights(t.unit, BOOTSTRAP if not smoke else 48):
            for c in columns:
                m = pred[name][c].notna().to_numpy()
                draws[c].append(weighted_aucs(aa[m], pred[name][c].to_numpy()[m], weights[:, m]))
            wt = weighted_task(yy.astype(float), pp.astype(float), aa, threshold, weights)
            for k, v in wt.items():
                task_draws.setdefault(k, []).append(v)
        for k, v in task_draws.items():
            ci = interval(np.concatenate(v))
            row[f'{k}_ci_low'], row[f'{k}_ci_high'] = ci['low'], ci['high']
        gates = {}
        for which, suffixes in [('marginal', ['marginal']), ('conditional', ['Y0', 'Y1'])]:
            chosen = [f'{fam}_{suffix}' for fam in FAMILIES for suffix in suffixes]
            bounds, details = [], {}
            for c in chosen:
                ci = interval(np.concatenate(draws[c]), .05 / len(chosen))
                upper = max(ci['high'], 1 - ci['low']) if ci['low'] is not None else None
                bounds.append(upper)
                details[c] = dict(auc=row[f'{c}_auc'], **ci, oriented_upper=upper)
            gates[which] = dict(pass_gate=None not in bounds and max(bounds) <= GATE, max_oriented_upper=max(bounds) if None not in bounds else None, probe_details=details)
            row[f'{which}_gate_pass'] = gates[which]['pass_gate']
            row[f'{which}_max_oriented_upper'] = gates[which]['max_oriented_upper']
            row[f'{which}_max_probe_auc'] = max(details[c]['auc'] for c in chosen)
        row['gates'] = gates
        row['independence_proven_by_probes'] = False
        for k in ['stat_dist_prior_samples', 'stat_dist_data_validation', 'stat_dist_data_test', 'stat_dist_exact_made',
                  'empirical_tv_validation', 'empirical_tv_test', 'empirical_tv_test_input']:
            if k in summary:
                row[k] = summary[k]
        row.update({k: v for k, v in summary.items() if k.startswith('paper_adversary_')})
        pred[name].to_csv(out / f'predictions_{name}.csv.gz', index=False, compression='gzip')
        curves = [dict(threshold=float(th), **{k: v for k, v in point_metrics(yy, pp, aa, th).items() if k in ['dp', 'eopp', 'eodds', 'accuracy']}) for th in np.linspace(0, 1, 21)]
        pd.DataFrame(curves).to_csv(out / f'threshold_curve_{name}.csv', index=False)
        rows.append(row)
    write_json(out / 'probe_audit.json', probe_audit)
    write_json(out / 'metrics.json', rows)
    write_json(out / 'complete.json', dict(tests=list(tests), bootstrap=BOOTSTRAP if not smoke else 48, gate=GATE,
                                          elapsed_seconds=time.time() - started, reports_generated=False))
    print('EVALUATED', folder, f'{time.time() - started:.0f}s', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int)
    p.add_argument('--dataset')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--condition', default='natural')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    shard = SHARDS[args.index] if args.index is not None else dict(dataset=args.dataset, seed=args.seed, condition=args.condition)
    d, _ = load(shard['dataset'])
    sets = source_sets(d, shard['condition'], args.smoke)
    tests = test_variants(sets['test'])
    root = BASE / ('smoke' if args.smoke else 'runs') / shard['dataset'] / f"seed{shard['seed']}" / shard['condition']
    folders = sorted(f.parent for f in root.glob('*/complete.json'))
    print(f'{len(folders)} completed runs in {root}', flush=True)
    for folder in folders:
        evaluate_run(folder, sets, tests, 'cpu', args.smoke)


if __name__ == '__main__':
    main()
