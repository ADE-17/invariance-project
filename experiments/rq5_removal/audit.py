"""Frozen independent probes and task metrics for the focused removal pilot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from experiments.rq5_sweep.probes import fit
from experiments.rq5_sweep.metrics import point_metrics, cluster_weights, interval
from experiments.rq5_sweep.run import cap, write_json
from experiments.rq4_ptbxl.probe_analysis import weighted_aucs
from src.evaluation.task_metrics import select_threshold_on_validation

FAMILIES = ['linear', 'tree', 'mlp0', 'mlp1', 'mlp2']


def finite_probability_threshold(y, probability):
    """Preserve ROC all-negative/all-positive rules using finite score bounds.

    sklearn's ROC curve includes +inf for the all-negative rule. Task scores
    here are bounded probabilities, so the next float32 above 1 preserves that
    decision rule on every valid test probability and remains JSON serializable.
    """
    probability = np.asarray(probability)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    threshold = float(select_threshold_on_validation(y, probability, criterion='youden'))
    if np.isposinf(threshold):
        return float(np.nextafter(np.float32(1.), np.float32(np.inf)))
    if np.isneginf(threshold):
        return float(np.nextafter(np.float32(0.), np.float32(-np.inf)))
    if not np.isfinite(threshold):
        raise ValueError('Invalid validation threshold')
    return threshold


def audit(folder, sets, variants, probabilities, seed, scope, metadata, bootstrap=2000):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'audit_complete.json').exists():
        return
    device = 'cpu'
    tests = [s for s in sets if s.endswith('test')]
    threshold = finite_probability_threshold(sets['validation'].y.to_numpy(), probabilities['validation'])
    pred = {s: sets[s][['row_id', 'unit', 'a', 'y']].copy().reset_index(drop=True) for s in tests}
    for s in tests:
        pred[s]['task_probability'] = probabilities[s].astype('float32')
    train = cap(sets['train'], 30000)
    val = cap(sets['probe_validation'], 10000)
    ti = sets['train'].index.get_indexer(train.index)
    vi = sets['probe_validation'].index.get_indexer(val.index)
    for variant, zz in variants.items():
        for label in [None, 0, 1]:
            tm = np.ones(len(train), bool) if label is None else train.y.to_numpy() == label
            vm = np.ones(len(val), bool) if label is None else val.y.to_numpy() == label
            masks = {s: np.ones(len(sets[s]), bool) if label is None else sets[s].y.to_numpy() == label for s in tests}
            suffix = 'marginal' if label is None else f'Y{label}'
            scores = fit(zz['train'][ti][tm], train.a.to_numpy()[tm], zz['probe_validation'][vi][vm], val.a.to_numpy()[vm],
                         {s: zz[s][masks[s]] for s in tests}, folder/'probes'/variant/suffix, seed+(0 if label is None else 100*(label+1)), device)
            for s in tests:
                for name, p in scores[s].items():
                    column = f'{variant}_{name}_{suffix}'
                    full = np.full(len(sets[s]), np.nan)
                    full[masks[s]] = p
                    pred[s][column] = full
    rows = []
    for s in tests:
        d = sets[s]
        row = dict(**metadata, test=s, seed=seed, threshold=threshold, scope=scope,
                   **point_metrics(d.y.to_numpy(), probabilities[s], d.a.to_numpy(), threshold))
        aa, yy, pp = d.a.to_numpy(), d.y.to_numpy(), probabilities[s]
        row['score_ks'] = float(ks_2samp(pp[aa == 0], pp[aa == 1]).statistic)
        row['score_conditional_ks'] = max(float(ks_2samp(pp[(aa == 0) & (yy == label)],
                                                                        pp[(aa == 1) & (yy == label)]).statistic) for label in [0, 1])
        columns = [c for c in pred[s] if any(f'_{family}_' in c for family in FAMILIES) and 'permuted' not in c]
        draws = {c: [] for c in columns}
        for weights in cluster_weights(d.unit, bootstrap):
            for c in columns:
                m = pred[s][c].notna().to_numpy()
                draws[c].append(weighted_aucs(d.a.to_numpy()[m], pred[s][c].to_numpy()[m], weights[:, m]))
        gates = {}
        for variant in variants:
            gates[variant] = {}
            for which, suffixes in [('marginal', ['marginal']), ('conditional', ['Y0', 'Y1'])]:
                chosen = [f'{variant}_{fam}_{suffix}' for fam in FAMILIES for suffix in suffixes]
                bounds = []
                details = {}
                for c in chosen:
                    ci = interval(np.concatenate(draws[c]), .05/len(chosen))
                    upper = max(ci['high'], 1-ci['low']) if ci['low'] is not None else None
                    bounds.append(upper)
                    m = pred[s][c].notna().to_numpy()
                    details[c] = dict(auc=float(roc_auc_score(d.a.to_numpy()[m], pred[s][c].to_numpy()[m])),
                                      **ci, oriented_upper=upper)
                gates[variant][which] = dict(pass_gate=None not in bounds and max(bounds) <= .55,
                                             max_oriented_upper=max(bounds) if None not in bounds else None,
                                             probe_details=details)
        row['gates'] = gates
        row['release_gate_pass'] = gates['release'][scope]['pass_gate']
        row['pre_noise_gate_pass'] = gates.get('pre_noise', {}).get(scope, {}).get('pass_gate')
        row['independence_proven_by_probes'] = False
        pred[s].to_csv(folder/f'predictions_{s}.csv.gz', index=False, compression='gzip')
        write_json(folder/f'evaluation_{s}.json', row)
        rows.append(row)
    write_json(folder/'audit_complete.json', dict(tests=tests, bootstrap=bootstrap, reports_generated=False))
    print('AUDITED', folder, flush=True)
