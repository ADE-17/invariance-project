"""Neutral group labels and bounded-memory paired cluster uncertainty."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from experiments.rq4_ptbxl.probe_analysis import weighted_aucs


def ece(y, p, bins=15):
    ids = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    return float(sum(np.mean(ids == b) * abs(np.mean(y[ids == b]) - np.mean(p[ids == b]))
                     for b in np.unique(ids)))


def point_metrics(y, p, a, threshold):
    y, p, a = np.asarray(y), np.asarray(p), np.asarray(a)
    pred = p >= threshold
    result = dict(n=len(y), positive=int(y.sum()), prevalence=float(y.mean()),
                  auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                  auprc=float(average_precision_score(y, p)) if y.sum() else None,
                  brier=float(brier_score_loss(y, p)), accuracy=float(np.mean(pred == y)),
                  ece=ece(y, p), ece10=ece(y, p, 10), ece20=ece(y, p, 20))
    for g in (0, 1):
        m = a == g
        yg, pg, qg = y[m], p[m], pred[m]
        result.update({f'group{g}_n': int(m.sum()), f'group{g}_positive': int(yg.sum()),
                       f'group{g}_prevalence': float(yg.mean()),
                       f'group{g}_auroc': float(roc_auc_score(yg, pg)) if len(np.unique(yg)) == 2 else None,
                       f'group{g}_ece': ece(yg, pg), f'group{g}_ece10': ece(yg, pg, 10),
                       f'group{g}_ece20': ece(yg, pg, 20),
                       f'group{g}_accuracy': float(np.mean(qg == yg)),
                       f'group{g}_tpr': float(qg[yg == 1].mean()) if (yg == 1).any() else None,
                       f'group{g}_fpr': float(qg[yg == 0].mean()) if (yg == 0).any() else None,
                       f'group{g}_ppr': float(qg.mean())})
    for key, rate in [('dp', 'ppr'), ('eopp', 'tpr'), ('fpr_gap', 'fpr')]:
        values = [result[f'group{g}_{rate}'] for g in (0, 1)]
        result[key] = abs(values[0] - values[1]) if None not in values else None
    result['eodds'] = max(result['eopp'], result['fpr_gap']) if result['eopp'] is not None and result['fpr_gap'] is not None else None
    values = [result[f'group{g}_auroc'] for g in (0, 1)]
    result['worst_group_auroc'] = min(values) if None not in values else None
    result['max_group_ece'] = max(result['group0_ece'], result['group1_ece'])
    return result


def cluster_weights(units, replicates, seed=20260908, batch=24):
    """Yield <=batch bootstrap weight rows; memory independent of total B."""
    unique, inv = np.unique(np.asarray(units).astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch):
        count = min(batch, replicates - start)
        weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=count)
        yield weights[:, inv].astype('float64')


def weighted_ece(y, p, weights, bins=15):
    ids = np.minimum((p * bins).astype(int), bins - 1)
    numerator = np.zeros(len(weights))
    for b in np.unique(ids):
        m = ids == b
        numerator += np.abs(weights[:, m] @ (y[m] - p[m]))
    return numerator / weights.sum(axis=1)


def weighted_task(y, p, a, threshold, weights):
    pred = (p >= threshold).astype(float)
    out = {'auroc': weighted_aucs(y, p, weights),
           'brier': weights @ ((y-p)**2) / weights.sum(axis=1)}
    for g in (0, 1):
        m = a == g
        for rate, m2 in [('ppr', m), ('tpr', m & (y == 1)), ('fpr', m & (y == 0))]:
            den = weights[:, m2].sum(axis=1)
            out[f'{rate}{g}'] = np.divide(weights[:, m2] @ pred[m2], den,
                                         out=np.full(len(weights), np.nan), where=den > 0)
        out[f'auroc{g}'] = weighted_aucs(y[m], p[m], weights[:, m])
        out[f'ece{g}'] = weighted_ece(y[m], p[m], weights[:, m])
    out['dp'] = abs(out.pop('ppr0') - out.pop('ppr1'))
    out['eopp'] = abs(out.pop('tpr0') - out.pop('tpr1'))
    out['eodds'] = np.maximum(out['eopp'], abs(out.pop('fpr0') - out.pop('fpr1')))
    out['worst_group_auroc'] = np.minimum(out.pop('auroc0'), out.pop('auroc1'))
    out['max_group_ece'] = np.maximum(out.pop('ece0'), out.pop('ece1'))
    return out


def interval(values, alpha=.05):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    if len(valid) < .95 * len(values):
        return {'low': None, 'high': None, 'valid_replicates': len(valid)}
    low, high = np.quantile(valid, [alpha/2, 1-alpha/2])
    return {'low': float(low), 'high': float(high), 'valid_replicates': len(valid)}
