"""Paper-level report for RQ6: Adult + the four additional categorical-FNF datasets that reach near-independence.

Produces results/rq6/paper/{tables/*.csv, figures/*.png} and results/rq6/PAPER_REPORT.md.

Selection rule (pre-specified, independence-based, never EO-based): for each dataset the FNF configuration at
gamma = 1 whose (feature set, MADE width) has the lowest mean marginal probe AUROC over seeds and conditions on the
matched test set. The ERM baseline uses the same feature set. Bootstraps are paired row bootstraps (B = 1000) on the
saved test predictions of the ERM and FNF fits of the same seed and condition, so deltas carry their own uncertainty.
"""
from pathlib import Path
import json
import gzip
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results' / 'rq6'
OUT = RES / 'paper'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
BASE = Path('/path/to/results/rq6/fnf_v1')
DATASETS = ['adult', 'dutch', 'crime', 'credit', 'compas']
COND = ['equalized', 'natural', 'shift50', 'shift75']
CL = {'equalized': 'equalized', 'natural': 'natural', 'shift50': 'shift 50', 'shift75': 'shift 75'}
SEEDS = [42, 123, 456]
B = 1000
TITLES = {'adult': 'Adult income', 'dutch': 'Dutch census', 'crime': 'Communities & Crime', 'credit': 'Taiwan credit default', 'compas': 'COMPAS recidivism'}
C = dict(g0='#2a78d6', g1='#eb6834', dp='#2a78d6', eo='#eb6834', probe='#2a78d6', cond='#eb6834', adv='#1baf7a', erm='#6b6b66', ink='#222', fnf='#4a3aa7')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e2',
                     'grid.linewidth': .6, 'axes.edgecolor': '#b5b5b0', 'figure.facecolor': 'white', 'legend.frameon': False})
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=15):
    ids = np.minimum((p * bins).astype(int), bins - 1)
    out = 0.
    for b in np.unique(ids):
        m = ids == b
        out += m.mean() * abs(y[m].mean() - p[m].mean())
    return out


def auroc(y, p):
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rates(y, p, a, thr):
    pred = p >= thr
    out = {}
    for g in (0, 1):
        m = a == g
        out[f'ppr{g}'] = pred[m].mean()
        out[f'tpr{g}'] = pred[m & (y == 1)].mean() if (m & (y == 1)).any() else np.nan
        out[f'fpr{g}'] = pred[m & (y == 0)].mean() if (m & (y == 0)).any() else np.nan
        out[f'ece{g}'] = ece(y[m], p[m])
    out['dp'] = abs(out['ppr0'] - out['ppr1'])
    out['eopp'] = abs(out['tpr0'] - out['tpr1'])
    out['eodds'] = max(out['eopp'], abs(out['fpr0'] - out['fpr1']))
    out['ece'] = ece(y, p)
    out['max_group_ece'] = max(out['ece0'], out['ece1'])
    out['auroc'] = auroc(y, p)
    return out


METRICS = ['dp', 'eopp', 'eodds', 'ece', 'ece0', 'ece1', 'max_group_ece', 'auroc', 'tpr0', 'tpr1', 'fpr0', 'fpr1', 'ppr0', 'ppr1']


# ----------------------------------------------------------------------------- data access
def load_metrics():
    df = pd.read_csv(RES / 'evaluation_metrics.csv', low_memory=False)
    df = df[df.dataset.isin(DATASETS)].copy()
    df['is_erm'] = df.method.eq('erm')
    return df


def matched(df):
    return df[df.test_condition == df.condition]


def select_configs(df):
    """Independence-based selection: gamma=1 configuration with the lowest mean marginal probe AUROC."""
    rows = []
    m = matched(df)
    for ds in DATASETS:
        f = m[(m.dataset == ds) & (~m.is_erm) & (m.gamma == 1.0)]
        g = f.groupby(['feature_set', 'made_hidden']).agg(probe=('marginal_max_probe_auc', 'mean'), auroc=('auroc', 'mean'), n=('run', 'size')).reset_index()
        best = g.sort_values('probe').iloc[0]
        run = f[(f.feature_set == best.feature_set) & (f.made_hidden == best.made_hidden)].run.iloc[0]
        rows.append(dict(dataset=ds, feature_set=best.feature_set, made_hidden=int(best.made_hidden), gamma=1.0, fnf_run=run,
                         erm_run=f'erm_{best.feature_set}', mean_probe_auc=round(best.probe, 4), mean_task_auroc=round(best.auroc, 4),
                         candidates=len(g), candidate_table=g.round(4).to_dict('records')))
    sel = pd.DataFrame(rows)
    sel.drop(columns='candidate_table').to_csv(TAB / 'selected_configurations.csv', index=False)
    (TAB / 'selection_candidates.json').write_text(json.dumps({r['dataset']: r['candidate_table'] for r in rows}, indent=1))
    return sel


def predictions(ds, seed, cond, run):
    folder = BASE / 'runs' / ds / f'seed{seed}' / cond / run / 'evaluation'
    with gzip.open(folder / f'predictions_test_{cond}.csv.gz', 'rt') as f:
        p = pd.read_csv(f, usecols=['row_id', 'a', 'y', 'task_probability'])
    met = [r for r in json.load(open(folder / 'metrics.json')) if r['test'] == f'test_{cond}'][0]
    return p.sort_values('row_id').reset_index(drop=True), met


# ----------------------------------------------------------------------------- paired bootstrap
def paired_bootstrap(sel):
    rows, dists = [], {}
    rng_master = np.random.default_rng(20260915)
    for _, s in sel.iterrows():
        for cond in COND:
            for seed in SEEDS:
                pe, me = predictions(s.dataset, seed, cond, s.erm_run)
                pf, mf = predictions(s.dataset, seed, cond, s.fnf_run)
                assert (pe.row_id.values == pf.row_id.values).all(), 'prediction rows differ between ERM and FNF'
                y, a = pe.y.to_numpy(), pe.a.to_numpy()
                pe_p, pf_p = pe.task_probability.to_numpy(), pf.task_probability.to_numpy()
                point = dict(erm=rates(y, pe_p, a, me['threshold']), fnf=rates(y, pf_p, a, mf['threshold']))
                n = len(y)
                rng = np.random.default_rng(rng_master.integers(1 << 31))
                boots = {k: {m: np.empty(B) for m in METRICS} for k in ('erm', 'fnf')}
                for b in range(B):
                    idx = rng.integers(0, n, n)
                    yb, ab = y[idx], a[idx]
                    for k, pp, thr in (('erm', pe_p, me['threshold']), ('fnf', pf_p, mf['threshold'])):
                        r = rates(yb, pp[idx], ab, thr)
                        for m in METRICS:
                            boots[k][m][b] = r[m]
                for k in ('erm', 'fnf'):
                    row = dict(dataset=s.dataset, condition=cond, seed=seed, model=k, run=s.erm_run if k == 'erm' else s.fnf_run,
                               threshold=me['threshold'] if k == 'erm' else mf['threshold'], n=n,
                               probe_auc=(me if k == 'erm' else mf)['marginal_max_probe_auc'],
                               cond_probe_auc=(me if k == 'erm' else mf)['conditional_max_probe_auc'])
                    for m in METRICS:
                        row[m] = point[k][m]; row[f'{m}_boot_mean'] = boots[k][m].mean(); row[f'{m}_boot_sd'] = boots[k][m].std(ddof=1)
                    rows.append(row)
                row = dict(dataset=s.dataset, condition=cond, seed=seed, model='delta', run=f'{s.fnf_run} - {s.erm_run}', threshold=np.nan, n=n,
                           probe_auc=mf['marginal_max_probe_auc'] - me['marginal_max_probe_auc'],
                           cond_probe_auc=mf['conditional_max_probe_auc'] - me['conditional_max_probe_auc'])
                for m in METRICS:
                    d = boots['fnf'][m] - boots['erm'][m]
                    row[m] = point['fnf'][m] - point['erm'][m]; row[f'{m}_boot_mean'] = d.mean(); row[f'{m}_boot_sd'] = d.std(ddof=1)
                    row[f'{m}_ci_low'], row[f'{m}_ci_high'] = np.quantile(d, [.025, .975])
                rows.append(row)
                dists[(s.dataset, cond, seed)] = dict(y=y, a=a, erm=pe_p, fnf=pf_p, thr_erm=me['threshold'], thr_fnf=mf['threshold'])
                print('bootstrapped', s.dataset, cond, seed, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / 'bootstrap_selected_per_seed.csv', index=False)
    return out, dists


def pooled(boot):
    """Across-seed summary: mean of per-seed point estimates, sd across seeds, and mean bootstrap sd."""
    cols = METRICS + ['probe_auc', 'cond_probe_auc']
    g = boot.groupby(['dataset', 'condition', 'model'])
    out = g[cols].mean().add_suffix('_mean').join(g[cols].std(ddof=1).add_suffix('_seed_sd'))
    out = out.join(g[[f'{m}_boot_sd' for m in METRICS]].mean())
    out = out.reset_index()
    out.to_csv(TAB / 'bootstrap_selected_pooled.csv', index=False)
    return out


# ----------------------------------------------------------------------------- tables (markdown)
def fmt(m, sd):
    return f'{m:.3f} ± {sd:.3f}'


def main_table(pool, ds):
    lines = ['| condition | model | probe AUC (A) | DP gap | ΔDP | EO gap | ΔEO | EOpp gap | ECE all | ECE A=0 | ECE A=1 | task AUROC |', '|' + '---|' * 12]
    for cond in COND:
        sub = pool[(pool.dataset == ds) & (pool.condition == cond)].set_index('model')
        e, f, d = sub.loc['erm'], sub.loc['fnf'], sub.loc['delta']
        lines.append(f"| {CL[cond]} | ERM | {e.probe_auc_mean:.3f} | {fmt(e.dp_mean, e.dp_seed_sd)} | | {fmt(e.eodds_mean, e.eodds_seed_sd)} | | {fmt(e.eopp_mean, e.eopp_seed_sd)} | {e.ece_mean:.3f} | {e.ece0_mean:.3f} | {e.ece1_mean:.3f} | {fmt(e.auroc_mean, e.auroc_seed_sd)} |")
        lines.append(f"| {CL[cond]} | FNF γ=1 | {f.probe_auc_mean:.3f} | {fmt(f.dp_mean, f.dp_seed_sd)} | {fmt(d.dp_mean, d.dp_seed_sd)} | {fmt(f.eodds_mean, f.eodds_seed_sd)} | {fmt(d.eodds_mean, d.eodds_seed_sd)} | {fmt(f.eopp_mean, f.eopp_seed_sd)} | {f.ece_mean:.3f} | {f.ece0_mean:.3f} | {f.ece1_mean:.3f} | {fmt(f.auroc_mean, f.auroc_seed_sd)} |")
    return '\n'.join(lines)


def seed_table(boot, ds):
    lines = ['| condition | seed | model | DP | EO | EOpp | ECE all | ECE A=0 | ECE A=1 | TPR A=0 / A=1 | FPR A=0 / A=1 | AUROC |', '|' + '---|' * 12]
    for cond in COND:
        for seed in SEEDS:
            sub = boot[(boot.dataset == ds) & (boot.condition == cond) & (boot.seed == seed)].set_index('model')
            for k, name in (('erm', 'ERM'), ('fnf', 'FNF γ=1'), ('delta', 'Δ (FNF−ERM)')):
                r = sub.loc[k]
                cell = lambda m: f'{r[m]:.3f} ± {r[f"{m}_boot_sd"]:.3f}'
                lines.append(f"| {CL[cond]} | {seed} | {name} | {cell('dp')} | {cell('eodds')} | {cell('eopp')} | {cell('ece')} | {cell('ece0')} | {cell('ece1')} | "
                             f"{r.tpr0:.3f} / {r.tpr1:.3f} | {r.fpr0:.3f} / {r.fpr1:.3f} | {cell('auroc')} |")
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- figures
def fig_sweep(df, sel, ds):
    s = sel[sel.dataset == ds].iloc[0]
    m = matched(df)
    m = m[(m.dataset == ds) & (m.feature_set == s.feature_set) & (m.is_erm | (m.made_hidden == s.made_hidden))]
    gammas = sorted(m.loc[~m.is_erm, 'gamma'].unique())
    xp = {g: i for i, g in enumerate(gammas)}
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 8.6), sharex=True, sharey='row')
    rows = [('decodability of A (AUROC)', [('marginal_max_probe_auc', 'strongest probe, marginal', C['probe']), ('conditional_max_probe_auc', 'strongest probe, within label', C['cond'])], (.45, 1.0)),
            ('fairness gap (Youden threshold)', [('dp', 'demographic parity gap', C['dp']), ('eodds', 'equalized odds gap', C['eo'])], (0, None)),
            ('calibration / utility', [('max_group_ece', 'worst-group ECE', C['eo']), ('ece', 'overall ECE', C['probe']), ('auroc', 'task AUROC', C['erm'])], (0, 1))]
    for r, (ylabel, series, ylim) in enumerate(rows):
        for c, cond in enumerate(COND):
            ax = axes[r, c]
            q = m[m.condition == cond]
            for col, label, color in series:
                f = q[~q.is_erm]
                mean = f.groupby('gamma')[col].mean()
                ax.plot([xp[g] for g in mean.index], mean.values, color=color, lw=1.8, label=label, zorder=3)
                ax.scatter([xp[g] for g in f.gamma], f[col], s=14, color=color, alpha=.65, lw=0, zorder=4)
                ax.axhline(q[q.is_erm][col].mean(), color=color, lw=1, ls=(0, (3, 2)), alpha=.9)
                ax.scatter([-.6] * q.is_erm.sum(), q[q.is_erm][col], s=18, marker='s', color=color, alpha=.8, lw=0, zorder=4)
            if r == 0:
                ax.set_title(CL[cond], fontsize=10)
            if r == 2:
                ax.set_xlabel('γ (share of global rank matching)')
            ax.set_xticks([-.6] + list(xp.values())); ax.set_xticklabels(['ERM'] + [f'{g:g}' for g in gammas])
            if ylim[1] is not None:
                ax.set_ylim(*ylim)
            else:
                top = max(m[col].max() for col, _, _ in series)
                ax.set_ylim(0, top * 1.08)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
        axes[r, 0].legend(fontsize=7.5, loc='best')
    fig.suptitle(f'{TITLES[ds]} — {s.feature_set}, MADE width {s.made_hidden}. Dots: individual seeds; line: seed mean; squares/dashed: ERM baseline.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(FIG / f'{ds}_sweep.png', dpi=170)
    plt.close(fig)


def _hist(ax, p, mask, color, label, alpha=.55):
    bins = np.linspace(0, 1, 31)
    ax.hist(p[mask], bins=bins, density=True, histtype='stepfilled', color=color, alpha=alpha, lw=0, label=label)
    ax.hist(p[mask], bins=bins, density=True, histtype='step', color=color, lw=1.2)


def fig_distributions(dists, ds, seed=42):
    """Petersen-style view: score distributions per group, marginally (independence) and within the positive class (EO)."""
    conds = ['natural', 'shift75']
    fig, axes = plt.subplots(len(conds), 4, figsize=(13, 3.4 * len(conds)), sharex=True)
    for r, cond in enumerate(conds):
        d = dists[(ds, cond, seed)]
        y, a = d['y'], d['a']
        panels = [('erm', 'ERM · all individuals', None), ('fnf', 'FNF γ=1 · all individuals', None),
                  ('erm', 'ERM · positives (Y=1) only', 1), ('fnf', 'FNF γ=1 · positives (Y=1) only', 1)]
        for c, (model, title, ylab) in enumerate(panels):
            ax = axes[r, c]
            p, thr = d[model], d[f'thr_{model}']
            mask = np.ones_like(y, bool) if ylab is None else (y == ylab)
            for g, color in ((0, C['g0']), (1, C['g1'])):
                _hist(ax, p, mask & (a == g), color, f'A={g}')
            ax.axvline(thr, color=C['ink'], lw=1, ls='--')
            if ylab == 1:
                tpr = [(p[(a == g) & (y == 1)] >= thr).mean() for g in (0, 1)]
                ax.text(.98, .95, f'TPR A=0: {tpr[0]:.2f}\nTPR A=1: {tpr[1]:.2f}\nEOpp gap {abs(tpr[0]-tpr[1]):.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=8)
            else:
                ppr = [(p[a == g] >= thr).mean() for g in (0, 1)]
                ax.text(.98, .95, f'P(Ŷ=1) A=0: {ppr[0]:.2f}\nP(Ŷ=1) A=1: {ppr[1]:.2f}\nDP gap {abs(ppr[0]-ppr[1]):.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=8)
            if r == 0:
                ax.set_title(title, fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f'{CL[cond]} prevalence\ndensity', fontsize=9)
            if r == len(conds) - 1:
                ax.set_xlabel('predicted probability  P(Y=1 | Z)')
            ax.set_xlim(0, 1)
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'{TITLES[ds]} (seed {seed}). Left pair: over all individuals, FNF makes the two groups\' score distributions overlap (independence).\n'
                 'Right pair: among true positives, the low-prevalence group A=0 is pushed below the threshold more often (unequal TPR). Dashed: Youden threshold.', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(FIG / f'{ds}_distributions.png', dpi=170)
    plt.close(fig)


def fig_rates_calibration(boot, dists, ds):
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
    # row 1: per-group TPR / FPR, ERM vs FNF, seeds as dots
    for c, cond in enumerate(COND):
        ax = fig.add_subplot(gs[0, c])
        sub = boot[(boot.dataset == ds) & (boot.condition == cond)]
        x = np.arange(2); w = .19
        for i, (model, alpha, name) in enumerate((('erm', .4, 'ERM'), ('fnf', 1.0, 'FNF γ=1'))):
            q = sub[sub.model == model]
            for j, (g, color) in enumerate(((0, C['g0']), (1, C['g1']))):
                pos = x - 1.5 * w + i * 2 * w + (j - .5) * w
                vals = q[[f'tpr{g}', f'fpr{g}']].mean().values
                ax.bar(pos, vals, w, color=color, alpha=alpha, label=f'{name}, A={g}' if c == 0 else None)
                ax.scatter(np.repeat(pos, len(q)), q[[f'tpr{g}', f'fpr{g}']].values.T.ravel(), s=9, color=C['ink'], zorder=5, lw=0)
        ax.set_xticks(x); ax.set_xticklabels(['TPR', 'FPR']); ax.set_ylim(0, 1); ax.set_title(CL[cond], fontsize=10)
        if c == 0:
            ax.set_ylabel('rate on matched test set'); ax.legend(fontsize=7, ncol=2, loc='upper right')
    # row 2: reliability diagrams per group, ERM vs FNF, natural and shift75 (seed 42)
    for c, cond in enumerate(['natural', 'shift75']):
        d = dists[(ds, cond, 42)]
        for k, model in enumerate(('erm', 'fnf')):
            ax = fig.add_subplot(gs[1, 2 * c + k])
            p, y, a = d[model], d['y'], d['a']
            bins = np.linspace(0, 1, 11)
            for g, color in ((0, C['g0']), (1, C['g1'])):
                m = a == g
                ids = np.clip(np.digitize(p[m], bins) - 1, 0, 9)
                xs, ys, ns = [], [], []
                for b in range(10):
                    sel = ids == b
                    if sel.sum() >= 20:
                        xs.append(p[m][sel].mean()); ys.append(y[m][sel].mean()); ns.append(sel.sum())
                ax.plot(xs, ys, marker='o', ms=4, lw=1.6, color=color, label=f'A={g} (ECE {ece(y[m], p[m]):.3f})')
            ax.plot([0, 1], [0, 1], color='#b5b5b0', lw=1, ls='--')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f'{"ERM" if model == "erm" else "FNF γ=1"} · {CL[cond]} · reliability by group', fontsize=9.5)
            ax.set_xlabel('mean predicted probability (bin)')
            if 2 * c + k == 0:
                ax.set_ylabel('observed positive rate')
            ax.legend(fontsize=7.5, loc='upper left')
    fig.suptitle(f'{TITLES[ds]} — per-group error rates (top; dots are seeds) and per-group calibration (bottom, seed 42)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(FIG / f'{ds}_rates_calibration.png', dpi=170)
    plt.close(fig)


def fig_summary(boot, pool):
    cols = {'adult': '#2a78d6', 'dutch': '#eb6834', 'crime': '#1baf7a', 'credit': '#eda100', 'compas': '#e87ba4'}
    markers = {'equalized': 'o', 'natural': 's', 'shift50': '^', 'shift75': 'D'}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    d = boot[boot.model == 'delta']
    for ds in DATASETS:
        for cond in COND:
            q = d[(d.dataset == ds) & (d.condition == cond)]
            ax.errorbar(q.dp, q.eodds, xerr=q.dp_boot_sd, yerr=q.eodds_boot_sd, fmt=markers[cond], color=cols[ds], ms=6, alpha=.85, lw=.8, capsize=0,
                        label=TITLES[ds] if cond == 'natural' else None)
    ax.axhline(0, color='#b5b5b0', lw=1); ax.axvline(0, color='#b5b5b0', lw=1)
    ax.set_xlabel('Δ demographic parity gap (FNF γ=1 − ERM)'); ax.set_ylabel('Δ equalized odds gap (FNF γ=1 − ERM)')
    ax.set_title('Per seed and condition, paired bootstrap ±1 sd', fontsize=10)
    h, l = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker=markers[c], color='#555', ls='', ms=6) for c in COND]; l += [CL[c] for c in COND]
    ax.legend(h, l, fontsize=7.5, ncol=2, loc='upper left')
    ax = axes[1]
    p = pool[pool.model != 'delta']
    xt, lab = [], []
    i = 0
    for ds in DATASETS:
        for cond in COND:
            e = p[(p.dataset == ds) & (p.condition == cond) & (p.model == 'erm')].iloc[0]
            f = p[(p.dataset == ds) & (p.condition == cond) & (p.model == 'fnf')].iloc[0]
            ax.plot([e.eodds_mean, f.eodds_mean], [i, i], color='#c9c9c4', lw=2, zorder=1)
            ax.scatter(e.eodds_mean, i, marker='s', s=34, color='white', edgecolor=cols[ds], lw=1.5, zorder=3)
            ax.scatter(f.eodds_mean, i, marker='o', s=40, color=cols[ds], zorder=4)
            xt.append(i); lab.append(f'{TITLES[ds]} · {CL[cond]}'); i += 1
        i += .6
    ax.set_yticks(xt); ax.set_yticklabels(lab, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel('equalized odds gap (Youden threshold, seed mean)')
    ax.set_title('Hollow square: ERM · filled circle: FNF γ=1', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / 'summary_delta.png', dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- dataset details
def dataset_details(sel):
    rows = []
    for ds in DATASETS:
        man = json.loads((BASE / 'prepared' / ds / 'manifest.json').read_text())
        s = sel[sel.dataset == ds].iloc[0]
        feats = json.loads((BASE / 'runs' / ds / 'seed42' / 'natural' / 'features.json').read_text())[s.feature_set]
        cond = pd.read_csv(BASE / 'prepared' / ds / 'conditions.csv')
        cond = cond[cond.split == 'train'].set_index('condition')
        rows.append(dict(dataset=ds, title=TITLES[ds], rows=man['rows'], splits=man['split_support'], attribute=man['mapping'], label=man['label'], unit=man['unit'],
                         feature_set=s.feature_set, features=feats['columns'], dims=feats['categorical']['dims'], cells=int(np.prod(feats['categorical']['dims'])),
                         bin_edges=feats['categorical'].get('bin_edges') or ({'x_age': feats['categorical'].get('age_bins')} if feats['categorical'].get('age_bins') else {}),
                         natural_prevalence={k: round(v, 3) for k, v in man['natural_group_prevalence'].items()},
                         train_prevalence={c: (round(cond.loc[c, 'group0_prevalence'], 3), round(cond.loc[c, 'group1_prevalence'], 3)) for c in COND},
                         train_n={c: int(cond.loc[c, 'n']) for c in COND}, source=man.get('source', {})))
    (TAB / 'dataset_details.json').write_text(json.dumps(rows, indent=1, default=str))
    return rows


# ----------------------------------------------------------------------------- markdown
def write_markdown(sel, pool, boot, details):
    L = []
    A = L.append
    A('# Invariance achieves independence but not equalized odds: Fair Normalizing Flows on five tabular benchmarks\n')
    A('RQ6 report, 2026-09-15. Adult income plus the four additional datasets on which the categorical FNF variant reaches '
      'near-independence (Dutch census, Communities & Crime, Taiwan credit default, COMPAS; COMPAS is reported as the weakest case). Law School is excluded because '
      'FNF at γ=1 leaves the task classifier at chance (AUROC ≈ .50), so its fairness gaps are uninformative; Heritage Health '
      '(continuous variant) is excluded because independence was not reached (see `REPORT.md`). Every number below is on the '
      'held-out test split under the *matched* prevalence condition (model trained and tested under the same condition).\n')
    A('Files: tables in `paper/tables/`, figures in `paper/figures/`, generated by `experiments/rq6_fnf/paper_report.py`.\n')

    A('## 1. Claim and evidence structure\n')
    A('A representation Z that is (near-)independent of the sensitive attribute A satisfies demographic parity for any downstream '
      'classifier, but when base rates P(Y=1|A) differ, independence forces the score to be miscalibrated within groups and the '
      'classifier to trade recall in the low-prevalence group for parity. Equalized odds (equal TPR and FPR) therefore does not follow '
      'and typically worsens as the base-rate gap grows (Petersen et al., arXiv:2305.01397). We test this with a method whose '
      'independence is *certified by construction* rather than adversarially estimated: Fair Normalizing Flows (Balunović, Ruoss & '
      'Vechev, ICLR 2022, arXiv:2106.05937), categorical variant, which controls the total-variation distance between the two '
      'group-conditional representation distributions exactly.\n')
    A('For each dataset we (i) inject controlled base-rate gaps (equalized / natural / shift50 / shift75), (ii) sweep the FNF '
      'trade-off γ and two density widths over three seeds, (iii) measure decodability of A with independent probes, DP, EO, equal '
      'opportunity, calibration (ECE overall and per group) and task AUROC, and (iv) compare the maximally invariant FNF configuration '
      'against an ERM baseline on identical features with paired bootstraps.\n')

    A('## 2. Fair Normalizing Flows — categorical variant as implemented\n')
    A('Implementation: `experiments/rq6_fnf/fnf.py`, `run.py` (a port of github.com/eth-sri/fnf; unit tests in `test_rq6.py`).\n')
    A('1. **Inputs.** All features are categorical; numeric ones are quantile-binned on the training split (bin edges in each run\'s '
      '`features.json`). An input is a tuple of category codes; the space of all combinations has ∏ dims cells (per-dataset counts in §3).\n'
      '2. **Per-group densities.** For each group a ∈ {0,1} a MADE (masked autoencoder for distribution estimation; hidden layers '
      '[h, h] with h ∈ {50, 100}; Adam, lr 1e-2, weight decay 1e-4, batch 64, 100 epochs, seed = shard seed + a) is fit on the one-hot '
      'encoded training rows of that group. The autoregressive masks make the product of Bernoulli conditionals a proper density; the '
      'density is evaluated on **every** cell and renormalised, giving exact discrete distributions p₀, p₁ over the cells.\n'
      '3. **Prediction classes for the mapping.** A logistic regression on one-hot inputs with group-balanced sample weights predicts Ŷ '
      'for every cell; this partitions the cells into two classes.\n'
      '4. **Rank-matching maps (the encoder).** Group 1 (the higher natural prevalence group in every dataset) is the reference and is '
      'left unchanged. Group 0 cells are mapped to group 1 cells by matching probability ranks: with probability 1−γ *within* the '
      'predicted class (preserving the logistic prediction, hence utility), with probability γ *globally* across all cells (ignoring '
      'utility). γ = 0 is the most utility-preserving map, γ = 1 the most invariant. The pushed-forward distribution of group 0 is '
      'computed exactly, and the statistical (total-variation) distance TV(p₁, T#p₀) between the two group-conditional representation '
      'distributions is reported as `stat_dist_exact_made` — an exact certificate for the fitted densities.\n'
      '5. **Encoding.** Each row is mapped through the map of its group (the sensitive attribute is required at encoding time, as in the '
      'original method) and one-hot encoded; this is Z.\n'
      '6. **Classifier.** An MLP with one hidden layer of 20 units on Z, trained with the mean of the two per-group cross-entropies '
      '(group-balanced, as in the paper), Adam lr 1e-3, weight decay 1e-4, batch 128, 100 epochs.\n'
      '7. **ERM baseline.** The same MLP on the one-hot inputs with plain cross-entropy (no mapping); its representation is the input.\n'
      '8. **Sweep.** γ ∈ {0, .02, .1, .2, .5, .9, 1} × h ∈ {50, 100} × two feature sets + two ERM fits = 30 fits per (dataset, seed, '
      'condition); seeds 42 / 123 / 456; 4 conditions → 360 fits per dataset, 1,800 for the five datasets here. Final-epoch checkpoints '
      '(paper protocol).\n')
    A('**Evaluation.** Thresholds are Youden-optimal on the *validation* split and applied to test. Decodability of A from Z is the '
      'maximum test AUROC over five probe families (logistic regression, gradient-boosted trees, three MLPs) fit on the probe-validation '
      'split, marginally and within each label. Fairness: DP gap |P(Ŷ=1|A=0)−P(Ŷ=1|A=1)|, equal-opportunity gap |TPR₀−TPR₁|, EO gap '
      'max(|TPR₀−TPR₁|, |FPR₀−FPR₁|). Calibration: ECE with 15 equal-width bins, overall and per group. Uncertainty: for the selected '
      'configuration, 1,000 paired row-bootstraps of the test set (same resampled rows for ERM and FNF of the same seed), giving a '
      'bootstrap sd per seed for each metric and for each FNF−ERM difference; across seeds we report the mean of point estimates ± sd '
      'across the three seeds.\n')
    A('**Prevalence conditions** (applied with fixed seeds to train / validation / probe-validation; the test split is the natural one '
      'with the same operation applied for the matched variant): *natural*; *equalized* — the higher-prevalence group is subsampled to '
      'the lower rate (DP and EO then coincide); *shift50* / *shift75* — 50 % / 75 % of A=0, Y=1 rows removed, widening the base-rate gap.\n')
    A('**Selection rule.** One FNF configuration per dataset, chosen *only on independence*: γ = 1 and the (feature set, MADE width) '
      'with the lowest mean marginal probe AUROC over seeds and conditions. EO never enters the choice. Candidates and their probe AUROC '
      'are in `paper/tables/selection_candidates.json`.\n')
    A('| dataset | selected feature set | MADE width | mean probe AUROC (A) | mean task AUROC | candidates |')
    A('|---|---|---|---|---|---|')
    for _, s in sel.iterrows():
        A(f'| {TITLES[s.dataset]} | {s.feature_set} | {s.made_hidden} | {s.mean_probe_auc:.3f} | {s.mean_task_auroc:.3f} | {s.candidates} |')
    A('')

    A('## 3. Datasets\n')
    A('| dataset | rows (train / val / probe-val / test) | A = 0 / A = 1 | Y | natural P(Y=1) A=0 / A=1 | features (selected set) | cells |')
    A('|---|---|---|---|---|---|---|')
    for d in details:
        sp = d['splits']
        A(f"| {d['title']} | {d['rows']:,} ({sp['train']:,} / {sp['validation']:,} / {sp['probe_validation']:,} / {sp['test']:,}) | "
          f"{d['attribute']['0']} / {d['attribute']['1']} | {d['label']} | {d['natural_prevalence']['0']:.2f} / {d['natural_prevalence']['1']:.2f} | "
          f"{', '.join(c[2:] for c in d['features'])} (dims {d['dims']}) | {d['cells']:,} |")
    A('')
    A('Training-split group prevalences per condition (A=0 / A=1):\n')
    A('| dataset | equalized | natural | shift50 | shift75 |')
    A('|---|---|---|---|---|')
    for d in details:
        tp = d['train_prevalence']
        A(f"| {d['title']} | " + ' | '.join(f'{tp[c][0]:.2f} / {tp[c][1]:.2f}' for c in COND) + ' |')
    A('')
    A('Sources and preprocessing: Adult — UCI Adult (48,842 rows, RQ5 cohort reused; features workclass, education, marital status, '
      'occupation, relationship, race, quantile-binned age; A = sex). Dutch census 2001 (Le Quy et al. mirror; occupation = high-level '
      'as label; A = sex; age, education, economic status, current economic activity, marital status, household position). '
      'Communities & Crime (UCI; 1,994 communities; Y = violent crimes per population above the median; A = share of Black residents '
      'at or above the median; six socio-economic covariates quantile-binned into five levels — the selected set uses four of them). '
      'Taiwan credit default (UCI; 30,000 clients; Y = default next month; A = sex; education, marriage, binned age, most recent '
      'repayment status, binned credit limit). COMPAS (ProPublica two-year recidivism with the standard filter; African-American vs '
      'Caucasian defendants, 5,278 rows; sex, age category, charge degree, binned prior count, three juvenile-count indicators). Splits '
      'are 60 / 10 / 10 / 20 by unit with a fixed seed; exact hashes of raw files are in each prepared manifest.\n')

    A('## 4. Cross-dataset summary\n')
    A('![summary](paper/figures/summary_delta.png)\n')
    A('*Figure S. Left: change in DP gap against change in EO gap when moving from ERM to FNF γ=1, one point per dataset × condition × '
      'seed with paired-bootstrap ±1 sd. Almost every point lies in the upper-left quadrant: DP removed, EO unchanged or worse. Right: '
      'EO gap of ERM (hollow) and FNF (filled) per dataset and condition.*\n')
    A('| dataset | condition | probe AUC ERM → FNF | DP ERM → FNF | ΔDP | EO ERM → FNF | ΔEO | worst-group ECE ERM → FNF | AUROC ERM → FNF |')
    A('|---|---|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        for cond in COND:
            sub = pool[(pool.dataset == ds) & (pool.condition == cond)].set_index('model')
            e, f, d = sub.loc['erm'], sub.loc['fnf'], sub.loc['delta']
            A(f'| {TITLES[ds]} | {CL[cond]} | {e.probe_auc_mean:.2f} → {f.probe_auc_mean:.2f} | {e.dp_mean:.3f} → {f.dp_mean:.3f} | {d.dp_mean:+.3f} ± {d.dp_seed_sd:.3f} | '
              f'{e.eodds_mean:.3f} → {f.eodds_mean:.3f} | {d.eodds_mean:+.3f} ± {d.eodds_seed_sd:.3f} | {e.max_group_ece_mean:.3f} → {f.max_group_ece_mean:.3f} | {e.auroc_mean:.2f} → {f.auroc_mean:.2f} |')
    A('')
    A('Reading: ΔDP is large and negative everywhere, so parity is achieved in every dataset and condition. ΔEO is positive in all '
      'four conditions for Adult, Dutch census and Taiwan credit, and for Communities & Crime under natural and shifted prevalence; '
      'in Dutch and Credit it grows monotonically with the injected base-rate gap. COMPAS is the weakest case: EO shrinks modestly '
      'under equalized/natural prevalence but stays clearly non-zero (.06–.19) while DP is ≈ .02, and turns upward only at shift75. '
      'Under equalized prevalence DP and EO coincide as constraints, yet ΔEO is still positive for Adult, Dutch and Credit because the '
      'residual dependence (probe AUROC .53–.56) is re-distributed within labels. Worst-group ECE rises under FNF in every shifted '
      'condition: a score independent of A cannot be calibrated in both groups when their base rates differ. ± is the sd across the '
      'three seeds of the per-seed paired differences; per-seed bootstrap sds are in Appendix A.\n')

    for ds in DATASETS:
        s = sel[sel.dataset == ds].iloc[0]
        d = [x for x in details if x['dataset'] == ds][0]
        A(f'## 5.{DATASETS.index(ds) + 1} {TITLES[ds]}\n')
        A(f"Selected FNF: `{s.fnf_run}` (feature set {s.feature_set}, MADE width {s.made_hidden}, γ = 1); baseline `{s.erm_run}`. "
          f"Features: {', '.join(c[2:] for c in d['features'])}; {d['cells']:,} cells. A = 0: {d['attribute']['0']}, A = 1: {d['attribute']['1']}; "
          f"Y: {d['label']}.\n")
        A(f'![{ds} sweep](paper/figures/{ds}_sweep.png)\n')
        A(f'*Figure {ds}-1. γ sweep (dots: seeds; line: seed mean; squares and dashed lines: ERM). Top: marginal decodability of A falls '
          'toward chance as γ → 1 while within-label decodability does not (or rises). Middle: DP gap collapses; EO gap does not. Bottom: '
          'worst-group ECE and task AUROC.*\n')
        A('**Selected configuration vs ERM, mean over seeds ± sd across seeds** (per-seed bootstrap sds in the appendix table):\n')
        A(main_table(pool, ds)); A('')
        A(f'![{ds} distributions](paper/figures/{ds}_distributions.png)\n')
        A(f'*Figure {ds}-2. Score distributions by group (seed 42). Left pair: over all individuals the two groups\' score distributions '
          'overlap after FNF (independence, DP ≈ 0). Right pair: among true positives the low-prevalence group A = 0 sits below the '
          'threshold far more often than A = 1, i.e. unequal TPR. Under ERM the marginal distributions differ but the positives are '
          'ranked more similarly.*\n')
        A(f'![{ds} rates](paper/figures/{ds}_rates_calibration.png)\n')
        A(f'*Figure {ds}-3. Top: per-group TPR and FPR, ERM (faded) vs FNF (solid), seeds as dots. Bottom: reliability diagrams per '
          'group; independence forces the FNF score to over-predict for the low-prevalence group and under-predict for the other, so '
          'per-group ECE rises even when overall ECE is small.*\n')

    A('## 6. Calibration\n')
    A('Overall ECE is often *lower* under FNF than ERM, while per-group ECE is much higher. This is the mechanism behind the EO result: '
      'when P(Z|A=0) = P(Z|A=1) the score s(Z) has the same distribution in both groups, so E[s|A=0] = E[s|A=1] while the true '
      'positive rates differ; group-wise calibration is therefore impossible, and the classifier at a common threshold necessarily has '
      'lower TPR (and higher FPR) in the low-prevalence group. Table columns "ECE A=0 / A=1" quantify this per dataset; the "worst-group '
      'ECE" panel of each sweep figure shows it growing with γ and with the injected base-rate gap.\n')

    A('## 7. Limitations\n')
    A('- Independence is approximate: the strongest probe still reaches AUROC .52–.59 at γ = 1 (exact TV between the fitted group '
      'densities .05–.10). The claims concern near-independence, which is the regime any practical invariance method operates in.\n'
      '- Prevalence shifts are synthetic row deletions; test variants are built identically from the natural test split so ERM and FNF '
      'are compared like for like. EO of a fixed score is invariant to removing positives from the test set, so the training-time '
      'base-rate gap is what matters.\n'
      '- Three seeds measure fitting variability on fixed splits; bootstrap sds measure test-sample variability. Communities & Crime is '
      'small (400 test rows), so its bootstrap sds are wide.\n'
      '- Youden thresholds are the operating points reported; fixed 0.5 thresholds understate all gaps because FNF compresses scores '
      'toward the base rate (see `REPORT.md`). Threshold curves and within-label KS distances are saved per fit for threshold-free views.\n'
      '- FNF requires A at encoding time, as in the original method; ERM does not use A.\n'
      '- Law School (excluded) shows the other failure mode: full invariance can leave no task signal (AUROC ≈ .50); Health (excluded) '
      'shows that the continuous variant\'s certificate is conditional on prior quality.\n')

    A('## Appendix A. Per-seed tables (point estimate ± paired-bootstrap sd, B = 1000)\n')
    for ds in DATASETS:
        A(f'### {TITLES[ds]}\n')
        A(seed_table(boot, ds)); A('')
    A('## Appendix B. Files\n')
    A('| path | content |\n|---|---|')
    A('| `paper/tables/selected_configurations.csv` | selection per dataset |')
    A('| `paper/tables/selection_candidates.json` | all γ = 1 candidates with mean probe AUROC / task AUROC |')
    A('| `paper/tables/bootstrap_selected_per_seed.csv` | per dataset × condition × seed × {erm, fnf, delta}: point estimates, bootstrap mean/sd, CIs of deltas |')
    A('| `paper/tables/bootstrap_selected_pooled.csv` | across-seed means, seed sds and mean bootstrap sds |')
    A('| `paper/tables/dataset_details.json` | rows, splits, attribute mapping, features, dims, bin edges, prevalences, sources |')
    A('| `evaluation_metrics.csv`, `summary_by_gamma.csv` | full sweep (all fits, all test variants) |')
    A('| `/path/to/results/rq6/fnf_v1/runs/<dataset>/seed*/<condition>/<run>/` | models, mappings, representations, predictions, probe audits |')
    (RES / 'PAPER_REPORT.md').write_text('\n'.join(L))


if __name__ == '__main__':
    df = load_metrics()
    sel = select_configs(df)
    print(sel.drop(columns='candidate_table').to_string())
    details = dataset_details(sel)
    boot, dists = paired_bootstrap(sel)
    pool = pooled(boot)
    for ds in DATASETS:
        fig_sweep(df, sel, ds)
        fig_distributions(dists, ds)
        fig_rates_calibration(boot, dists, ds)
    fig_summary(boot, pool)
    write_markdown(sel, pool, boot, details)
    print('written', RES / 'PAPER_REPORT.md')
