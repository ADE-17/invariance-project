"""Class-conditional FNF trial: tables and figures comparing ERM, marginal FNF (gamma=1) and class-conditional FNF
(cond-y oracle, cond-yhat deployable) on the matched test condition. Writes results/rq6/CONDITIONAL_TRIAL.md and
results/rq6/paper/figures/conditional_*.png; reads per-run evaluation/metrics.json and training_summary.json directly."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rq6_fnf.calibration_bias import run_rows as calibration_rows

BASE = Path('/path/to/results/rq6/fnf_v1/runs')
RES = Path(__file__).resolve().parents[2] / 'results' / 'rq6'
FIG = RES / 'paper' / 'figures'
TRIAL = dict(adult=('paper6_age', 50), dutch=('core6_hh', 50), crime=('socio4', 50), credit=('core5', 50), compas=('paper7', 50))
SEEDS = [42, 123, 456]
COND = ['equalized', 'natural', 'shift50', 'shift75']
CL = {'equalized': 'equalized', 'natural': 'natural', 'shift50': 'shift 50', 'shift75': 'shift 75'}
TITLES = {'adult': 'Adult income', 'dutch': 'Dutch census', 'crime': 'Communities & Crime', 'credit': 'Taiwan credit', 'compas': 'COMPAS'}
MODELS = [('erm', 'ERM'), ('fnf1', 'FNF γ=1 (Z ⊥ A)'), ('cy0.5', 'class-cond. FNF γ=.5, oracle Y'), ('cy1', 'class-cond. FNF γ=1, oracle Y'),
          ('cyhat0.5', 'class-cond. FNF γ=.5, predicted Ŷ'), ('cyhat1', 'class-cond. FNF γ=1, predicted Ŷ')]
COLORS = {'erm': '#6b6b66', 'fnf1': '#2a78d6', 'cy0.5': '#f0a86f', 'cy1': '#eb6834', 'cyhat0.5': '#7dd3b0', 'cyhat1': '#1baf7a'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e2',
                     'grid.linewidth': .6, 'axes.edgecolor': '#b5b5b0', 'figure.facecolor': 'white', 'legend.frameon': False})


def run_name(ds, key):
    fs, h = TRIAL[ds]
    if key == 'erm':
        return f'erm_{fs}'
    if key == 'fnf1':
        return f'fnf_{fs}_gamma1_made{h}'
    cond_on = 'yhat' if key.startswith('cyhat') else 'y'
    g = key.replace('cyhat', '').replace('cy', '')
    return f'cfnf_{fs}_cond-{cond_on}_gamma{g}_made{h}'


CAL = ['cal_slope', 'group0_cal_slope', 'group1_cal_slope', 'cal_intercept_fixed_slope', 'group0_cal_intercept_fixed_slope', 'group1_cal_intercept_fixed_slope',
       'cal_intercept_fixed_slope_gap', 'group0_cal_mean_bias', 'group1_cal_mean_bias', 'cal_mean_bias_gap']
COLS = ['marginal_max_probe_auc', 'conditional_max_probe_auc', 'dp', 'eopp', 'eodds', 'eodds_ci_low', 'eodds_ci_high', 'dp_ci_low', 'dp_ci_high',
        'ece', 'group0_ece', 'group1_ece', 'max_group_ece', 'auroc', 'group0_tpr', 'group1_tpr', 'group0_fpr', 'group1_fpr', 'threshold']


def collect():
    rows = []
    for ds in TRIAL:
        for seed in SEEDS:
            for cond in COND:
                for key, label in MODELS:
                    folder = BASE / ds / f'seed{seed}' / cond / run_name(ds, key)
                    mp = folder / 'evaluation' / 'metrics.json'
                    if not mp.exists():
                        rows.append(dict(dataset=ds, seed=seed, condition=cond, model=key, label=label, status='failed' if (folder / 'failed.json').exists() else 'missing'))
                        continue
                    r = [x for x in json.load(open(mp)) if x['test'] == f'test_{cond}'][0]
                    row = dict(dataset=ds, seed=seed, condition=cond, model=key, label=label, status='ok', **{c: r.get(c) for c in COLS})
                    ts = json.load(open(folder / 'training_summary.json'))
                    for k in ['stat_dist_exact_made', 'stat_dist_exact_made_stratum0', 'stat_dist_exact_made_stratum1', 'empirical_tv_test',
                              'empirical_tv_test_withinY0', 'empirical_tv_test_withinY1', 'train_rows_stratum1', 'predicted_class_share']:
                        if k in ts:
                            row[k] = ts[k] if not isinstance(ts[k], dict) else json.dumps(ts[k])
                    if 'yhat_accuracy' in ts:
                        row['yhat_test_accuracy'] = ts['yhat_accuracy']['test']
                    cal = [x for x in calibration_rows(folder, False) if x['test_condition'] == cond][0]
                    row.update({k: cal[k] for k in CAL})
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'paper' / 'tables' / 'conditional_trial_per_seed.csv', index=False)
    return df


def fmt(v, sd):
    return f'{v:.3f} ± {sd:.3f}' if np.isfinite(sd) else f'{v:.3f}'


def table(df, ds):
    ok = df[(df.dataset == ds) & (df.status == 'ok')]
    lines = ['| condition | model | probe A (marginal) | probe A within Y | DP gap | EOpp gap | EO gap | ECE all | ECE A=0 | ECE A=1 | calib. intercept A=0 / A=1 | calib. slope A=0 / A=1 | TPR A=0 / A=1 | AUROC |', '|' + '---|' * 14]
    for cond in COND:
        for key, label in MODELS:
            q = ok[(ok.condition == cond) & (ok.model == key)]
            if q.empty:
                continue
            m, s = q.mean(numeric_only=True), q.std(numeric_only=True, ddof=1)
            lines.append(f"| {CL[cond]} | {label} | {m.marginal_max_probe_auc:.3f} | {m.conditional_max_probe_auc:.3f} | {fmt(m.dp, s.dp)} | {fmt(m.eopp, s.eopp)} | "
                         f"{fmt(m.eodds, s.eodds)} | {m.ece:.3f} | {m.group0_ece:.3f} | {m.group1_ece:.3f} | {m.group0_cal_intercept_fixed_slope:+.2f} / {m.group1_cal_intercept_fixed_slope:+.2f} | "
                         f"{m.group0_cal_slope:.2f} / {m.group1_cal_slope:.2f} | {m.group0_tpr:.2f} / {m.group1_tpr:.2f} | {fmt(m.auroc, s.auroc)} |")
    return '\n'.join(lines)


def figure(df, ds):
    ok = df[(df.dataset == ds) & (df.status == 'ok')]
    panels = [('conditional_max_probe_auc', 'decodability of A within label (probe AUROC)'), ('dp', 'DP gap'), ('eodds', 'EO gap'),
              ('max_group_ece', 'worst-group ECE'), ('cal_intercept_fixed_slope_gap', 'calibration intercept gap (A=1 − A=0)'), ('auroc', 'task AUROC')]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.8))
    keys = [k for k, _ in MODELS]
    w = .13
    for ax, (col, title) in zip(axes, panels):
        for i, key in enumerate(keys):
            for c, cond in enumerate(COND):
                q = ok[(ok.condition == cond) & (ok.model == key)]
                if q.empty:
                    continue
                x = c + (i - (len(keys) - 1) / 2) * w
                ax.bar(x, q[col].mean(), w * .92, color=COLORS[key], label=dict(MODELS)[key] if (c == 0 and col == 'dp') else None)
                ax.scatter([x] * len(q), q[col], s=8, color='#222', zorder=5, lw=0)
        ax.set_xticks(range(4)); ax.set_xticklabels([CL[c] for c in COND], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        if col == 'conditional_max_probe_auc':
            ax.set_ylim(.45, 1.0); ax.axhline(.5, color='#b5b5b0', lw=1, ls='--')
        elif col == 'auroc':
            ax.set_ylim(.5, 1.0)
        elif col.startswith('cal_'):
            ax.axhline(0, color='#b5b5b0', lw=1, ls='--')
        else:
            ax.set_ylim(0, None)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc='lower center', ncol=6, bbox_to_anchor=(.5, -.01))
    fig.suptitle(f'{TITLES[ds]} — marginal vs class-conditional invariance (seeds as dots, bars = seed mean; matched test condition, Youden threshold)', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.savefig(FIG / f'conditional_{ds}.png', dpi=170)
    plt.close(fig)


def write(df):
    L = []
    A = L.append
    A(f"# Class-conditional invariance (Z ⊥ A | Y) — {', '.join(TITLES[d] for d in TRIAL)}; seeds {' / '.join(map(str, SEEDS))}\n")
    A('Generated by `experiments/rq6_fnf/conditional_report.py`. Per-seed values in `paper/tables/conditional_trial_per_seed.csv`. '
      'Tables: mean over seeds ± sd across seeds; all on the matched test condition at the Youden threshold. '
      'Calibration intercept is the Cox fixed-slope intercept a in P(Y=1|p)=σ(a+logit p) per group (a>0: risk under-predicted for that group; a<0: over-predicted); '
      'calibration slope is b from the joint fit (1 = well spread, <1 over-confident).\n')
    A('**Variant.** Same categorical FNF machinery, but MADE densities are fit per (group, stratum) and group 0 is rank-matched onto '
      'group 1 *within each stratum* of a conditioning variable C, so the certificate is TV(P(Z|A=0,C=c), P(Z|A=1,C=c)) per stratum. '
      '`oracle Y`: C is the true label on every split including test (theoretical check; not deployable). `predicted Ŷ`: C is the '
      'logistic-regression predicted class computed from X (deployable). γ mixes within-predicted-class (1−γ) and global (γ) rank '
      'matching inside each stratum, as before.\n')
    miss = df[df.status != 'ok']
    if len(miss):
        A('**Runs not available:** ' + '; '.join(f"{r.dataset} seed {r.seed} {r.condition} {r.label} ({r.status})" for r in miss.itertuples()) + '\n')
    A('## Summary\n')
    ok = df[df.status == 'ok']
    A('| dataset | condition | metric | ERM | FNF γ=1 | class-cond. oracle γ=1 | class-cond. predicted γ=1 |')
    A('|---|---|---|---|---|---|---|')
    for ds in TRIAL:
        for cond in COND:
            for col, name in [('conditional_max_probe_auc', 'probe A within Y'), ('dp', 'DP gap'), ('eodds', 'EO gap'), ('max_group_ece', 'worst-group ECE'), ('cal_intercept_fixed_slope_gap', 'calib. intercept gap (A=1 − A=0)'), ('auroc', 'AUROC')]:
                vals = []
                for key in ['erm', 'fnf1', 'cy1', 'cyhat1']:
                    q = ok[(ok.dataset == ds) & (ok.condition == cond) & (ok.model == key)][col]
                    vals.append((f'{q.mean():+.3f}' if 'gap' in col and col.startswith('cal') else f'{q.mean():.3f}') if len(q) else '—')
                A(f'| {TITLES[ds]} | {CL[cond]} | {name} | ' + ' | '.join(vals) + ' |')
    A('')
    for ds in TRIAL:
        A(f'## {TITLES[ds]}\n')
        A(f'![{ds}](paper/figures/conditional_{ds}.png)\n')
        A(table(df, ds)); A('')
        strata = ok[(ok.dataset == ds) & ok.model.isin(['cy1', 'cyhat1'])][['condition', 'seed', 'model', 'train_rows_stratum1', 'stat_dist_exact_made_stratum0', 'stat_dist_exact_made_stratum1', 'yhat_test_accuracy']]
        if len(strata):
            A('Stratum diagnostics (γ = 1 runs): training rows in the positive stratum per group, exact TV certificate per stratum, and test accuracy of the predicted class used by the deployable variant.\n')
            A('| condition | seed | model | rows in stratum 1 {A=0: n, A=1: n} | exact TV stratum 0 | exact TV stratum 1 | Ŷ accuracy (test) |')
            A('|---|---|---|---|---|---|---|')
            for r in strata.sort_values(['condition', 'model', 'seed']).itertuples():
                acc = '' if pd.isna(r.yhat_test_accuracy) else f'{r.yhat_test_accuracy:.3f}'
                A(f'| {CL[r.condition]} | {r.seed} | {dict(MODELS)[r.model]} | {r.train_rows_stratum1} | {r.stat_dist_exact_made_stratum0:.3f} | {r.stat_dist_exact_made_stratum1:.3f} | {acc} |')
            A('')
    (RES / 'CONDITIONAL_TRIAL.md').write_text('\n'.join(L))


if __name__ == '__main__':
    df = collect()
    print(df.groupby(['dataset', 'model']).status.value_counts().to_string())
    for ds in TRIAL:
        figure(df, ds)
    write(df)
    print('written', RES / 'CONDITIONAL_TRIAL.md')
