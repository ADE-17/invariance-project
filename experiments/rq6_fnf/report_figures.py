"""Figures and summary tables for the RQ6 report, built from results/rq6/evaluation_metrics.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'results' / 'rq6'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
COND = ['equalized', 'natural', 'shift50', 'shift75']
COND_LABEL = {'equalized': 'equalized', 'natural': 'natural', 'shift50': 'shift 50', 'shift75': 'shift 75'}
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'grey': '#6b6b66', 'ink': '#222'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e2', 'grid.linewidth': .6, 'axes.edgecolor': '#b5b5b0', 'figure.facecolor': 'white'})


def load():
    df = pd.read_csv(ROOT / 'evaluation_metrics.csv', low_memory=False)
    df['gamma'] = df['gamma'].astype(float)
    df['is_erm'] = df['method'].eq('erm')
    return df


def matched(df):
    return df[df['test_condition'] == df['condition']]


def sweep(df, dataset, feature_set, cols, prior=None):
    d = matched(df)
    d = d[(d.dataset == dataset) & (d.feature_set == feature_set)]
    if prior is not None:
        d = d[d.is_erm | (d.prior == prior)]
    erm = d[d.is_erm].groupby('condition')[cols].mean()
    fnf = d[~d.is_erm].groupby(['condition', 'gamma'])[cols].agg(['mean', 'std', 'count'])
    return erm, fnf


def gamma_panels(df, dataset, feature_set, title, fname, prior=None, gamma_scale='linear'):
    cols = ['marginal_max_probe_auc', 'conditional_max_probe_auc', 'paper_adversary_Y1_test_balanced_accuracy',
            'dp', 'eodds', 'fixed05_dp', 'fixed05_eodds', 'auroc']
    erm, fnf = sweep(df, dataset, feature_set, cols, prior)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.2), sharex=True, sharey='row')
    rows = [('decodability of A',
             [('marginal_max_probe_auc', 'strongest probe, marginal', C['blue']),
              ('conditional_max_probe_auc', 'strongest probe, within label', C['orange']),
              ('paper_adversary_Y1_test_balanced_accuracy', 'paper adversary, within Y=1', C['aqua'])], (.45, 1.02)),
            ('gap, Youden threshold',
             [('dp', 'demographic parity gap', C['blue']), ('eodds', 'equalized odds gap', C['orange'])], (0, .7)),
            ('gap @0.5 / task AUROC',
             [('fixed05_dp', 'DP gap @0.5', C['blue']), ('fixed05_eodds', 'EOdds gap @0.5', C['orange']),
              ('auroc', 'task AUROC', C['grey'])], (0, 1.0))]
    for r, (ylabel, series, ylim) in enumerate(rows):
        for c, cond in enumerate(COND):
            ax = axes[r, c]
            if cond not in fnf.index.get_level_values(0):
                ax.set_axis_off(); continue
            g = fnf.loc[cond]
            x = g.index.values
            xp = x if gamma_scale == 'linear' else np.arange(len(x))
            for col, label, color in series:
                m, s = g[(col, 'mean')].values, g[(col, 'std')].fillna(0).values
                ax.plot(xp, m, color=color, lw=2, marker='o', ms=4, label=label)
                ax.fill_between(xp, m - s, m + s, color=color, alpha=.15, lw=0)
                if cond in erm.index:
                    ax.axhline(erm.loc[cond, col], color=color, lw=1, ls=(0, (3, 2)), alpha=.8)
            if r == 0:
                ax.set_title(COND_LABEL[cond], fontsize=10)
            if gamma_scale != 'linear':
                ax.set_xticks(xp); ax.set_xticklabels([f'{v:g}' for v in x])
            ax.set_ylim(*ylim)
            if r == 2:
                ax.set_xlabel('gamma (KL weight)')
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
        axes[r, 0].legend(loc='upper left' if r else 'lower left', fontsize=7.5, frameon=False)
    fig.suptitle(f'{title}\nsolid: FNF mean over seeds/hyper-parameters, band: ±1 sd, dashed: ERM baseline (same colour)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def tension_scatter(df, fname):
    d = matched(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = dict(zip(COND, [C['aqua'], C['blue'], C['orange'], C['yellow']]))
    for ax, (dataset, fs, label) in zip(axes, [('adult', 'paper6', 'Adult, paper6 features (categorical FNF)'),
                                              ('health', 'core', 'Health, core features (continuous FNF)')]):
        s = d[(d.dataset == dataset) & (d.feature_set == fs)]
        for cond in COND:
            q = s[s.condition == cond]
            f = q[~q.is_erm]
            ax.scatter(f.dp, f.eodds, s=14 + 60 * f.gamma.fillna(0), color=colors[cond], alpha=.55, lw=0, label=f'FNF {COND_LABEL[cond]} (size ∝ gamma)')
            e = q[q.is_erm]
            ax.scatter(e.dp, e.eodds, s=70, marker='X', color=colors[cond], edgecolor='white', lw=.8, zorder=5)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color='#b5b5b0', lw=1, ls='--')
        ax.set_xlabel('demographic parity gap (Youden threshold)')
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel('equalized odds gap')
    axes[0].legend(fontsize=7.5, frameon=False, loc='upper left')
    axes[1].text(.02, .95, 'X markers: ERM baseline', transform=axes[1].transAxes, fontsize=8, va='top')
    fig.suptitle('Independence vs equalized odds: every fit on its matched test condition', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def rate_bars(df, fname):
    d = matched(df)
    d = d[(d.dataset == 'adult') & (d.feature_set == 'paper6')]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, cond in zip(axes, COND):
        s = d[d.condition == cond]
        erm = s[s.is_erm][['group0_tpr', 'group1_tpr', 'group0_fpr', 'group1_fpr']].mean()
        g1 = s[(~s.is_erm) & (s.gamma == 1.0)][['group0_tpr', 'group1_tpr', 'group0_fpr', 'group1_fpr']].mean()
        x = np.arange(2)
        w = .18
        for i, (vals, name, alpha) in enumerate([(erm, 'ERM', .45), (g1, 'FNF γ=1', 1.0)]):
            ax.bar(x - w * 1.5 + i * w * 2 - w / 2, [vals.group0_tpr, vals.group0_fpr], w, color=C['blue'], alpha=alpha, label=f'{name}, A=0 (low prevalence)' if cond == COND[0] else None)
            ax.bar(x - w * 1.5 + i * w * 2 + w / 2, [vals.group1_tpr, vals.group1_fpr], w, color=C['orange'], alpha=alpha, label=f'{name}, A=1' if cond == COND[0] else None)
        ax.set_xticks(x); ax.set_xticklabels(['TPR', 'FPR'])
        ax.set_title(COND_LABEL[cond], fontsize=10); ax.set_ylim(0, 1)
    axes[0].set_ylabel('rate on matched test set')
    axes[0].legend(fontsize=7, frameon=False, loc='upper right')
    fig.suptitle('Adult: per-group error rates at the Youden threshold, ERM (faded) vs FNF γ=1 (solid)', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def cross_condition_heatmap(df, fname):
    d = df[(df.dataset == 'adult') & (df.feature_set == 'paper6') & ((df.gamma == 1.0) | df.is_erm)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, (is_erm, title) in zip(axes, [(True, 'ERM'), (False, 'FNF γ=1')]):
        s = d[d.is_erm == is_erm].groupby(['condition', 'test_condition'])['eodds'].mean().unstack().loc[COND, COND]
        im = ax.imshow(s.values, cmap='Blues', vmin=0, vmax=.7)
        ax.set_xticks(range(4)); ax.set_xticklabels([COND_LABEL[c] for c in COND], rotation=30, ha='right')
        ax.set_yticks(range(4)); ax.set_yticklabels([COND_LABEL[c] for c in COND])
        ax.set_xlabel('test prevalence condition'); ax.set_title(f'{title}: equalized odds gap', fontsize=10)
        ax.grid(False)
        if not is_erm:
            ax.set_yticklabels([])
        for i in range(4):
            for j in range(4):
                v = s.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white' if v > .4 else C['ink'])
    axes[0].set_ylabel('training prevalence condition')
    fig.colorbar(im, ax=axes, shrink=.8, label='EOdds gap')
    fig.savefig(FIG / fname, dpi=160, bbox_inches='tight')
    plt.close(fig)


def auroc_vs_probe(df, fname, dataset='adult', feature_set='paper6', probe='marginal_max_probe_auc'):
    """One dot per fit (matched test condition): task AUROC against decodability of A, one panel per prevalence."""
    d = matched(df)
    d = d[(d.dataset == dataset) & (d.feature_set == feature_set)]
    gammas = sorted(d.loc[~d.is_erm, 'gamma'].dropna().unique())
    ramp = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
    step = max(1, len(ramp) // len(gammas))
    gcolor = {g: ramp[min(i * step + (len(ramp) - step * len(gammas)), len(ramp) - 1)] for i, g in enumerate(gammas)}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharex=True, sharey=True)
    for ax, cond in zip(axes, COND):
        s = d[d.condition == cond]
        for g in gammas:
            q = s[(~s.is_erm) & (s.gamma == g)]
            ax.scatter(q[probe], q.auroc, s=26, color=gcolor[g], edgecolor='white', lw=.5, label=f'FNF γ={g:g}' if cond == COND[0] else None, zorder=3)
        e = s[s.is_erm]
        ax.scatter(e[probe], e.auroc, s=80, marker='X', color=C['orange'], edgecolor='white', lw=.8, label='ERM' if cond == COND[0] else None, zorder=4)
        ax.axvline(.5, color='#b5b5b0', lw=1, ls='--')
        ax.set_title(COND_LABEL[cond], fontsize=10)
        ax.set_xlabel('strongest probe AUROC for A (marginal)')
    axes[0].set_ylabel('task AUROC (matched test set)')
    axes[0].legend(fontsize=7.5, frameon=False, loc='lower right', ncol=2)
    fig.suptitle(f'{dataset.capitalize()}, {feature_set}: every fit, task AUROC against decodability of A (dashed: chance)', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def summary_tables(df):
    d = matched(df)
    cols = ['marginal_max_probe_auc', 'conditional_max_probe_auc', 'paper_adversary_marginal_test_balanced_accuracy',
            'paper_adversary_Y1_test_balanced_accuracy', 'dp', 'eopp', 'eodds', 'fixed05_dp', 'fixed05_eodds', 'auroc', 'accuracy',
            'group0_tpr', 'group1_tpr', 'group0_fpr', 'group1_fpr']
    out = d.groupby(['dataset', 'feature_set', 'condition', 'method', 'gamma', 'prior'], dropna=False)[cols].agg(['mean', 'std', 'count'])
    out.columns = ['_'.join(c) for c in out.columns]
    out.round(4).to_csv(ROOT / 'summary_by_gamma.csv')
    return out


if __name__ == '__main__':
    df = load()
    summary_tables(df)
    gamma_panels(df, 'adult', 'paper6', 'Adult income, paper6 features, categorical FNF (MADE priors, exact TV)', 'adult_paper6_gamma_sweep.png', gamma_scale='ordinal')
    gamma_panels(df, 'adult', 'paper6_age', 'Adult income, paper6 + binned age, categorical FNF', 'adult_paper6_age_gamma_sweep.png', gamma_scale='ordinal')
    gamma_panels(df, 'health', 'core', 'Heritage Health, core features (14), continuous FNF, GMM priors', 'health_core_gmm_gamma_sweep.png', prior='gmm', gamma_scale='ordinal')
    gamma_panels(df, 'health', 'core', 'Heritage Health, core features (14), continuous FNF, flow priors', 'health_core_flow_gamma_sweep.png', prior='flow', gamma_scale='ordinal')
    gamma_panels(df, 'health', 'extended', 'Heritage Health, extended features (54), continuous FNF, both priors', 'health_extended_gamma_sweep.png', gamma_scale='ordinal')
    tension_scatter(df, 'dp_vs_eodds_scatter.png')
    rate_bars(df, 'adult_group_rates.png')
    cross_condition_heatmap(df, 'adult_eodds_cross_condition.png')
    auroc_vs_probe(df, 'adult_paper6_auroc_vs_probe.png')
    auroc_vs_probe(df, 'adult_paper6_age_auroc_vs_probe.png', feature_set='paper6_age')
    print('figures written to', FIG)
