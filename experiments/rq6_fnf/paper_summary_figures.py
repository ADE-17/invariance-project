"""Paper summary figures for the marginal vs class-conditional invariance comparison (replaces / accompanies tab:fnf-results).

Figure A  fig_criteria_planes:  small multiples (rows = decodability plane, independence-separation plane, sufficiency plane;
                                columns = datasets). Points = seed means per method, faint dots = seeds, arrows ERM -> M-FNF and ERM -> C-FNF.
Figure B  fig_criteria_dots:    dot-plot grid (rows = criteria, columns = datasets, x = method); seeds as dots, mean as bar, ERM as dashed reference.
Figure C  fig_prevalence_gap:   EO gap and calibration-bias gap against the base-rate gap across the four prevalence conditions, per method.
All figures use the shift-50 condition for A and B and the matched test condition throughout. Output: results/rq6/paper/figures/summary_*.{png,pdf}.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path('/path/to/results/rq6/fnf_v1/runs')
FIG = Path(__file__).resolve().parents[2] / 'results' / 'rq6' / 'paper' / 'figures'
SEL = dict(adult='paper6_age', dutch='core6_hh', crime='socio4', credit='core5', compas='paper7')
TITLES = dict(adult='Adult', dutch='Dutch census', crime='Communities & Crime', credit='Taiwan credit', compas='COMPAS')
CONDS = ['equalized', 'natural', 'shift50', 'shift75']
METHODS = [('erm', 'ERM', '#6b6b66', 'o'), ('fnf1', r'$Z \perp A$', '#2a78d6', 's'),
           ('cy1', r'$Z \perp A \mid Y$', '#eb6834', '^'), ('cyhat1', r'$Z \perp A \mid \hat{Y}$', '#1baf7a', 'v')]
COLOR = {k: c for k, _, c, _ in METHODS}
MARK = {k: m for k, _, _, m in METHODS}
LABEL = {k: l for k, l, _, _ in METHODS}
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e2', 'grid.linewidth': .6, 'axes.edgecolor': '#b5b5b0',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b', 'figure.facecolor': 'white',
                     'legend.frameon': False, 'axes.titlesize': 13, 'axes.titleweight': 'normal'})


def run_name(ds, key):
    fs = SEL[ds]
    return {'erm': f'erm_{fs}', 'fnf1': f'fnf_{fs}_gamma1_made50',
            'cy1': f'cfnf_{fs}_cond-y_gamma1_made50', 'cyhat1': f'cfnf_{fs}_cond-yhat_gamma1_made50'}[key]


def collect():
    cal = pd.read_csv(FIG.parent / 'tables' / 'calibration_bias_all_runs.csv.gz')
    cal = cal[cal.condition == cal.test_condition].set_index(['dataset', 'seed', 'condition', 'run'])
    rows = []
    for ds in SEL:
        for seed in (42, 123, 456):
            for cond in CONDS:
                for key in COLOR:
                    folder = BASE / ds / f'seed{seed}' / cond / run_name(ds, key)
                    mp = folder / 'evaluation' / 'metrics.json'
                    if not mp.exists():
                        continue
                    r = [x for x in json.load(open(mp)) if x['test'] == f'test_{cond}'][0]
                    ppv, npv = {}, {}
                    for g in (0, 1):
                        tpr, fpr, ppr, pi = (r[f'group{g}_{k}'] for k in ['tpr', 'fpr', 'ppr', 'prevalence'])
                        ppv[g] = tpr * pi / ppr if ppr > 0 else np.nan
                        npv[g] = (1 - fpr) * (1 - pi) / (1 - ppr) if ppr < 1 else np.nan
                    c = cal.loc[(ds, seed, cond, folder.name)]
                    rows.append(dict(dataset=ds, seed=seed, condition=cond, model=key, auroc=r['auroc'],
                                     d_a=r['marginal_max_probe_auc'], d_ay=r['conditional_max_probe_auc'], dp=r['dp'], eo=r['eodds'],
                                     dppv=abs(ppv[1] - ppv[0]), dnpv=abs(npv[1] - npv[0]), ece_worst=r['max_group_ece'],
                                     dbias=abs(c['cal_mean_bias_gap']), dcal=abs(c['cal_intercept_fixed_slope_gap']),
                                     base_gap=abs(r['group1_prevalence'] - r['group0_prevalence'])))
    df = pd.DataFrame(rows)
    df.to_csv(FIG.parent / 'tables' / 'summary_figures_per_seed.csv', index=False)
    return df


def legend_handles(keys=('erm', 'fnf1', 'cy1', 'cyhat1')):
    return [Line2D([], [], marker=MARK[k], color=COLOR[k], ls='', ms=7, mec='white', mew=.6, label=LABEL[k]) for k in keys]


def fig_criteria_planes(df):
    d = df[df.condition == 'shift50']
    planes = [('d_a', 'd_ay', 'decodability of A\nfrom Z (marginal)', 'decodability of A\nwithin class', (.45, 1.0), (.45, 1.0), .5),
              ('dp', 'eo', 'demographic parity gap', 'equalized odds gap', (-.02, .68), (-.02, .68), 0),
              ('dppv', 'dbias', 'predictive parity gap (ΔPPV)', 'calibration bias gap (Δbias)', (-.02, .72), (-.02, .6), 0)]
    fig, axes = plt.subplots(3, 5, figsize=(15, 9.6))
    for j, ds in enumerate(SEL):
        for i, (xc, yc, xl, yl, xlim, ylim, ref) in enumerate(planes):
            ax = axes[i, j]
            g = d[d.dataset == ds]
            mean = g.groupby('model')[[xc, yc]].mean()
            for src, dst in [('erm', 'fnf1'), ('erm', 'cy1')]:
                x0, y0 = mean.loc[src]; x1, y1 = mean.loc[dst]
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='-|>', color=COLOR[dst], lw=1.1, shrinkA=6, shrinkB=6, alpha=.9))
            for k in COLOR:
                q = g[g.model == k]
                ax.scatter(q[xc], q[yc], s=10, color=COLOR[k], alpha=.35, lw=0, zorder=3)
                ax.scatter(mean.loc[k, xc], mean.loc[k, yc], s=58, color=COLOR[k], marker=MARK[k], ec='white', lw=.7, zorder=4)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            if ref:
                ax.axhline(ref, color='#b5b5b0', lw=.8, ls='--', zorder=1); ax.axvline(ref, color='#b5b5b0', lw=.8, ls='--', zorder=1)
                ax.text(.51, .455, 'chance', fontsize=9, color='#8a8a85', va='bottom')
            if i == 0:
                ax.set_title(TITLES[ds])
            if j == 0:
                ax.set_ylabel(yl)
            else:
                ax.set_yticklabels([])
            if i == 2 or True:
                ax.set_xlabel(xl if j == 2 or i != 99 else '')
    for i, (xc, yc, xl, yl, *_) in enumerate(planes):
        for j in range(5):
            axes[i, j].set_xlabel(xl if j == 2 else '')
    rows = ['a  What the representation encodes', 'b  Independence vs separation of the classifier', 'c  Sufficiency of the classifier']
    for i, t in enumerate(rows):
        axes[i, 0].text(-.42, 1.22 if i == 0 else 1.1, t, transform=axes[i, 0].transAxes, fontsize=13, fontweight='bold', color='#0b0b0b', ha='left')
    fig.legend(handles=legend_handles(), loc='lower center', ncol=4, bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .04, 1, .985))
    fig.subplots_adjust(hspace=.6)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'summary_criteria_planes.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def fig_criteria_dots(df):
    d = df[df.condition == 'shift50']
    rows = [('d_a', 'decodability of A\n(probe AUROC)', (.45, 1.0)), ('d_ay', 'decodability of A\nwithin class', (.45, 1.0)),
            ('dp', 'ΔDP\n(independence)', (0, None)), ('eo', 'ΔEO\n(separation)', (0, None)),
            ('dppv', 'ΔPPV\n(sufficiency, decision)', (0, None)), ('dbias', 'Δbias\n(sufficiency, score)', (0, None)), ('auroc', 'task AUROC', (.5, 1.0))]
    keys = list(COLOR)
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 14), sharey='row')
    for j, ds in enumerate(SEL):
        g = d[d.dataset == ds]
        for i, (col, lab, ylim) in enumerate(rows):
            ax = axes[i, j]
            erm = g[g.model == 'erm'][col].mean()
            ax.axhline(erm, color='#6b6b66', lw=.9, ls='--', zorder=1)
            for x, k in enumerate(keys):
                q = g[g.model == k][col]
                ax.hlines(q.mean(), x - .28, x + .28, color=COLOR[k], lw=2.6, zorder=3)
                ax.scatter(np.full(len(q), x) + np.linspace(-.1, .1, len(q)), q, s=16, color=COLOR[k], ec='white', lw=.5, zorder=4)
            if col in ('d_a', 'd_ay'):
                ax.axhline(.5, color='#b5b5b0', lw=.8, ls=':', zorder=1)
            ax.set_xlim(-.6, len(keys) - .4); ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(['ERM', r'$Z\perp A$', r'$Z\perp A|Y$', r'$Z\perp A|\hat Y$'] if i == len(rows) - 1 else [], fontsize=10)
            if ylim[0] is not None:
                ax.set_ylim(ylim[0], ylim[1])
            if i == 0:
                ax.set_title(TITLES[ds])
            if j == 0:
                ax.set_ylabel(lab)
    for i, (col, _, ylim) in enumerate(rows):
        if ylim[1] is None:
            axes[i, 0].set_ylim(0, d[col].max() * 1.08)
    fig.legend(handles=legend_handles() + [Line2D([], [], color='#6b6b66', ls='--', lw=1, label='ERM level')],
               loc='lower center', ncol=5, bbox_to_anchor=(.5, -.002))
    fig.tight_layout(rect=(0, .03, 1, .99))
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'summary_criteria_dots.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def fig_prevalence_gap(df):
    panels = [('eo', 'equalized odds gap'), ('dp', 'demographic parity gap'), ('dbias', 'calibration bias gap (Δbias)')]
    fig, axes = plt.subplots(len(panels), 5, figsize=(15, 8.4), sharey='row')
    for j, ds in enumerate(SEL):
        g = df[df.dataset == ds]
        for i, (col, lab) in enumerate(panels):
            ax = axes[i, j]
            for k in COLOR:
                m = g[g.model == k].groupby('condition').agg(x=('base_gap', 'mean'), y=(col, 'mean'), lo=(col, 'min'), hi=(col, 'max')).sort_values('x')
                ax.fill_between(m.x, m.lo, m.hi, color=COLOR[k], alpha=.12, lw=0)
                ax.plot(m.x, m.y, color=COLOR[k], lw=1.8, marker=MARK[k], ms=5.5, mec='white', mew=.6)
            ax.set_ylim(0, df[col].max() * 1.06)
            if i == 0:
                ax.set_title(TITLES[ds])
            if j == 0:
                ax.set_ylabel(lab)
            if i == len(panels) - 1 and j == 2:
                ax.set_xlabel('between-group base-rate gap  |P(Y=1 | A=1) − P(Y=1 | A=0)|   ')
    fig.legend(handles=legend_handles(), loc='lower center', ncol=4, bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .04, 1, .99))
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'summary_prevalence_gap.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    df = collect()
    print(df.groupby(['dataset', 'model']).size().unstack())
    fig_criteria_planes(df)
    fig_criteria_dots(df)
    fig_prevalence_gap(df)
    print('written', sorted(p.name for p in FIG.glob('summary_*')))
