"""Petersen-style score-distribution figure with continuous densities.
Rows: marginal score densities p(r | A) and class-conditional densities p(r | A, Y); columns: ERM, Z ⊥ A, Z ⊥ A | Y.
Colour = class Y (blue Y=0, orange Y=1; grey for the marginal row), line style = group (solid A=1, dashed A=0).
One figure per dataset (shift-50 condition, matched test split, seed 42). Output: results/rq6/paper/figures/score_densities_<ds>.{png,pdf}."""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path('/path/to/results/rq6/fnf_v1/runs')
FIG = Path(__file__).resolve().parents[2] / 'results/rq6/paper/figures'
SEL = dict(adult='paper6_age', dutch='core6_hh', crime='socio4', credit='core5', compas='paper7')
TITLES = dict(adult='Adult', dutch='Dutch census', crime='Communities & Crime', credit='Taiwan credit', compas='COMPAS')
METHODS = [('erm', 'ERM'), ('fnf1', r'$Z \perp A$'), ('cy1', r'$Z \perp A \mid Y$')]
CY = {0: '#2a78d6', 1: '#eb6834'}
GREY = '#3d3d3a'
LS = {0: (0, (4, 2.2)), 1: 'solid'}
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
                     'axes.edgecolor': '#8a8a85', 'figure.facecolor': 'white', 'legend.frameon': False})
COND, SEED = 'shift50', 42


def run_name(ds, key):
    fs = SEL[ds]
    return {'erm': f'erm_{fs}', 'fnf1': f'fnf_{fs}_gamma1_made50', 'cy1': f'cfnf_{fs}_cond-y_gamma1_made50'}[key]


def density(values, grid, bw=.06):
    """Gaussian KDE on [0, 1] with boundary reflection so mass at the edges is not lost."""
    v = np.asarray(values, float)
    if len(v) < 5:
        return np.zeros_like(grid)
    v = np.clip(v, 0, 1)
    kde = gaussian_kde(np.concatenate([v, -v, 2 - v]), bw_method=bw)
    return 3 * kde(grid)


def figure(ds):
    grid = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8), sharex=True)
    for j, (key, name) in enumerate(METHODS):
        folder = BASE / ds / f'seed{SEED}' / COND / run_name(ds, key)
        t = pd.read_csv(folder / 'evaluation' / f'predictions_test_{COND}.csv.gz', usecols=['a', 'y', 'task_probability'])
        m = [r for r in json.load(open(folder / 'evaluation' / 'metrics.json')) if r['test'] == f'test_{COND}'][0]
        r, a, y = t.task_probability.to_numpy(), t.a.to_numpy(), t.y.to_numpy()
        ax = axes[0, j]
        for g in (0, 1):
            ax.plot(grid, density(r[a == g], grid), color=GREY, ls=LS[g], lw=2.2)
        ax.set_title(name, pad=8)
        ax.text(.98, .95, f"$\\Delta$DP = {m['dp']:.2f}\n$D_A$ = {m['marginal_max_probe_auc']:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=13)
        ax = axes[1, j]
        for yy in (0, 1):
            for g in (0, 1):
                ax.plot(grid, density(r[(a == g) & (y == yy)], grid), color=CY[yy], ls=LS[g], lw=2.2)
        ax.text(.98, .95, f"$\\Delta$EO = {m['eodds']:.2f}\n$D_{{A\\mid Y}}$ = {m['conditional_max_probe_auc']:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=13)
        ax.axvline(m['threshold'], color='#8a8a85', lw=1, ls=':')
        axes[0, j].axvline(m['threshold'], color='#8a8a85', lw=1, ls=':')
        axes[1, j].set_xlabel('predicted risk $r$')
    for i, lab in enumerate([r'$p(r \mid A)$', r'$p(r \mid A, Y)$']):
        axes[i, 0].set_ylabel(lab)
        for j in range(3):
            axes[i, j].set_ylim(0, axes[i, j].get_ylim()[1] * 1.28)
            axes[i, j].set_yticks([]); axes[i, j].spines['left'].set_visible(False)
            axes[i, j].set_xlim(0, 1); axes[i, j].set_xticks([0, .5, 1])
    handles = [Line2D([], [], color=GREY, ls=LS[1], lw=2.2, label='$A = 1$'), Line2D([], [], color=GREY, ls=LS[0], lw=2.2, label='$A = 0$'),
               Line2D([], [], color=CY[0], lw=2.2, label='$Y = 0$'), Line2D([], [], color=CY[1], lw=2.2, label='$Y = 1$'),
               Line2D([], [], color='#8a8a85', lw=1, ls=':', label='threshold')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=14, bbox_to_anchor=(.5, -.02), handlelength=2.6, columnspacing=2)
    fig.tight_layout(pad=.4, h_pad=.8, w_pad=.6, rect=(0, .07, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'score_densities_{ds}.{ext}', dpi=220, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


if __name__ == '__main__':
    for ds in (sys.argv[1:] or SEL):
        figure(ds); print('written', ds)
