"""Deployment-shift figure and table: models trained under one prevalence condition, evaluated on test variants with a different
between-group base-rate gap (label shift within group: P(X | A, Y) fixed, P(Y | A) changed). Dutch and Adult, seed means.
Outputs: results/rq6/paper/figures/deployment_shift.{png,pdf}, deployment_shift_heatmaps.{png,pdf};
interpretation/paper_draft/fnf_deployment_shift_table.tex. Reads paper/tables/cross_condition_per_seed.csv."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'results/rq6/paper/figures'
df = pd.read_csv(ROOT / 'results/rq6/paper/tables/cross_condition_per_seed.csv')
TESTS = ['equalized', 'natural', 'shift50', 'shift75']
TL = {'equalized': 'equalized', 'natural': 'natural', 'shift50': 'shift 50', 'shift75': 'shift 75'}
METHODS = [('erm', 'ERM', '#6b6b66', 'o'), ('fnf1', 'M-FNF (Z ⊥ A)', '#2a78d6', 's'),
           ('cy1', 'C-FNF, oracle Y (Z ⊥ A | Y)', '#eb6834', '^'), ('cyhat1', 'C-FNF, predicted Ŷ', '#1baf7a', 'v')]
TITLES = {'dutch': 'Dutch census', 'adult': 'Adult'}
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 13, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e2',
                     'grid.linewidth': .6, 'axes.edgecolor': '#8a8a85', 'axes.linewidth': .8, 'xtick.color': '#0b0b0b', 'ytick.color': '#0b0b0b',
                     'figure.facecolor': 'white', 'legend.frameon': False})
PAPER = [('erm', 'ERM', '#6b6b66', 'o'), ('fnf1', r'$Z \perp A$', '#2a78d6', 's'), ('cy1', r'$Z \perp A \mid Y$', '#eb6834', '^')]
TRAIN = 'equalized'


def fig_lines():
    metrics = [('dp', r'$\Delta$DP', 'independence'), ('eo', r'$\Delta$EO', 'separation'), ('dbias', r'$\Delta_{\mathrm{bias}}$', 'sufficiency')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for i, ds in enumerate(['dutch', 'adult']):
        d = df[(df.dataset == ds) & (df.train == TRAIN)]
        gaps = d.groupby('test').gap.mean()
        for j, (col, sym, crit) in enumerate(metrics):
            ax = axes[i, j]
            ax.axvspan(-.3, .3, color='#ececea', lw=0, zorder=0)
            for k, name, color, mk in PAPER:
                q = d[d.model == k].groupby('test')[col].agg(['mean', 'min', 'max']).loc[TESTS]
                x = np.arange(4)
                ax.fill_between(x, q['min'], q['max'], color=color, alpha=.15, lw=0)
                ax.plot(x, q['mean'], color=color, lw=2.4, marker=mk, ms=9, mec='white', mew=.8, label=name, zorder=3)
            ax.set_xticks(range(4))
            ax.set_xticklabels([f'{gaps[t]:.2f}\n{TL[t]}' for t in TESTS], fontsize=13)
            ax.set_xlim(-.4, 3.4)
            ax.tick_params(length=3, pad=3)
            if i == 0:
                ax.set_title(f'{sym}  ({crit})', pad=6, fontsize=16)
            if j == 0:
                ax.set_ylabel(TITLES[ds], fontsize=16)
            if i == 1 and j == 1:
                ax.set_xlabel('base-rate gap of the test population and its prevalence condition', fontsize=14, labelpad=6)
    for j in range(3):
        top = max(axes[i, j].get_ylim()[1] for i in range(2)) * (1.3 if j == 1 else 1.0)
        for i in range(2):
            axes[i, j].set_ylim(0, top)
            axes[i, j].set_yticks(np.arange(0, top - .04, .1))
            axes[i, j].tick_params(axis='y', labelsize=14)
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(h, l, loc='upper center', ncol=3, fontsize=14, handlelength=2, columnspacing=1.6, borderaxespad=.1, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(pad=.4, h_pad=.6, w_pad=.8)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'deployment_shift.{ext}', dpi=220, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def fig_heatmaps():
    cmap = LinearSegmentedColormap.from_list('blue', ['#fcfcfb', '#2a78d6'])
    panels = [('fnf1', 'dp', r'$Z \perp A$:  $\Delta$DP'), ('fnf1', 'eo', r'$Z \perp A$:  $\Delta$EO'), ('cy1', 'dp', r'$Z \perp A \mid Y$:  $\Delta$DP'), ('cy1', 'eo', r'$Z \perp A \mid Y$:  $\Delta$EO')]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for i, ds in enumerate(['dutch', 'adult']):
        for j, (k, col, title) in enumerate(panels):
            ax = axes[i, j]
            m = df[(df.dataset == ds) & (df.model == k)].groupby(['train', 'test'])[col].mean().unstack().loc[TESTS, TESTS]
            ax.imshow(m.to_numpy(), cmap=cmap, vmin=0, vmax=.45, aspect='auto')
            for r in range(4):
                for c in range(4):
                    v = m.iloc[r, c]
                    ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=11, color='white' if v > .28 else '#0b0b0b')
            ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.grid(False)
            ax.set_xticklabels([TL[t] for t in TESTS], rotation=30, ha='right')
            ax.set_yticklabels([TL[t] for t in TESTS] if j == 0 else [])
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f'{TITLES[ds]}\ntraining condition')
            if i == 1:
                ax.set_xlabel('test condition')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'deployment_shift_heatmaps.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def table():
    L = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llcccc}', r'\toprule',
         r' & & \multicolumn{4}{c}{Test population (base-rate gap)} \\', r'\cmidrule(lr){3-6}']
    for ds in ['dutch', 'adult']:
        d = df[(df.dataset == ds) & (df.train == TRAIN)]
        gaps = d.groupby('test').gap.mean()
        L.append('Dataset / metric & Method & ' + ' & '.join(f'{TL[t]} ({gaps[t]:.2f})' for t in TESTS) + r' \\')
        L.append(r'\midrule')
        for col, name in [('dp', r'$\Delta\mathrm{DP}$'), ('eo', r'$\Delta\mathrm{EO}$'), ('dbias', r'$\Delta_{\mathrm{bias}}$')]:
            for r, (k, mname, _, _) in enumerate([m for m in METHODS if m[0] != 'cyhat1']):
                q = d[d.model == k].groupby('test')[col].mean().loc[TESTS]
                short = {'erm': 'ERM', 'fnf1': 'M-FNF', 'cy1': 'C-FNF ($Y$)'}[k]
                L.append((f'{TITLES[ds]} {name}' if r == 0 else '') + f' & {short} & ' + ' & '.join(f'{v:.2f}' for v in q) + r' \\')
            L.append(r'\addlinespace[2pt]')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L += [r'\end{tabular}',
          r'\caption{\textbf{Fairness guarantees under deployment shift.} Models trained under the equalized prevalence condition (base-rate gap $\approx 0$) and evaluated on test populations whose between-group base-rate gap grows from left to right (seed means over three seeds; Youden threshold fixed on the training-condition validation split). The test populations differ from the training population only in $P(Y\mid A)$; $P(X\mid A,Y)$ is unchanged. Marginal FNF loses its parity guarantee as soon as the base rates move; class-conditional FNF keeps $\Delta\mathrm{EO}$ fixed at every gap while $\Delta\mathrm{DP}$ follows the base-rate gap. The calibration-bias gap grows for every method, including ERM.}',
          r'\label{tab:deployment-shift}', r'\end{table}']
    (ROOT / 'interpretation/paper_draft/fnf_deployment_shift_table.tex').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    fig_lines(); fig_heatmaps(); table()
    print('written')
