"""Post-hoc Cox calibration slope/intercept for every evaluated run (all test variants), from saved predictions.

Writes evaluation/calibration_bias.json per run and one aggregated table
results/rq6/paper/tables/calibration_bias_all_runs.csv.gz. Idempotent: existing per-run files are reused unless --force.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rq6_fnf.data import BASE, DATASETS
from experiments.rq6_fnf.calibration import group_calibration
from experiments.rq5_sweep.run import write_json

OUT = Path(__file__).resolve().parents[2] / 'results/rq6/paper/tables'


def run_rows(folder, force):
    target = folder / 'evaluation' / 'calibration_bias.json'
    if target.exists() and not force:
        return json.loads(target.read_text())
    config = json.loads((folder / 'config.json').read_text())
    rows = []
    for f in sorted((folder / 'evaluation').glob('predictions_test_*.csv.gz')):
        t = pd.read_csv(f, usecols=['a', 'y', 'task_probability'])
        rows.append(dict(dataset=config['dataset'], seed=config['seed'], condition=config['condition'], run=folder.name,
                         method=config['method'], variant=config.get('variant'), feature_set=config['feature_set'], gamma=config.get('gamma'),
                         made_hidden=config.get('made_hidden'), condition_on=config.get('condition_on'), test_condition=f.stem.removeprefix('predictions_test_').removesuffix('.csv'),
                         **group_calibration(t.y.to_numpy(), t.task_probability.to_numpy(), t.a.to_numpy())))
    write_json(target, rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='*', default=[d for d in DATASETS if d != 'health'])
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    rows = []
    for ds in args.datasets:
        folders = sorted(f.parent.parent for f in (BASE / 'runs' / ds).glob('seed*/*/*/evaluation/complete.json'))
        print(ds, len(folders), 'evaluated runs', flush=True)
        for folder in folders:
            rows.extend(run_rows(folder, args.force))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'calibration_bias_all_runs.csv.gz', index=False, compression='gzip')
    print('wrote', len(df), 'rows ->', OUT / 'calibration_bias_all_runs.csv.gz')


if __name__ == '__main__':
    main()
