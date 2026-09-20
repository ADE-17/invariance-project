"""Stitch per-run JSON into flat CSV tables under results/rq6. No plots, no narrative."""
from pathlib import Path
import json
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rq6_fnf.data import DATASETS, BASE

OUT = Path(__file__).resolve().parents[2] / 'results' / 'rq6'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, summaries, priors = [], [], []
    for f in sorted((BASE / 'runs').rglob('evaluation/metrics.json')):
        for row in json.loads(f.read_text()):
            row = dict(row)
            row.pop('gates', None)
            row['run_dir'] = str(f.parents[1])
            metrics.append(row)
    for f in sorted((BASE / 'runs').rglob('training_summary.json')):
        s = json.loads(f.read_text())
        flat = {k: v for k, v in s.items() if not isinstance(v, (list, dict))}
        flat['run_dir'] = str(f.parent)
        summaries.append(flat)
        for i, info in enumerate(s.get('prior_info', []) or []):
            priors.append(dict(run_dir=str(f.parent), group=i, **info))
    pd.DataFrame(metrics).to_csv(OUT / 'evaluation_metrics.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'training_summaries.csv', index=False)
    pd.DataFrame(priors).to_csv(OUT / 'prior_fits.csv', index=False)
    for d in DATASETS:
        c = BASE / 'prepared' / d / 'conditions.csv'
        if c.exists():
            pd.read_csv(c).to_csv(OUT / f'{d}_condition_prevalence.csv', index=False)
    print(len(metrics), 'metric rows;', len(summaries), 'training summaries ->', OUT)


if __name__ == '__main__':
    main()
