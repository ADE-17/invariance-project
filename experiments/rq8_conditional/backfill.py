"""Add calibration diagnostics for saved RQ8 predictions; never retrain or recalibrate."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import BASE,write,sha
from .calibration import diagnostics

REFERENCE=Path('/path/to/results/rq8/shift50_v1')


def main():
    outputs=[]
    for f in sorted(REFERENCE.glob('*/seed*/runs/*/evaluation/*_metrics.json')):
        original=json.loads(f.read_text());c=original['config'];s=original['split']
        out=BASE/'reference_calibration'/c['pathology']/f'seed{c["seed"]}'/c['id']/f'{s}.json'
        if out.exists():outputs.append(str(out));continue
        d=pd.read_csv(f.parent/f'{s}_predictions.csv.gz',usecols=['unit','a','y','task_probability'])
        metrics,details=diagnostics(d.y.to_numpy(),d.task_probability.to_numpy(),d.a.to_numpy(),d.unit.to_numpy())
        # Check numerical agreement with previously saved mean bias and CITL.
        for g in (0,1):
            for key in [f'group{g}_calibration_bias',f'group{g}_citl']:
                if original['metrics'].get(key) is not None:
                    assert np.isclose(metrics[key],original['metrics'][key],atol=2e-5), (f,key)
        write(out,dict(config=c,split=s,metrics=metrics,weak_calibration=details,
                       source_metrics=str(f),source_sha256=sha(f),predictions_unchanged=True))
        outputs.append(str(out))
        print(f'Calibration {c["pathology"]} seed{c["seed"]} {c["id"]} {s}',flush=True)
    assert len(outputs)==120, f'Expected 60 RQ8 runs x 2 evaluation splits, got {len(outputs)}'
    write(BASE/'reference_calibration/complete.json',dict(rows=len(outputs),original_results_modified=False))


if __name__=='__main__':main()
