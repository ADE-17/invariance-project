"""Isolated end-to-end GPU test plus one real-data training epoch."""
import os,sys,json,time
from pathlib import Path
os.environ['RQ8C_BASE']='/path/to/results/rq8/conditional_smoke_v1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd,torch
from experiments.rq8_conditional.common import ROOT,SPLITS,write
from experiments.rq8_conditional.methods import grid
from experiments.rq8_conditional.train import run
from experiments.rq8_conditional.evaluate import evaluate


def main():
    assert torch.cuda.is_available(),'GPU smoke must execute on an allocated GPU'
    torch.set_num_threads(4)
    (ROOT/'prepared').mkdir(parents=True,exist_ok=True);(ROOT/'baseline').mkdir(exist_ok=True)
    rng=np.random.default_rng(42)
    for s in SPLITS:
        n=320;a=np.arange(n)%2;x=rng.normal(size=(n,32)).astype('float32');x[:,0]+=a
        p=1/(1+np.exp(-x[:,1]));y=(rng.random(n)<p).astype(int)
        pd.DataFrame(dict(row_id=np.arange(n),unit=[f'{s}_{i//2}' for i in range(n)],a=a,y=y,
                          uncertain=False,Age=50)).to_csv(ROOT/'prepared'/f'{s}.csv',index=False)
        np.save(ROOT/'baseline'/f'{s}_z.npy',x);np.save(ROOT/'baseline'/f'{s}_p.npy',p)
    for c in grid():run(c,smoke=True)
    evaluate(1,smoke=True)
    d=json.loads((ROOT/'runs/02_conditional_fnf/evaluation/test_metrics.json').read_text())
    assert d['weak_calibration']['group0']['fixed_slope_converged']
    assert 'group0_calibration_bias_score' in d['metrics']
    write(ROOT/'smoke_complete.json',dict(passed=True,gpu=torch.cuda.get_device_name(0),
        test='three gamma training + full audit schema + patient-clustered calibration'))


if __name__=='__main__':main()
