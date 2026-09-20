"""GPU integration tests on isolated synthetic artifacts, never research results."""
import os,sys,json,traceback
from pathlib import Path
worker=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
os.environ['RQ8_BASE']=f'/path/to/results/rq8/smoke_v1/worker{worker}'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from experiments.rq8_final.common import *
from experiments.rq8_final.train import run
from experiments.rq8_final.methods import grid

def main():
    (ROOT/'prepared').mkdir(parents=True,exist_ok=True);(ROOT/'baseline').mkdir(exist_ok=True)
    rng=np.random.default_rng(42)
    for s in SPLITS:
        n=320;a=rng.integers(0,2,n);x=rng.normal(size=(n,64)).astype('float32');x[:,0]+=a*2
        p=1/(1+np.exp(-x[:,1]));y=(rng.random(n)<p).astype(int)
        pd.DataFrame(dict(row_id=np.arange(n),unit=[f'{s}_{i}' for i in range(n)],a=a,y=y,uncertain=False,Age=50)).to_csv(ROOT/'prepared'/f'{s}.csv',index=False)
        np.save(ROOT/'baseline'/f'{s}_z.npy',x);np.save(ROOT/'baseline'/f'{s}_p.npy',p)
    unique={}
    for c in grid():unique.setdefault(c['method'],c)
    results=[]
    for i,c in enumerate(unique.values()):
        if i%3!=worker:continue
        try:
            run(c,smoke=True)
            if '--audit' in sys.argv:
                from experiments.rq8_final.evaluate import evaluate
                evaluate(c['index'],smoke=True)
            results.append(dict(method=c['method'],passed=True))
        except Exception as e:
            traceback.print_exc();results.append(dict(method=c['method'],passed=False,error=repr(e)))
    write(ROOT/('audit_smoke_results.json' if '--audit' in sys.argv else 'smoke_results.json'),results)
    if not all(r['passed'] for r in results):raise SystemExit(1)

if __name__=='__main__':main()
