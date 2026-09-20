"""One independent configuration per GPU, followed by its full held-out audit."""
import os
import sys
import argparse
import traceback
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True)
    args=ap.parse_args()
    jobs=[(p,s,i) for p in ['Edema','Cardiomegaly'] for s in [42,123,456] for i in range(3)]
    p,s,i=jobs[args.index]
    os.environ['RQ8C_PATHOLOGY']=p;os.environ['RQ8C_SEED']=str(s)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from experiments.rq8_conditional.common import ROOT,write
    from experiments.rq8_conditional.methods import grid
    from experiments.rq8_conditional.train import run
    from experiments.rq8_conditional.evaluate import evaluate
    import torch
    torch.set_num_threads(4)
    c=grid()[i];out=ROOT/'runs'/c['id'];out.mkdir(parents=True,exist_ok=True)
    start=time.time();stage='training'
    try:
        run(c);stage='evaluation';evaluate(i)
        write(out/'pipeline_complete.json',dict(pathology=p,seed=s,elapsed_seconds=time.time()-start))
    except Exception as exc:
        write(out/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
