"""One pathology/seed per GPU; shared reference then nine train/audit runs."""
import os,sys,argparse,time,traceback
from pathlib import Path
shards=[(p,s) for p in ['Edema','Cardiomegaly'] for s in [42,123,456]]
ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
pathology,seed=shards[args.index];os.environ['RQ8_PATHOLOGY']=pathology;os.environ['RQ8_SEED']=str(seed)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.rq8_final.common import *
from experiments.rq8_final.methods import grid
from experiments.rq8_final.train import run
from experiments.rq8_final.evaluate import evaluate

def main():
    start=time.time()
    from experiments.rq8_final import baseline
    # Baseline has an independent CLI; make its argument parsing explicit.
    old=sys.argv;sys.argv=[old[0]]+(['--smoke'] if args.smoke else [])
    try:baseline.main()
    except Exception as e:
        write(ROOT/'baseline_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:sys.argv=old
    failures=[]
    for c in grid():
        stage='training';t=time.time()
        try:
            run(c,args.smoke);stage='evaluation';evaluate(c['index'],args.smoke)
            write(ROOT/'runs'/c['id']/'pipeline_complete.json',dict(pathology=PATHOLOGY,seed=SEED,elapsed_seconds=time.time()-t))
        except Exception as e:
            traceback.print_exc();failure=dict(id=c['id'],stage=stage,error=repr(e),traceback=traceback.format_exc());failures.append(failure)
            write(ROOT/'runs'/c['id']/'failure.json',failure)
    write(ROOT/'shard_summary.json',dict(pathology=PATHOLOGY,seed=SEED,elapsed_seconds=time.time()-start,failures=failures,expected_configurations=len(grid())))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
