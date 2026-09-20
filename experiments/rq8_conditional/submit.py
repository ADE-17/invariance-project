"""Preflight exact RQ8 inputs, freeze source, launch at most eight GPUs after smoke."""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .common import BASE,SPLITS,write,sha
from .backfill import REFERENCE


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke-job',required=True);args=ap.parse_args()
    assert args.smoke_job.isdigit()
    if (BASE/'submission.json').exists():
        print((BASE/'submission.json').read_text());return
    source=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,'-m','unittest','experiments.rq8_conditional.tests','-v'],cwd=source,check=True)
    manifest=[]
    for pathology in ['Edema','Cardiomegaly']:
        for seed in [42,123,456]:
            old=REFERENCE/pathology/f'seed{seed}';new=BASE/pathology/f'seed{seed}'
            new.mkdir(parents=True,exist_ok=True)
            assert (old/'baseline/best.pt').exists()
            patients={}
            for s in SPLITS:
                csv=old/'prepared'/f'{s}.csv';d=pd.read_csv(csv)
                z=np.load(old/'baseline'/f'{s}_z.npy',mmap_mode='r')
                p=np.load(old/'baseline'/f'{s}_p.npy')
                assert z.shape==(len(d),1024) and len(p)==len(d)
                assert np.isfinite(p).all() and np.isfinite(z[:100]).all()
                assert all(((d.a==g)&(d.y==y)).sum()>=2 for g in (0,1) for y in (0,1))
                patients[s]=set(d.unit)
                manifest.append(dict(pathology=pathology,seed=seed,split=s,rows=len(d),
                    prepared_sha256=sha(csv),baseline_prediction_sha256=sha(old/'baseline'/f'{s}_p.npy'),
                    feature_path=str(old/'baseline'/f'{s}_z.npy')))
            for i,s in enumerate(SPLITS):
                for t in SPLITS[i+1:]:assert not patients[s]&patients[t],(s,t,'patient overlap')
            for name in ['prepared','baseline']:
                link=new/name
                if not link.exists():link.symlink_to((old/name).resolve(),target_is_directory=True)
                assert link.resolve()==(old/name).resolve()
    write(BASE/'input_manifest.json',manifest)
    snap=BASE/'source_snapshot'
    if snap.exists():raise RuntimeError('Snapshot exists without submission manifest; inspect before resubmitting')
    shutil.copytree(REFERENCE/'source_snapshot',snap,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(source/'experiments/rq8_conditional',snap/'experiments/rq8_conditional',ignore=shutil.ignore_patterns('__pycache__'))
    subprocess.run([sys.executable,'-m','unittest','experiments.rq8_conditional.tests'],cwd=snap,check=True)
    write(BASE/'source_hashes.json',{str(p.relative_to(snap)):sha(p) for p in snap.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    (BASE/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    jobs=[dict(index=i,pathology=p,seed=s,gamma=g) for i,(p,s,g) in enumerate(
        (p,s,g) for p in ['Edema','Cardiomegaly'] for s in [42,123,456] for g in [.05,.5,.95])]
    write(BASE/'configurations.json',jobs)
    def submit(*args):return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0]
    # A failed GPU integration smoke blocks production automatically.
    jid=submit('--array=0-17%8',f'--dependency=afterok:{args.smoke_job}',str(snap/'experiments/rq8_conditional/run.sbatch'))
    submission=dict(submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sweep_job=jid,
        smoke_job=args.smoke_job,max_concurrent_gpus=8,cpus_per_job=8,memory_per_job='48G',
        erasure_fits=18,reused_erm_references=6,condition='shift50',source_snapshot=str(snap),
        pathology=['Edema','Cardiomegaly'],seeds=[42,123,456],gamma=[.05,.5,.95])
    write(BASE/'submission.json',submission)
    cal=submit(str(snap/'experiments/rq8_conditional/backfill.sbatch'))
    submission['calibration_job']=cal;write(BASE/'submission.json',submission)
    agg=submit(f'--dependency=afterany:{jid},afterok:{cal}',str(snap/'experiments/rq8_conditional/aggregate.sbatch'))
    submission['collection_job']=agg;write(BASE/'submission.json',submission)
    print(json.dumps(submission,indent=2))


if __name__=='__main__':main()
