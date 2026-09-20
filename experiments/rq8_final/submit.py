"""Launch six fixed-cohort seed shards from an immutable tested snapshot."""
import sys,shutil,subprocess,json,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.rq8_final.common import *
from experiments.rq8_final.methods import grid

def main():
    if (BASE/'submission.json').exists():
        print((BASE/'submission.json').read_text());return
    for worker in range(3):
        p=BASE.parent/'smoke_v1'/f'worker{worker}'/'Edema'/'seed42'/'audit_smoke_results.json'
        tests=json.loads(p.read_text());assert tests and all(r['passed'] for r in tests)
    for pathology in ['Edema','Cardiomegaly']:
        assert (BASE/'prepared'/pathology/'complete.json').exists()
    source=Path(__file__).resolve().parents[2];snap=BASE/'source_snapshot'
    assert not snap.exists(),'Do not overwrite immutable source snapshot'
    for name in ['rq8_final','rq6_fnf','rq5_sweep','rq5_removal','rq5_guarantees','rq4_ptbxl']:
        shutil.copytree(source/'experiments'/name,snap/'experiments'/name,ignore=shutil.ignore_patterns('__pycache__','*.ipynb'))
    shutil.copytree(source/'src',snap/'src',ignore=shutil.ignore_patterns('__pycache__'))
    subprocess.run([sys.executable,'-m','unittest','experiments.rq8_final.tests'],cwd=snap,check=True)
    write(BASE/'source_hashes.json',{str(p.relative_to(snap)):sha(p) for p in snap.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    configs=[dict(c,pathology=p,seed=s) for p in ['Edema','Cardiomegaly'] for s in [42,123,456] for c in grid()]
    write(BASE/'configurations.json',configs)
    (BASE/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    jid=subprocess.check_output(['sbatch','--parsable','--array=0-5%6',str(snap/'experiments/rq8_final/run.sbatch')],text=True).strip().split(';')[0]
    write(BASE/'submission.json',dict(submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sweep_job=jid,max_concurrent_gpus=6,cpus_per_job=8,memory_per_job='48G',time_limit_per_job='48h',erasure_fits=54,matched_erm_references=6,evaluations=60,seeds=[42,123,456],pathologies=['Edema','Cardiomegaly'],condition='shift50',source_snapshot=snap,smoke_job='564874'))
    collect=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{jid}',str(snap/'experiments/rq8_final/aggregate.sbatch')],text=True).strip().split(';')[0]
    sub=json.loads((BASE/'submission.json').read_text());sub['collection_job']=collect;write(BASE/'submission.json',sub)
    print((BASE/'submission.json').read_text())

if __name__=='__main__':main()
