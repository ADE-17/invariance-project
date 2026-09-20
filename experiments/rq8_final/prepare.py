"""Fixed patient cohorts; 50% female-positive deletion independently per split."""
import sys,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from concurrent.futures import ThreadPoolExecutor
from experiments.rq8_final.common import *

def shift50(d,seed):
    eligible=d.index[(d.a==0)&(d.y==1)].to_numpy()
    remove=np.random.default_rng(seed).choice(eligible,len(eligible)//2,replace=False)
    return d.drop(index=remove).copy(),remove

def prepare():
    out=BASE/'prepared'/PATHOLOGY;out.mkdir(parents=True,exist_ok=True)
    with open(out/'prepare.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (out/'complete.json').exists():
            d=pd.read_csv(DATA/'train.csv');d['row_id']=np.arange(len(d));raw=len(d)
            d=d[d.Sex.isin(['Female','Male']) & d['Frontal/Lateral'].eq('Frontal')].copy()
            d['unit']=d.Path.str.extract(r'(patient\d+)')[0]
            assert d.unit.notna().all() and d.groupby('unit').Sex.nunique().max()==1
            d['a']=d.Sex.map({'Female':0,'Male':1}).astype(int)
            d['uncertain']=d[PATHOLOGY].eq(-1)
            d['y']=d[PATHOLOGY].fillna(0).replace(-1,1).astype(int)
            d['image_path']=[str(DATA/('/'.join(p.split('/')[1:]))) for p in d.Path]
            with ThreadPoolExecutor(8) as pool:assert all(pool.map(lambda p:Path(p).is_file(),d.image_path))
            # Same master patient partition as RQ7; fixed across model seeds.
            patients=np.random.default_rng(42).permutation(sorted(d.unit.unique()))
            boundaries=[0,.6,.65,.7,.8,1.];support=[];allsets={};deletions=[]
            for i,s in enumerate(SPLITS):
                ids=patients[int(len(patients)*boundaries[i]):int(len(patients)*boundaries[i+1])]
                original=d[d.unit.isin(ids)].copy();shifted,removed=shift50(original,20260914+i)
                removed_rows=original.loc[removed,['row_id','unit','a','y']].copy();removed_rows['split']=s;deletions.append(removed_rows)
                shifted=shifted.reset_index(drop=True);shifted.to_csv(out/f'{s}.csv',index=False);allsets[s]=shifted
                for a,g in shifted.groupby('a'):
                    before=original[original.a==a]
                    support.append(dict(pathology=PATHOLOGY,split=s,a=a,images=len(g),patients=g.unit.nunique(),positives=int(g.y.sum()),prevalence=g.y.mean(),
                        before_images=len(before),before_positives=int(before.y.sum()),before_prevalence=before.y.mean(),removed_positive_images=int(before.y.sum()-g.y.sum())))
            for s in SPLITS:
                for t in SPLITS:
                    if s!=t:assert set(allsets[s].unit).isdisjoint(allsets[t].unit)
                assert all(allsets[s].groupby('a').y.nunique()==2)
            pd.DataFrame(support).to_csv(out/'support.csv',index=False);pd.concat(deletions).to_csv(out/'removed_rows.csv',index=False)
            write(out/'complete.json',dict(pathology=PATHOLOGY,condition='shift50',raw_images=raw,master_patients=len(patients),
                label_policy='U-ones; blank negative',partition_seed=42,shift_seed=20260914,shift_rule='remove floor(n/2) female-positive images in each split',
                metadata_sha256=sha(DATA/'train.csv'),hashes={s:sha(out/f'{s}.csv') for s in SPLITS}))
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'prepared').exists():(ROOT/'prepared').symlink_to(out)
    assert (ROOT/'prepared').resolve()==out.resolve()
    log(f'{PATHOLOGY} / seed {SEED}: shifted cohort ready')

if __name__=='__main__':prepare()
