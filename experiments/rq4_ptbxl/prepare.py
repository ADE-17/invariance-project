"""Prepare official v1.0.3 metadata; audit prevalence without reading ECG signals."""
from pathlib import Path
import ast, hashlib, json, urllib.request
import numpy as np
import pandas as pd

ROOT=Path('.')
OUT=ROOT/'results/rq4'
CACHE=Path('/path/to/results/rq4')
META=CACHE/'metadata'
WAVE=Path('/path/to/ptb-xl')
CLASSES=['NORM','MI','STTC','CD','HYP']
DISEASES=CLASSES[1:]
AGE_BINS=['<40','40–59','60–79','80–89','90+ (deidentified)','Unknown']

def load():
    df=pd.read_csv(META/'ptbxl_database.csv')
    mapping=pd.read_csv(META/'scp_statements.csv',index_col=0)
    mapping=mapping[mapping.diagnostic==1].diagnostic_class.dropna().to_dict()
    codes=df.scp_codes.map(ast.literal_eval)
    for c in CLASSES:
        # Code presence, including likelihood zero (unknown), per official example.
        df[c]=codes.map(lambda x:int(any(mapping.get(k)==c for k in x)))
    assert df.sex.isin([0,1]).all()
    df['sex_name']=df.sex.map({0:'Male',1:'Female'})
    df['recorded_sex']=1-df.sex.astype(int) # shared fairness API: Female=0, Male=1
    def agebin(a):
        if pd.isna(a) or a<0:return AGE_BINS[-1]
        if a>=90:return AGE_BINS[4]
        if a<40:return AGE_BINS[0]
        if a<60:return AGE_BINS[1]
        if a<80:return AGE_BINS[2]
        return AGE_BINS[3]
    df['age_group']=df.age.map(agebin)
    df['split']=np.where(df.strat_fold<=8,'train',np.where(df.strat_fold==9,'validation','test'))
    assert df.groupby('patient_id').split.nunique().max()==1
    assert df.ecg_id.is_unique
    df['row_index']=np.arange(len(df))
    return df

def shifted(df,target,drop=0.5):
    """Remove 50% of Female positives, within existing split, no relabeling."""
    cand=df[(df.recorded_sex==0)&(df[target]==1)]
    # Stable recorded trial draw, shared by all methods. Not a multi-seed experiment.
    remove=cand.sample(n=int(len(cand)*drop),random_state=42).index
    return df.drop(remove)

def support(df,condition,target_context='all'):
    rows=[]
    for split in ['all','train','validation','test']:
        sub=df if split=='all' else df[df.split==split]
        for kind,cols in [('overall',[]),('sex',['sex_name']),('age',['age_group']),('sex_age',['sex_name','age_group'])]:
            gs=[((),sub)] if not cols else sub.groupby(cols,observed=True)
            for key,g in gs:
                key=key if isinstance(key,tuple) else (key,)
                names=dict(zip(cols,key))
                for c in CLASSES:
                    rows.append(dict(condition=condition,shift_target=target_context,split=split,stratification=kind,sex=names.get('sex_name','All'),age=names.get('age_group','All'),label=c,n_ecgs=len(g),n_patients=g.patient_id.nunique(),positive=int(g[c].sum()),negative=int(len(g)-g[c].sum()),prevalence=float(g[c].mean())))
    return rows

def main():
    META.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    for name in ['ptbxl_database.csv','scp_statements.csv','SHA256SUMS.txt','ptbxl_v103_changelog.txt','LICENSE.txt']:
        p=META/name
        if not p.exists():urllib.request.urlretrieve('https://physionet.org/files/ptb-xl/1.0.3/'+name,p)
    df=load();df.to_csv(OUT/'cohort.csv',index=False)
    rows=support(df,'natural')
    for target in DISEASES:
        # Validation always natural; training/test shifts are constructed independently.
        shifted_df=pd.concat([shifted(g,target) if s!='validation' else g for s,g in df.groupby('split')])
        rows+=support(shifted_df,'prevalence_shift',target)
        for split,g in shifted_df.groupby('split'):
            g[['ecg_id','patient_id','row_index',target,'recorded_sex','age_group']].to_csv(OUT/f'{target}_{split}_shift_manifest.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'prevalence_all_strata.csv',index=False)
    old=pd.read_csv(WAVE/'ptbxl_database.csv')
    audit=dict(metadata_version='1.0.3',waveform_source=str(WAVE),local_metadata_records=len(old),official_records=len(df),patients=int(df.patient_id.nunique()),removed_local_ecg_ids=sorted(set(old.ecg_id)-set(df.ecg_id)),age_missing=int(df.age.isna().sum()),age_ge_90=int((df.age>=90).sum()),no_diagnostic_superclass=int((df[CLASSES].sum(axis=1)==0).sum()),sex_mapping={'PTBXL_0':'Male','PTBXL_1':'Female','project_0':'Female','project_1':'Male'},metadata_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in META.iterdir()})
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(rows).query("condition=='natural' and split=='all' and stratification=='sex'") [['sex','label','n_ecgs','positive','prevalence']].to_string(index=False))
    print(json.dumps({k:v for k,v in audit.items() if k not in ['metadata_sha256','removed_local_ecg_ids']},indent=2))
if __name__=='__main__':main()
