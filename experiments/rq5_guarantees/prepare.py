"""Prepare leakage-aware tabular cohorts. Downloads are explicit CLI actions."""
from pathlib import Path
import argparse,json,hashlib,urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
CACHE=Path('/path/to/results/rq5')

def assign_splits(df):
    splitter=GroupShuffleSplit(n_splits=1,train_size=.6,random_state=42)
    train,rest=next(splitter.split(df,groups=df.unit))
    rem=df.iloc[rest];val,other=next(GroupShuffleSplit(n_splits=1,train_size=.25,random_state=43).split(rem,groups=rem.unit))
    rem2=rem.iloc[other];probe,test=next(GroupShuffleSplit(n_splits=1,train_size=1/3,random_state=44).split(rem2,groups=rem2.unit))
    df=df.copy();df['split']='train';df.loc[rem.iloc[val].index,'split']='validation';df.loc[rem2.iloc[probe].index,'split']='probe_validation';df.loc[rem2.iloc[test].index,'split']='test'
    assert df.groupby('unit').split.nunique().max()==1
    return df

def synthetic(n=20000):
    rng=np.random.default_rng(42);a=rng.integers(0,2,n);y=(rng.random(n)<np.where(a==0,.2,.8)).astype(int)
    noisy=y^(rng.random(n)<.2).astype(int)
    df=pd.DataFrame(dict(x_measurement=noisy+rng.normal(0,.03,n),x_proxy=a+rng.normal(0,.2,n),x_noise=rng.normal(size=n),a=a,y=y,unit=np.arange(n).astype(str)))
    return assign_splits(df)

def adult():
    rawdir=CACHE/'raw/adult';rawdir.mkdir(parents=True,exist_ok=True)
    cols=['age','workclass','fnlwgt','education','education_num','marital','occupation','relationship','race','sex','capital_gain','capital_loss','hours','country','income'];frames=[]
    for name in ['adult.data','adult.test']:
        p=rawdir/name
        if not p.exists():urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/2/adult.zip',rawdir/'adult.zip') if not (rawdir/'adult.zip').exists() else None
        if not p.exists():
            import zipfile
            with zipfile.ZipFile(rawdir/'adult.zip') as z:p.write_bytes(z.read(name))
        frames.append(pd.read_csv(p,names=cols,skipinitialspace=True,skiprows=1 if name.endswith('test') else 0,na_values='?').dropna(subset=['income']))
    d=pd.concat(frames,ignore_index=True);d['a']=(d.sex=='Male').astype(int);d['y']=d.income.str.strip().str.rstrip('.').eq('>50K').astype(int)
    # Exact duplicate feature rows share a unit even if their labels differ.
    d['unit']=pd.util.hash_pandas_object(d[cols[:-1]],index=False).astype(str)
    features=d.drop(columns=['sex','income','fnlwgt','a','y','unit']).add_prefix('x_')
    return assign_splits(pd.concat([features,d[['a','y','unit']]],axis=1))

def acs(state,year):
    from folktables import ACSDataSource,ACSIncome
    raw=ACSDataSource(survey_year=str(year),horizon='1-Year',survey='person',root_dir=str(CACHE/'raw/acs')).get_data(states=[state],download=True)
    # Official ACSIncome eligibility; align household IDs before transforming features.
    d=raw[(raw.AGEP>16)&(raw.PINCP>100)&(raw.WKHP>0)&(raw.PWGTP>=1)].copy()
    features=d[[c for c in ACSIncome.features if c!='SEX']].copy()
    for c in ['COW','SCHL','MAR','OCCP','POBP','RELP','RAC1P']:features[c]=features[c].astype('string').fillna('Missing').astype(str)
    features=features.add_prefix('x_');features['a']=(d.SEX==1).astype(int);features['y']=(d.PINCP>50000).astype(int)
    features['unit']=str(year)+'_'+state+'_'+d.SERIALNO.astype(str);features=features.reset_index(drop=True)
    return features

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['synthetic','adult','acs'],default='synthetic');p.add_argument('--n',type=int,default=20000);p.add_argument('--state',default='CA');p.add_argument('--year',type=int,default=2018);p.add_argument('--target-state',default='MI');p.add_argument('--target-year',type=int,default=2019);args=p.parse_args()
    folder=CACHE/'prepared'/args.dataset;folder.mkdir(parents=True,exist_ok=True)
    if args.dataset=='synthetic':d=synthetic(args.n)
    elif args.dataset=='adult':d=adult()
    else:
        source=assign_splits(acs(args.state,args.year));geo=acs(args.target_state,args.year);geo['split']='geographic_test';temporal=acs(args.state,args.target_year);temporal['split']='temporal_test'
        d=pd.concat([source,geo,temporal],ignore_index=True)
    d['row_id']=np.arange(len(d));d.to_csv(folder/'cohort.csv',index=False)
    counts=d.groupby(['split','a']).y.agg(['count','sum','mean']).reset_index();counts.to_csv(folder/'support.csv',index=False)
    manifest={**vars(args),'rows':len(d),'cohort_sha256':hashlib.sha256((folder/'cohort.csv').read_bytes()).hexdigest(),'split_fractions':[.6,.1,.1,.2],'sex_mapping':'a=0 Female, a=1 Male; synthetic a is abstract group','unit':'ACS household; Adult exact-feature duplicate cluster; synthetic individual','features':[c for c in d if c.startswith('x_')],'survey_weights':'Not used: record-cohort experiment, not survey population estimate'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));print(counts.to_string(index=False))
if __name__=='__main__':main()
