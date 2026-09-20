"""Single-replicate ECG ERM/DANN/CDANN trial, official folds, held-out audits."""
from pathlib import Path
import argparse, hashlib, json, sys, time, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.rq4_ptbxl.prepare import OUT,CACHE,META,WAVE,DISEASES,CLASSES,load,shifted
from src.evaluation.fairness_metrics import compute_all_fairness_metrics
from src.evaluation.task_metrics import select_threshold_on_validation
from src.models.gradient_reversal import GradientReversalFunction
from src.evaluation.calibration import compute_ece

class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); layers=[]; inc=12
        for c,k,s in [(32,11,2),(64,7,2),(128,5,2)]:
            layers += [nn.Conv1d(inc,c,k,stride=s,padding=k//2,bias=False),nn.BatchNorm1d(c),nn.ReLU()];inc=c
        self.net=nn.Sequential(*layers,nn.AdaptiveAvgPool1d(1),nn.Flatten())
    def forward(self,x):return self.net(x)

def mlp():return nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1),nn.Flatten(0))

def balanced_bce(logits,a):
    # Sex-balanced loss avoids majority-only adversary solutions.
    losses=[nn.functional.binary_cross_entropy_with_logits(logits[a==s],a[a==s].float()) for s in [0,1] if (a==s).any()]
    return sum(losses)/len(losses)

def cache_signals(df):
    import wfdb
    file=CACHE/'signals100.npy'; check=CACHE/'waveform_audit.json'
    meta_hash=hashlib.sha256((META/'ptbxl_database.csv').read_bytes()).hexdigest()
    if file.exists() and check.exists():
        audit=json.loads(check.read_text())
        assert audit['metadata_sha256']==meta_hash
        return np.load(file,mmap_mode='r')
    expected={}
    for l in (META/'SHA256SUMS.txt').read_text().splitlines():
        h,n=l.split(maxsplit=1);expected[n.lstrip('*').removeprefix('./')]=h
    x=np.empty((len(df),12,1000),dtype=np.float32); mismatches=[];seen={};duplicates=[]
    for i,row in enumerate(df.itertuples()):
        for ext in ['.hea','.dat']:
            rel=row.filename_lr+ext;p=WAVE/rel
            actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if actual!=expected.get(rel):mismatches.append(rel)
            if ext=='.dat':
                if actual in seen:duplicates.append([seen[actual],int(row.ecg_id)])
                seen[actual]=int(row.ecg_id)
        sig,info=wfdb.rdsamp(str(WAVE/row.filename_lr))
        assert info['fs']==100 and sig.shape==(1000,12) and np.isfinite(sig).all()
        x[i]=sig.T
        if i%5000==0:print('Loaded',i,flush=True)
    if mismatches:raise RuntimeError(f'Waveform checksum mismatches: {mismatches[:10]} ({len(mismatches)})')
    assert not duplicates, duplicates
    np.save(file,x)
    check.write_text(json.dumps(dict(metadata_sha256=meta_hash,records=len(df),checked_files=len(df)*2,checksum_mismatches=mismatches,duplicate_waveforms=duplicates,shape=list(x.shape),reader='wfdb.rdsamp',fs=100),indent=2))
    return x

def loader(x,df,batch=128,shuffle=False):
    idx=df.row_index.to_numpy()
    return DataLoader(TensorDataset(torch.from_numpy(np.asarray(x[idx]).copy()),torch.tensor(df.y.to_numpy(),dtype=torch.float32),torch.tensor(df.recorded_sex.to_numpy(),dtype=torch.long)),batch_size=batch,shuffle=shuffle)

@torch.no_grad()
def extract(enc,head,dl,device):
    enc.eval();head.eval();zs=[];ps=[]
    for x,y,a in dl:
        z=enc(x.to(device));zs.append(z.cpu().numpy());ps.append(torch.sigmoid(head(z).squeeze(-1)).cpu().numpy())
    return np.concatenate(zs),np.concatenate(ps)

def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None

def task_metrics(y,p):
    return dict(auroc=auc(y,p),auprc=float(average_precision_score(y,p)) if np.sum(y)>0 else None,brier=float(brier_score_loss(y,p)),ece=float(compute_ece(y,p)),n=int(len(y)),positive=int(np.sum(y)))

def fit_probes(ztr,atr,zv,av,ztest,device):
    """Fit on train, choose nonlinear epoch on validation; never fit on test."""
    if len(np.unique(atr))<2 or len(np.unique(av))<2:return None
    scaler=StandardScaler().fit(ztr);tr=scaler.transform(ztr).astype('float32');v=scaler.transform(zv).astype('float32')
    te={k:scaler.transform(z).astype('float32') for k,z in ztest.items()}
    linear=LogisticRegression(max_iter=1500,class_weight='balanced').fit(tr,atr)
    probe=mlp().to(device);opt=torch.optim.Adam(probe.parameters(),lr=1e-3,weight_decay=1e-4)
    dl=DataLoader(TensorDataset(torch.from_numpy(tr),torch.tensor(atr)),batch_size=256,shuffle=True)
    best=float('inf');state=None
    for epoch in range(30):
        probe.train()
        for x,a in dl:
            opt.zero_grad();loss=balanced_bce(probe(x.to(device)),a.to(device));loss.backward();opt.step()
        probe.eval()
        with torch.no_grad():vl=balanced_bce(probe(torch.from_numpy(v).to(device)),torch.tensor(av,device=device)).item()
        if vl<best:best=vl;state={k:t.detach().cpu().clone() for k,t in probe.state_dict().items()}
    probe.load_state_dict(state);probe.eval()
    result={}
    for k,z in te.items():
        with torch.no_grad():p=torch.sigmoid(probe(torch.from_numpy(z).to(device))).cpu().numpy()
        result[k]={'linear':linear.predict_proba(z)[:,1],'mlp':p}
    return result

def evaluate(enc,head,x,train,val,tests,device,run):
    train_z,_=extract(enc,head,loader(x,train),device);val_z,val_p=extract(enc,head,loader(x,val),device)
    test_data={k:extract(enc,head,loader(x,d),device) for k,d in tests.items()}
    threshold=float(select_threshold_on_validation(val.y.to_numpy(),val_p,criterion='youden'))
    probs={k:{} for k in tests}
    for label in [None,0,1]:
        tm=np.ones(len(train),dtype=bool) if label is None else train.y.to_numpy()==label
        vm=np.ones(len(val),dtype=bool) if label is None else val.y.to_numpy()==label
        masks={k:np.ones(len(d),dtype=bool) if label is None else d.y.to_numpy()==label for k,d in tests.items()}
        res=fit_probes(train_z[tm],train.recorded_sex.to_numpy()[tm],val_z[vm],val.recorded_sex.to_numpy()[vm],{k:z[masks[k]] for k,(z,p) in test_data.items()},device)
        if res is None:raise RuntimeError('Insufficient class/sex support for probe')
        for k in tests:
            for name,p in res[k].items():
                col=f'probe_{name}'+('' if label is None else f'_Y{label}')
                arr=np.full(len(tests[k]),np.nan);arr[masks[k]]=p;probs[k][col]=arr
    rows=[]
    for k,d in tests.items():
        z,p=test_data[k];y=d.y.to_numpy();a=d.recorded_sex.to_numpy();pred=(p>=threshold).astype(int)
        row={'test_condition':k,'threshold':threshold,**task_metrics(y,p),**compute_all_fairness_metrics(y,p,a,threshold)}
        group_rows=[]
        for group,g in [('sex',d.sex_name),('age',d.age_group),('sex_age',d.sex_name+' / '+d.age_group)]:
            for name in sorted(g.unique()):
                mask=(g==name).to_numpy();yg=y[mask];pg=p[mask];yp=pred[mask]
                entry={'stratification':group,'group':name,**task_metrics(yg,pg),'n_patients':int(d.loc[mask].patient_id.nunique()),'prevalence':float(yg.mean()),'tpr':float(yp[yg==1].mean()) if (yg==1).any() else None,'fpr':float(yp[yg==0].mean()) if (yg==0).any() else None,'ppr':float(yp.mean())}
                group_rows.append(entry)
        sex_metrics=[g for g in group_rows if g['stratification']=='sex'];row['worst_group_auroc']=min(g['auroc'] for g in sex_metrics)
        for col,pp in probs[k].items():
            mask=np.isfinite(pp);row[col+'_auroc']=auc(a[mask],pp[mask]);row[col+'_n']=int(mask.sum())
        # Pooled feature spread diagnoses degenerate near-constant encodings.
        row['feature_mean_std']=float(z.std(axis=0).mean())
        frame=d[['ecg_id','patient_id','sex_name','recorded_sex','age_group','y']].copy();frame['probability']=p
        for col,pp in probs[k].items():frame[col]=pp
        frame.to_csv(run/f'predictions_{k}.csv',index=False)
        pd.DataFrame(group_rows).to_csv(run/f'groups_{k}.csv',index=False)
        (run/f'eval_{k}.json').write_text(json.dumps(row,indent=2,allow_nan=False));rows.append(row)
    return rows

def run_one(args,target,condition,method,x,df):
    run=CACHE/'trials'/target/condition/(method if method=='erm' else f'{method}_lambda{args.lam:g}')
    run.mkdir(parents=True,exist_ok=True)
    result=run/'complete.json'
    if result.exists():print('Skipping completed',run,flush=True);return
    # One fixed RNG state for comparable trial initializations, not a seed sweep.
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data=df.copy();data['y']=data[target]
    natural_train=data[data.split=='train'];val=data[data.split=='validation'];test=data[data.split=='test']
    train=natural_train if condition=='natural' else shifted(natural_train,target)
    tests={'natural':test,'prevalence_shift':shifted(test,target)}
    for name,cohort in {'train':train,'validation':val,**tests}.items():
        cohort[['ecg_id','patient_id','row_index','y','recorded_sex','age_group']].to_csv(run/f'cohort_{name}.csv',index=False)
    # Same normalization across all methods/conditions, fitted only on natural train.
    mean=np.asarray(x[natural_train.row_index]).mean(axis=(0,2),keepdims=True)
    std=np.asarray(x[natural_train.row_index]).std(axis=(0,2),keepdims=True).clip(1e-6)
    xn=(np.asarray(x)-mean)/std
    enc=Encoder().to(device);head=nn.Linear(128,1).to(device)
    adv=nn.ModuleList([mlp() for _ in range(2 if method=='cdann' else 1)]).to(device)
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-4)
    aopt=torch.optim.Adam(adv.parameters(),lr=1e-3,weight_decay=1e-4)
    dl=loader(xn,train,shuffle=True);vdl=loader(xn,val)
    config={**vars(args),'target':target,'training_condition':condition,'method':method,'rng_seed':42,'device':device,'metadata_version':'1.0.3','n_train':len(train),'n_val':len(val),'normalization_mean':mean.ravel().tolist(),'normalization_std':std.ravel().tolist(),'checkpoint_rule':'minimum natural validation BCE among epochs >=6','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run/'config.json').write_text(json.dumps(config,indent=2));history=[];best=float('inf');best_epoch=None
    start=time.time()
    for epoch in range(1,args.epochs+1):
        enc.train();head.train();adv.train();tl=al=0;num=0;skipped=0
        alpha=0 if method=='erm' else args.lam*min((epoch-1)/5,1)
        for bx,y,a in dl:
            bx,y,a=bx.to(device),y.to(device),a.to(device)
            z=enc(bx);task=nn.functional.binary_cross_entropy_with_logits(head(z).squeeze(-1),y)
            adloss=torch.zeros((),device=device)
            if method!='erm':
                rev=GradientReversalFunction.apply(z,alpha);ls=[]
                for c,ad in enumerate(adv):
                    mask=torch.ones_like(y,dtype=torch.bool) if method=='dann' else y==c
                    if mask.any() and a[mask].unique().numel()==2:ls.append(balanced_bce(ad(rev[mask]),a[mask]))
                    else:skipped+=1
                if ls:adloss=sum(ls)/len(ls)
            opt.zero_grad();aopt.zero_grad();(task+adloss).backward();nn.utils.clip_grad_norm_(enc.parameters(),5);opt.step()
            if method!='erm':aopt.step()
            tl+=task.item()*len(y);al+=adloss.item()*len(y);num+=len(y)
        _,vp=extract(enc,head,vdl,device)
        clipped=np.clip(vp,1e-7,1-1e-7);yv=val.y.to_numpy();vl=float(-(yv*np.log(clipped)+(1-yv)*np.log(1-clipped)).mean())
        h=dict(epoch=epoch,alpha=alpha,train_bce=tl/num,adv_loss=al/num,val_bce=vl,val_auroc=auc(yv,vp),skipped_adversary_cells=skipped,elapsed_seconds=time.time()-start);history.append(h)
        assert np.isfinite(vl)
        if epoch>=6 and vl<best:
            best=vl;best_epoch=epoch
            torch.save({'encoder':enc.state_dict(),'head':head.state_dict(),'adversary':adv.state_dict(),'epoch':epoch,'alpha':alpha},run/'best.pt')
        print(target,condition,method,json.dumps(h),flush=True)
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
    torch.save({'encoder':enc.state_dict(),'head':head.state_dict(),'adversary':adv.state_dict(),'epoch':args.epochs,'alpha':alpha},run/'last.pt')
    saved=torch.load(run/'best.pt',map_location=device,weights_only=True);enc.load_state_dict(saved['encoder']);head.load_state_dict(saved['head'])
    # Probe on actual training cohort; no natural-train samples withheld by injection are reintroduced.
    rows=evaluate(enc,head,xn,train,val,tests,device,run)
    for row in rows:row.update(target=target,training_condition=condition,method=method,lambda_adv=0 if method=='erm' else args.lam,best_epoch=best_epoch,run_dir=str(run))
    result.write_text(json.dumps(rows,indent=2,allow_nan=False))
    print('COMPLETED',run,flush=True)

def aggregate():
    rows=[]
    for p in (CACHE/'trials').rglob('complete.json'):rows+=json.loads(p.read_text())
    if rows:pd.DataFrame(rows).to_csv(OUT/'trial_results.csv',index=False)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--lam',type=float,default=1.0);p.add_argument('--targets',nargs='+',default=DISEASES);p.add_argument('--methods',nargs='+',default=['erm','dann','cdann']);p.add_argument('--conditions',nargs='+',default=['natural','prevalence_shift']);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    assert args.epochs>=6;torch.set_num_threads(4)
    df=load();x=cache_signals(df)
    if args.prepare_only:return
    # No diagnostic statements: unknown diagnostic status, not a clean negative.
    df=df[df[CLASSES].sum(axis=1)>0].copy()
    for target in args.targets:
        for condition in args.conditions:
            for method in args.methods:
                run_one(args,target,condition,method,x,df);aggregate()
if __name__=='__main__':main()
