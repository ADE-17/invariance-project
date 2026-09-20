"""One configuration per job. Never fit or select on test labels."""
import sys, time, argparse, copy, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.mixture import GaussianMixture
import joblib
from experiments.rq8_final.common import *
from experiments.rq8_final.methods import *

def arrays(model,xs,device,sample=False):
    model.eval();zs={};ps={}
    with torch.no_grad():
        for s,x in xs.items():
            z=[];p=[]
            for start in range(0,len(x),2048):
                b=torch.from_numpy(np.asarray(x[start:start+2048]).copy()).to(device)
                zz,ll,_=model(b,sample=sample);z.append(zz.cpu().numpy());p.append(ll.sigmoid().cpu().numpy())
            zs[s]=np.concatenate(z);ps[s]=np.concatenate(p)
    return zs,ps

def head_fit(zs,sets,out,mlp=False,device='cpu',smoke=False):
    scaler=StandardScaler().fit(zs['train']);xs={s:scaler.transform(x).astype('float32') for s,x in zs.items()}
    joblib.dump(scaler,out/'head_scaler.joblib')
    if not mlp:
        candidates=[]
        for c in [.01,.1,1.]:
            m=LogisticRegression(C=c,max_iter=2000).fit(xs['train'],sets['train'].y)
            candidates.append((log_loss(sets['validation'].y,m.predict_proba(xs['validation'])[:,1]),m))
        _,model=min(candidates,key=lambda v:v[0]);joblib.dump(model,out/'head.joblib')
        return {s:model.predict_proba(x)[:,1].astype('float32') for s,x in xs.items()}
    model=nn.Sequential(nn.Linear(xs['train'].shape[1],128),nn.ReLU(),nn.Linear(128,1)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    tx=torch.from_numpy(xs['train']).to(device);ty=torch.tensor(sets['train'].y.to_numpy(),device=device).float()
    vx=torch.from_numpy(xs['validation']).to(device);best=float('inf');state=None;stale=0
    for ep in range(2 if smoke else 40):
        model.train()
        for ix in torch.randperm(len(tx),device=device).split(512):
            opt.zero_grad();loss=F.binary_cross_entropy_with_logits(model(tx[ix]).flatten(),ty[ix]);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():vl=log_loss(sets['validation'].y,model(vx).flatten().sigmoid().cpu().numpy())
        if vl<best:best=vl;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=7:break
    model.load_state_dict(state);torch.save(model.cpu(),out/'head.pt');model.to(device)
    with torch.no_grad():return {s:np.concatenate([model(torch.from_numpy(x[i:i+2048]).to(device)).flatten().sigmoid().cpu().numpy() for i in range(0,len(x),2048)]) for s,x in xs.items()}

def train_adapter(c,xs,sets,out,device,smoke):
    model=Adapter(xs['train'].shape[1],c['dim'],c['method']=='vib_mmd').to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    tx=torch.from_numpy(xs['train']).to(device);y=torch.tensor(sets['train'].y.to_numpy(),device=device).float()
    a=torch.tensor(sets['train'].a.to_numpy(),device=device)
    adversaries=nn.ModuleList([nn.Linear(c['dim'],1),nn.Sequential(nn.Linear(c['dim'],128),nn.ReLU(),nn.Linear(128,1)),
        nn.Sequential(nn.Linear(c['dim'],256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))]).to(device)
    ao=torch.optim.Adam(adversaries.parameters(),lr=.001)
    def advloss(z,ba):return torch.stack([.5*sum(F.binary_cross_entropy_with_logits(ad(z[ba==g]).flatten(),ba[ba==g].float()) for g in [0,1]) for ad in adversaries]).mean()
    criterion=hsic if c['method']=='hsic' else mmd
    history=[];best=float('inf');state=None;lam=0.;fallback=None;best_task=float('inf');started=time.time()
    ref=roc_auc_score(sets['validation'].y,np.load(ROOT/'baseline/validation_p.npy'))
    vi=np.random.default_rng(SEED).choice(len(sets['validation']),min(1024,len(sets['validation'])),replace=False)
    va=torch.tensor(sets['validation'].a.to_numpy()[vi],device=device)
    for ep in range(1,(2 if smoke else 50)+1):
        model.train();total=0;ltot=0;nt=0
        for ix in torch.randperm(len(tx),device=device).split(256):
            bx,by,ba=tx[ix],y[ix],a[ix]
            if min(int((ba==g).sum()) for g in [0,1])<2:continue
            if c['method']=='adversarial':
                for _ in range(5):
                    ao.zero_grad();zz=model(bx)[0].detach();al=advloss(zz,ba);al.backward();ao.step()
            opt.zero_grad();z,l,kl=model(bx,sample=c['method']=='vib_mmd');task=F.binary_cross_entropy_with_logits(l,by)
            if c['method']=='adversarial':
                for p in adversaries.parameters():p.requires_grad_(False)
                penalty=-advloss(z,ba)
            else:penalty=criterion(z,ba)
            if ep==1 and nt==0 and c['ratio']>0:
                pars=list(model.net.parameters())
                gt=torch.autograd.grad(task,pars,retain_graph=True)
                gp=torch.autograd.grad(penalty,pars,retain_graph=True)
                tn=torch.stack([v.square().sum() for v in gt]).sum().sqrt()
                pn=torch.stack([v.square().sum() for v in gp]).sum().sqrt()
                lam=float(c['ratio']*tn/pn.clamp_min(1e-8));write(out/'lambda.json',dict(lambda_value=lam,task_norm=float(tn),penalty_norm=float(pn)))
            loss=task+lam*min(ep/5,1.)*penalty+c.get('beta',0)*kl
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            for p in adversaries.parameters():p.requires_grad_(True)
            total+=float(task)*len(ix);ltot+=float(penalty)*len(ix);nt+=len(ix)
        vz,vp=arrays(model,{'validation':xs['validation']},device,sample=c['method']=='vib_mmd')
        auc=roc_auc_score(sets['validation'].y,vp['validation']);vl=log_loss(sets['validation'].y,vp['validation'])
        with torch.no_grad():dep=float(mmd(torch.from_numpy(vz['validation'][vi]).to(device),va))
        if vl<best_task:best_task=vl;fallback=copy.deepcopy(model.state_dict())
        # Plain ERM selects task loss; erasers select discrepancy only within the fixed task budget.
        score=vl if c['method']=='erm_adapter' else dep
        if auc>=.9*ref and score<best and ep>=min(5,2 if smoke else 5):
            best=score;state=copy.deepcopy(model.state_dict());selected=ep
        history.append(dict(epoch=ep,train_bce=total/nt,train_penalty=ltot/nt,validation_bce=vl,validation_auroc=auc,validation_mmd=dep,elapsed_seconds=time.time()-started))
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if ep%5==0:log(f'{c["id"]}: epoch {ep} val AUROC {auc:.4f} MMD {dep:.4f}')
    fallback_used=state is None
    model.load_state_dict(fallback if fallback_used else state)
    torch.save(model.cpu(),out/'model.pt');torch.save(adversaries.cpu().state_dict(),out/'training_adversaries.pt');model.to(device)
    zs,ps=arrays(model,xs,device,sample=c['method']=='vib_mmd')
    if c['method']=='vib_mmd':
        mean_z,mean_p=arrays(model,xs,device,sample=False)
        for s in SPLITS:np.save(out/f'{s}_mean_z.npy',mean_z[s]);np.save(out/f'{s}_mean_p.npy',mean_p[s])
        # Many independent observations: mean representation over 100 releases (sufficient for Gaussian mean attacks).
        with torch.no_grad():
            for s,x in xs.items():
                blocks=[]
                for st in range(0,len(x),2048):
                    h=model.net(torch.from_numpy(x[st:st+2048]).to(device));mu,lv=h.chunk(2,1)
                    blocks.append((mu+torch.randn_like(mu)*torch.exp(.5*lv.clamp(-8,4))/10).cpu().numpy())
                np.save(out/f'{s}_repeat100_z.npy',np.concatenate(blocks))
    return zs,ps,dict(fallback_used=fallback_used,selected_epoch=None if fallback_used else selected,actual_lambda=lam)

def train_lfr(c,xs,sets,out,device,smoke):
    tx=torch.from_numpy(xs['train']).to(device);y=torch.tensor(sets['train'].y.to_numpy(),device=device).float();a=torch.tensor(sets['train'].a.to_numpy(),device=device)
    model=LFR(tx,c['k']).to(device);opt=torch.optim.Adam(model.parameters(),lr=.003)
    state=None;best=float('inf');history=[];fallback=None;bt=float('inf')
    ref=roc_auc_score(sets['validation'].y,np.load(ROOT/'baseline/validation_p.npy'))
    def pred(s):
        with torch.no_grad():
            z,p,_=model(torch.from_numpy(xs[s]).to(device))
        return z.cpu().numpy(),p.cpu().numpy()
    for ep in range(1,(2 if smoke else 60)+1):
        for ix in torch.randperm(len(tx),device=device).split(512):
            z,p,recon=model(tx[ix]); ba=a[ix]
            if min(int((ba==g).sum()) for g in [0,1])<2:continue
            parity=(z[ba==0].mean(0)-z[ba==1].mean(0)).abs().mean()
            loss=F.binary_cross_entropy(p,y[ix])+.01*F.mse_loss(recon,tx[ix])+c['az']*parity
            opt.zero_grad();loss.backward();opt.step()
        z,p=pred('validation');auc=roc_auc_score(sets['validation'].y,p);vl=log_loss(sets['validation'].y,p)
        aa=sets['validation'].a.to_numpy();dep=float(abs(z[aa==0].mean(0)-z[aa==1].mean(0)).mean())
        if vl<bt:bt=vl;fallback=copy.deepcopy(model.state_dict())
        if auc>=.9*ref and dep<best:best=dep;state=copy.deepcopy(model.state_dict())
        history.append(dict(epoch=ep,validation_auroc=auc,validation_bce=vl,validation_membership_gap=dep))
    used=state is None;model.load_state_dict(fallback if used else state);torch.save(model.cpu(),out/'model.pt');model.to(device)
    zps={s:pred(s) for s in SPLITS};pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    return {s:v[0] for s,v in zps.items()},{s:v[1] for s,v in zps.items()},dict(fallback_used=used)

def train_fnf(c,xs,sets,out,device,smoke):
    priors=[];infos=[]
    for g in [0,1]:
        x=xs['train'][sets['train'].a.to_numpy()==g]
        gm=GaussianMixture(n_components=4,covariance_type='full',reg_covar=1e-3,max_iter=200,random_state=SEED).fit(x)
        joblib.dump(gm,out/f'prior{g}.joblib');priors.append(GMMPrior(gm,device))
        infos.append(dict(group=g,converged=gm.converged_,train_nll=-gm.score(x)))
    masks=alternating_masks(c['dim'],4,np.random.default_rng(SEED))
    flows=nn.ModuleList([Flow(c['dim'],[128,128],4,masks) for _ in [0,1]]).to(device)
    flows[1].load_state_dict(flows[0].state_dict());head=nn.Sequential(nn.Linear(c['dim'],64),nn.ReLU(),nn.Linear(64,1)).to(device)
    opt=torch.optim.AdamW(list(flows.parameters())+list(head.parameters()),lr=.001,weight_decay=.0001)
    tx=torch.from_numpy(xs['train']).to(device);y=torch.tensor(sets['train'].y.to_numpy(),device=device).float();a=torch.tensor(sets['train'].a.to_numpy(),device=device)
    def predict(s):
        with torch.no_grad():
            z=np.zeros_like(xs[s]);p=np.zeros(len(z),dtype='float32')
            for st in range(0,len(z),2048):
                bx=torch.from_numpy(xs[s][st:st+2048]).to(device);ba=sets[s].a.to_numpy()[st:st+2048]
                zz=torch.empty_like(bx)
                for g in [0,1]:
                    ix=torch.tensor(ba==g,device=device)
                    if ix.any():zz[ix]=flows[g].encode(bx[ix])[0]
                z[st:st+len(bx)]=zz.cpu().numpy();p[st:st+len(bx)]=head(zz).flatten().sigmoid().cpu().numpy()
        return z,p
    state=None;fallback=None;best=float('inf');bt=float('inf');history=[];start=time.time()
    ref=roc_auc_score(sets['validation'].y,np.load(ROOT/'baseline/validation_p.npy'))
    for ep in range(1,(2 if smoke else 60)+1):
        gam=c['gamma']*min(ep/10,1);ks=[]
        for ix in torch.randperm(len(tx),device=device).split(256):
            bx,ba=tx[ix],a[ix];zz=torch.empty_like(bx)
            for g in [0,1]:
                m=ba==g
                if m.any():zz[m]=flows[g].encode(bx[m])[0]
            pred=F.binary_cross_entropy_with_logits(head(zz).flatten(),y[ix])
            kl=symmetric_kl(flows,priors,128 if not smoke else 32)
            loss=(1-gam)*pred+gam*kl/c['dim']
            if not torch.isfinite(loss):raise RuntimeError('FNF divergence')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(flows.parameters())+list(head.parameters()),5);opt.step();ks.append(float(kl))
        z,p=predict('validation');auc=roc_auc_score(sets['validation'].y,p);vl=log_loss(sets['validation'].y,p)
        ids=np.random.default_rng(SEED).choice(len(z),min(1024,len(z)),replace=False)
        with torch.no_grad():dep=float(mmd(torch.tensor(z[ids],device=device),torch.tensor(sets['validation'].a.to_numpy()[ids],device=device)))
        st=lambda:dict(flows=copy.deepcopy(flows.state_dict()),head=copy.deepcopy(head.state_dict()))
        if vl<bt:bt=vl;fallback=st()
        if auc>=.9*ref and dep<best and ep>=min(10,2 if smoke else 10):best=dep;state=st();selected=ep
        history.append(dict(epoch=ep,gamma_effective=gam,train_kl=np.mean(ks),validation_auroc=auc,validation_bce=vl,validation_mmd=dep,elapsed_seconds=time.time()-start))
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if ep%5==0:log(f'{c["id"]}: epoch {ep} AUC {auc:.4f} KL {np.mean(ks):.3f}')
    used=state is None;state=fallback if used else state;flows.load_state_dict(state['flows']);head.load_state_dict(state['head'])
    torch.save(dict(flows=flows.cpu(),head=head.cpu()),out/'model.pt');flows.to(device);head.to(device)
    zp={s:predict(s) for s in SPLITS}
    return {s:v[0] for s,v in zp.items()},{s:v[1] for s,v in zp.items()},dict(fallback_used=used,priors=infos,requires_sex_at_inference=True,
        jacobian_convention='log pz=log px-logdet(encode)',kl_divided_by_dimension=True)

def run(c,smoke=False):
    out=ROOT/'runs'/c['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    torch.manual_seed(SEED);np.random.seed(SEED);start=time.time()
    device='cuda' if torch.cuda.is_available() else 'cpu';sets=frames();raw=load_features()
    if smoke:
        sets={s:d.iloc[:min(320,len(d))].copy() for s,d in sets.items()};raw={s:np.array(x[:len(sets[s])]) for s,x in raw.items()}
    write(out/'config.json',dict(**c,smoke=smoke,utility_budget_relative_auroc_loss=.1))
    scaler=StandardScaler().fit(raw['train']);xs={s:scaler.transform(x).astype('float32') for s,x in raw.items()};joblib.dump(scaler,out/'scaler.joblib')
    method=c['method'];meta={}
    if method=='constant_control':
        ps={s:np.full(len(sets[s]),sets['train'].y.mean(),dtype='float32') for s in SPLITS}
        zs={s:p[:,None] for s,p in ps.items()};meta=dict(control=True,expected_task_collapse=True)
    elif method=='raw_erm':
        zs={s:np.asarray(x) for s,x in raw.items()};ps={s:np.load(ROOT/'baseline'/f'{s}_p.npy')[:len(sets[s])] for s in SPLITS}
    elif method=='score_release':
        base={s:np.load(ROOT/'baseline'/f'{s}_p.npy')[:len(sets[s])] for s in SPLITS}
        eraser=ScoreRelease(c['bins'],c['aware']).fit(base['train'],sets['train'].y,sets['train'].a);joblib.dump(eraser,out/'eraser.joblib')
        zs={s:eraser.transform(base[s],sets[s].a,42+si) for si,s in enumerate(SPLITS)};ps={s:z[:,0] for s,z in zs.items()}
        for si,s in enumerate(SPLITS):
            repeats=eraser.transform(base[s],sets[s].a,1042+si,repeats=100)
            # Histograms and average: capture repeated-release distribution without an arbitrary column order.
            hist=np.stack([(repeats==q).mean(1) for q in eraser.q.astype('float32')],axis=1).astype('float32')
            np.save(out/f'{s}_repeat100_z.npy',hist)
            np.save(out/f'{s}_mean_z.npy',(eraser.probabilities(base[s],sets[s].a)@eraser.q)[:,None].astype('float32'))
        meta=dict(requires_sex_at_inference=c['aware'],train_tv=eraser.train_tv,scope='released_score_only')
    else:
        if method in ['fnf','pca_erm','gaussian_transport','lfr']:
            pca=PCA(n_components=c['dim'],random_state=SEED,svd_solver='randomized',whiten=True).fit(xs['train'])
            joblib.dump(pca,out/'pca.joblib');xs={s:pca.transform(x).astype('float32') for s,x in xs.items()}
        if method in ['erm_adapter','mmd','hsic','adversarial','vib_mmd']:zs,ps,meta=train_adapter(c,xs,sets,out,device,smoke)
        elif method=='lfr':zs,ps,meta=train_lfr(c,xs,sets,out,device,smoke)
        elif method=='fnf':zs,ps,meta=train_fnf(c,xs,sets,out,device,smoke)
        elif method in ['leace','rff_leace']:
            from concept_erasure import LeaceEraser
            if method=='rff_leace':
                rng=np.random.default_rng(SEED);w=rng.normal(size=(xs['train'].shape[1],c['dim']))/(np.sqrt(xs['train'].shape[1])*c['bw']);b=rng.uniform(0,2*np.pi,c['dim'])
                np.savez(out/'rff.npz',w=w,b=b)
                xs={s:(np.sqrt(2/c['dim'])*np.cos(x@w+b)).astype('float32') for s,x in xs.items()}
            eraser=LeaceEraser.fit(torch.from_numpy(xs['train']).double(),torch.tensor(sets['train'].a.to_numpy()).double()[:,None])
            torch.save(eraser,out/'eraser.pt');zs={s:eraser(torch.from_numpy(x).double()).float().numpy() for s,x in xs.items()}
            ps=head_fit(zs,sets,out,mlp=c.get('head')=='mlp',device=device,smoke=smoke)
        elif method=='pca_erm':zs=xs;ps=head_fit(zs,sets,out,device=device)
        elif method=='gaussian_transport':
            maps=[]
            for g in [0,1]:
                x=xs['train'][sets['train'].a.to_numpy()==g];mu=x.mean(0);cov=np.cov(x,rowvar=False)
                ev,u=np.linalg.eigh(cov);w=(u/np.sqrt(ev.clip(1e-4)))@u.T;maps.append((mu,w))
            joblib.dump(maps,out/'maps.joblib');zs={}
            for s,x in xs.items():
                z=np.zeros_like(x)
                for g,(mu,w) in enumerate(maps):m=sets[s].a.to_numpy()==g;z[m]=(x[m]-mu)@w
                zs[s]=z
            ps=head_fit(zs,sets,out,mlp=True,device=device,smoke=smoke);meta=dict(requires_sex_at_inference=True,scope='Gaussian moments only; independent nonlinear audit required')
        elif method in ['leopard','obliviator']:
            from experiments.rq8_final.recent import recent
            zs,ps,meta=recent(c,xs,sets,out,device,smoke)
        else:raise ValueError(method)
    for s in SPLITS:
        assert len(zs[s])==len(ps[s])==len(sets[s]) and np.isfinite(zs[s]).all() and np.isfinite(ps[s]).all()
        np.save(out/f'{s}_z.npy',np.asarray(zs[s],dtype='float32'));np.save(out/f'{s}_p.npy',np.asarray(ps[s],dtype='float32'))
    write(out/'complete.json',dict(**meta,elapsed_seconds=time.time()-start,seed=SEED,method=method,smoke=smoke,
         baseline_sha256=sha(ROOT/'baseline/best.pt') if (ROOT/'baseline/best.pt').exists() else 'synthetic_smoke'))
    log(f'Finished {c["id"]} in {time.time()-start:.1f}s')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    run(grid()[args.index],args.smoke)
