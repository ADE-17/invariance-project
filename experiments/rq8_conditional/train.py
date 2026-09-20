"""RQ8 conditional extension; train-only priors; no outcome input at inference."""
import copy
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, log_loss
from .common import ROOT, SEED, SPLITS, frames, load_features, write, sha, log
from .methods import ConditionalFNF, GMMPrior, conditional_kl, mmd


def predict(model, x, a, device):
    """Pure inference API intentionally has no y or DataFrame argument."""
    model.eval()
    zs, ps = [], []
    with torch.no_grad():
        for start in range(0,len(x),2048):
            bx = torch.from_numpy(np.asarray(x[start:start+2048]).copy()).to(device)
            ba = torch.as_tensor(a[start:start+2048], device=device)
            z, logits = model(bx, ba)
            zs.append(z.cpu().numpy()); ps.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(zs), np.concatenate(ps)


def run(c, smoke=False, epochs=None):
    out = ROOT/'runs'/c['id']; out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists(): return
    start = time.time(); torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not smoke and device != 'cuda': raise RuntimeError('Production training requires allocated GPU')
    sets, raw = frames(), load_features()
    write(out/'config.json',dict(**c,smoke=smoke,utility_budget_relative_auroc_loss=.1,
                                selection='worst-label validation MMD',epochs=epochs or (2 if smoke else 60)))
    scaler = StandardScaler().fit(raw['train'])
    xs = {s:scaler.transform(x).astype('float32') for s,x in raw.items()}
    del raw
    pca = PCA(n_components=c['dim'],random_state=SEED,svd_solver='randomized',whiten=True).fit(xs['train'])
    joblib.dump(scaler,out/'scaler.joblib'); joblib.dump(pca,out/'pca.joblib')
    xs = {s:pca.transform(x).astype('float32') for s,x in xs.items()}
    priors, infos = [], []
    for label in (0,1):
        group_priors = []
        for group in (0,1):
            mask = (sets['train'].y.to_numpy()==label)&(sets['train'].a.to_numpy()==group)
            x = xs['train'][mask]
            if len(x)<20: raise ValueError(f'Insufficient prior support y={label}, a={group}')
            gm = GaussianMixture(n_components=4,covariance_type='full',reg_covar=1e-3,
                                 max_iter=200,random_state=SEED).fit(x)
            joblib.dump(gm,out/f'prior_y{label}_a{group}.joblib')
            group_priors.append(GMMPrior(gm,device))
            infos.append(dict(label=label,group=group,n=len(x),converged=bool(gm.converged_),
                              train_nll=float(-gm.score(x))))
        priors.append(group_priors)
    model = ConditionalFNF(c['dim'],SEED).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    tx=torch.from_numpy(xs['train']).to(device)
    ty=torch.tensor(sets['train'].y.to_numpy(),device=device).float()
    ta=torch.tensor(sets['train'].a.to_numpy(),device=device)
    vy=sets['validation'].y.to_numpy(); va=sets['validation'].a.to_numpy()
    rng=np.random.default_rng(SEED)
    # Equal group support per label prevents a rare positive group being missed.
    audit_ids={}
    for label in (0,1):
        idx=[np.flatnonzero((vy==label)&(va==g)) for g in (0,1)]
        n=min(256,*map(len,idx))
        if n<2: raise ValueError('Insufficient validation stratum support')
        audit_ids[label]=np.concatenate([rng.choice(v,n,replace=False) for v in idx])
    ref=roc_auc_score(vy,np.load(ROOT/'baseline/validation_p.npy'))
    best=float('inf'); best_bce=float('inf'); state=None; fallback=None
    selected=None; fallback_epoch=None; history=[]
    nepoch=epochs or (2 if smoke else 60)
    for ep in range(1,nepoch+1):
        model.train(); gam=c['gamma']*min(ep/10,1); losses=[]; bces=[]
        for ix in torch.randperm(len(tx),device=device).split(256):
            _, logits=model(tx[ix],ta[ix])
            task=F.binary_cross_entropy_with_logits(logits,ty[ix])
            kl, each=conditional_kl(model.flows,priors,32 if smoke else 128)
            loss=(1-gam)*task+gam*kl/c['dim']
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite conditional FNF objective')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step();losses.append([float(v.detach()) for v in each]);bces.append(float(task.detach()))
        z,p=predict(model,xs['validation'],va,device)
        auc=roc_auc_score(vy,p);bce=log_loss(vy,p)
        with torch.no_grad():
            deps=[float(mmd(torch.tensor(z[ids],device=device),torch.tensor(va[ids],device=device)))
                  for ids in audit_ids.values()]
        dep=max(deps)
        if bce<best_bce: best_bce=bce;fallback=copy.deepcopy(model.state_dict());fallback_epoch=ep
        if auc>=.9*ref and ep>=min(10,nepoch) and dep<best:
            best=dep;state=copy.deepcopy(model.state_dict());selected=ep
        history.append(dict(epoch=ep,gamma_effective=gam,train_bce=float(np.mean(bces)),
            train_conditional_kl_y0=float(np.mean(losses,axis=0)[0]),
            train_conditional_kl_y1=float(np.mean(losses,axis=0)[1]),validation_auroc=float(auc),
            validation_bce=float(bce),validation_conditional_mmd_y0=deps[0],
            validation_conditional_mmd_y1=deps[1],validation_worst_conditional_mmd=dep,
            elapsed_seconds=time.time()-start))
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if ep%5==0 or smoke: log(f'{c["id"]} epoch {ep}/{nepoch} AUC={auc:.4f} conditional MMD={deps}')
    used=state is None
    model.load_state_dict(fallback if used else state)
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=c),out/'model.pt')
    for s in SPLITS:
        z,p=predict(model,xs[s],sets[s].a.to_numpy(),device)
        assert len(z)==len(p)==len(sets[s]) and np.isfinite(z).all() and np.isfinite(p).all()
        np.save(out/f'{s}_z.npy',z.astype('float32'));np.save(out/f'{s}_p.npy',p.astype('float32'))
    write(out/'complete.json',dict(elapsed_seconds=time.time()-start,seed=SEED,method=c['method'],
        smoke=smoke,selected_epoch=fallback_epoch if used else selected,fallback_used=used,priors=infos,
        requires_sex_at_inference=True,requires_outcome_at_inference=False,
        objective='equal-label mean of half-symmetric conditional KL divided by dimension',
        baseline_sha256=sha(ROOT/'baseline/best.pt') if (ROOT/'baseline/best.pt').exists() else 'synthetic_smoke'))
