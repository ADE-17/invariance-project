"""Auditable RQ8 objectives. Adaptations and references are in METHODOLOGY.md."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linprog
from experiments.rq6_fnf.fnf import Flow, FlowPrior, GMMPrior, alternating_masks, latent_log_density

def kernel(z):
    z=(z-z.mean(0))/z.std(0,unbiased=False).clamp_min(1e-5)
    dist=torch.cdist(z,z).square()
    vals=dist.detach()[dist.detach()>1e-10]
    bw=vals.median() if len(vals) else dist.new_tensor(1.)
    return sum(torch.exp(-dist/(2*bw.clamp_min(1e-6)*s)) for s in [.25,.5,1,2,4])/5

def mmd(z,a):
    k=kernel(z);u=a==0;v=a==1
    if min(int(u.sum()),int(v.sum()))<2:return z.sum()*0
    return k[u][:,u].mean()+k[v][:,v].mean()-2*k[u][:,v].mean()

def hsic(z,a):
    k=kernel(z);l=(a[:,None]==a[None,:]).float()
    def center(x):return x-x.mean(0,keepdim=True)-x.mean(1,keepdim=True)+x.mean()
    k,l=center(k),center(l)
    return (k*l).sum()/(k.norm()*l.norm()).clamp_min(1e-8)

def symmetric_kl(flows,priors,n):
    """log p_Z(f(x)) = log p_X(x) - log|det Df(x)| for encode x -> z."""
    loss=0
    for g in (0,1):
        x=priors[g].sample(n);z,logdet=flows[g].encode(x)
        own=priors[g].log_prob(x)-logdet
        other=latent_log_density(z,priors[1-g],flows[1-g])
        loss=loss+.5*(own-other).mean()
    return loss

class Adapter(nn.Module):
    def __init__(self,d,dim=32,variational=False):
        super().__init__();self.dim=dim;self.variational=variational
        self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,dim*(2 if variational else 1)))
        self.head=nn.Linear(dim,1)
    def forward(self,x,sample=False):
        h=self.net(x);kl=h.sum()*0
        if self.variational:
            mu,lv=h.chunk(2,1);lv=lv.clamp(-8,4)
            kl=.5*(mu.square()+lv.exp()-lv-1).sum(1).mean()
            z=mu+torch.randn_like(mu)*torch.exp(.5*lv) if sample else mu
        else:z=h
        return z,self.head(z).flatten(),kl

class LFR(nn.Module):
    """Zemel et al: soft prototype membership, reconstruction, label BCE, mean parity."""
    def __init__(self,x,k):
        super().__init__();self.prototypes=nn.Parameter(x[torch.randperm(len(x))[:k]].clone())
        self.weights=nn.Parameter(torch.zeros(k))
    def forward(self,x):
        z=torch.softmax(-torch.cdist(x,self.prototypes),dim=1)
        p=(z@self.weights.sigmoid()).clamp(1e-6,1-1e-6)
        return z,p,z@self.prototypes

class ScoreRelease:
    """Train-fit stochastic channel; same shared output alphabet for every group."""
    def __init__(self,bins=32,aware=False,tv=.01):self.bins=bins;self.aware=aware;self.tv=tv
    def fit(self,p,y,a):
        p,y,a=map(np.asarray,(p,y,a));self.edges=np.unique(np.quantile(p,np.linspace(0,1,self.bins+1)[1:-1]))
        b=np.searchsorted(self.edges,p,side='right');nb=len(self.edges)+1
        self.q=np.linspace(.01,.99,16)
        c=b+a*nb if self.aware else b; nc=nb*(2 if self.aware else 1)
        cost=np.zeros((nc,16))
        for j,q in enumerate(self.q):cost[:,j]=np.bincount(c,weights=-(y*np.log(q)+(1-y)*np.log1p(-q)),minlength=nc)/len(p)
        diff=np.bincount(c[a==0],minlength=nc)/(a==0).sum()-np.bincount(c[a==1],minlength=nc)/(a==1).sum()
        # channel variables nc*16 and 16 epigraph variables for L1 discrepancy
        nv=nc*16+16;eq=np.zeros((nc,nv));ub=np.zeros((33,nv))
        for i in range(nc):eq[i,i*16:(i+1)*16]=1
        for j in range(16):
            ub[2*j,j:nc*16:16]=diff;ub[2*j,nc*16+j]=-1
            ub[2*j+1,j:nc*16:16]=-diff;ub[2*j+1,nc*16+j]=-1
        ub[-1,nc*16:]=1
        result=linprog(np.r_[cost.ravel(),np.zeros(16)],A_ub=ub,b_ub=np.r_[np.zeros(32),2*self.tv],
            A_eq=eq,b_eq=np.ones(nc),bounds=(0,None),method='highs')
        if not result.success:raise RuntimeError(result.message)
        self.K=result.x[:nc*16].reshape(nc,16).clip(0)
        self.K/=self.K.sum(1,keepdims=True);self.nb=nb
        self.train_tv=float(abs(diff@self.K).sum()/2)
        return self
    def probabilities(self,p,a):
        b=np.searchsorted(self.edges,p,side='right');c=b+np.asarray(a)*self.nb if self.aware else b
        return self.K[c]
    def transform(self,p,a,seed,repeats=1):
        k=self.probabilities(p,a);r=np.random.default_rng(seed).random((len(p),repeats))
        ix=(r[:,:,None]>np.cumsum(k,axis=1)[:,None,:]).sum(2).clip(0,15)
        return self.q[ix].astype('float32')

def grid():
    cs=[dict(method='raw_erm')]
    cs += [dict(method='fnf',dim=8,gamma=g) for g in [.05,.5,.95]]
    cs += [dict(method='mmd',dim=2,ratio=r) for r in [.1,.3,1.]]
    cs += [dict(method='adversarial',dim=32,ratio=r) for r in [.3,1.,3.]]
    from experiments.rq8_final.common import SEED,PATHOLOGY
    for i,c in enumerate(cs):c.update(index=i,id=f'{i:02d}_{c["method"]}',seed=SEED,pathology=PATHOLOGY,condition='shift50')
    return cs
