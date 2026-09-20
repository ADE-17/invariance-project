"""Independent frozen-representation audit and paired patient uncertainty."""
import sys,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,balanced_accuracy_score
from scipy.stats import ks_2samp
from scipy.optimize import minimize,brentq
from scipy.special import expit,logit
from experiments.rq8_final.common import *
from experiments.rq8_final.methods import grid
from experiments.rq8_final.probes import fit
from experiments.rq5_sweep.metrics import point_metrics,cluster_weights,interval,weighted_task,weighted_ece
from experiments.rq4_ptbxl.probe_analysis import weighted_aucs
from experiments.rq5_removal.audit import finite_probability_threshold

FAMILIES=['linear','tree','rbf_svm','mlp0','mlp1','mlp2']

def extended(y,p,a,t):
    result=point_metrics(y,p,a,t)
    result['bce']=float(log_loss(y,p,labels=[0,1]))
    for g in [0,1]:
        m=a==g;yg,pg=y[m],p[m];lg=logit(np.clip(pg,1e-6,1-1e-6))
        slope=None;intercept=None
        if np.std(lg)>1e-6 and len(np.unique(yg))==2:
            def fun(b):
                eta=b[0]+b[1]*lg;err=expit(eta)-yg
                return np.mean(np.logaddexp(0,eta)-yg*eta),np.array([err.mean(),(err*lg).mean()])
            r=minimize(fun,[0.,1.],jac=True,method='BFGS')
            if np.linalg.norm(r.jac)<1e-4:intercept=float(r.x[0]);slope=float(r.x[1])
        result.update({f'group{g}_auprc':float(average_precision_score(yg,pg)),f'group{g}_brier':float(np.mean((yg-pg)**2)),
            f'group{g}_calibration_bias':float(pg.mean()-yg.mean()),f'group{g}_calibration_slope':slope,
            f'group{g}_calibration_intercept':intercept,
            f'group{g}_citl':float(brentq(lambda b:expit(lg+b).mean()-yg.mean(),-40,40)) if 0<yg.mean()<1 else None})
    result['worst_group_auprc']=min(result['group0_auprc'],result['group1_auprc'])
    result['score_variance']=float(np.var(p));result['score_ks']=float(ks_2samp(p[a==0],p[a==1]).statistic)
    result['ece_gap']=abs(result['group0_ece']-result['group1_ece'])
    return result

def metric_deltas(model,reference):
    return {k:float(v-reference[k]) for k,v in model.items()
            if isinstance(v,(int,float)) and isinstance(reference.get(k),(int,float))}

def detailed_weighted(y,p,a,t,w):
    out=weighted_task(y,p,a,t,w);den=w.sum(1);q=(p>=t).astype(float)
    out['ece']=weighted_ece(y,p,w);out['auprc']=weighted_ap(y,p,w)
    for g in [0,1]:
        m=a==g;wg=w[:,m];yg=y[m];pg=p[m];gd=wg.sum(1)
        out[f'group{g}_ece']=weighted_ece(yg,pg,wg)
        out[f'group{g}_brier']=wg@((yg-pg)**2)/gd
        out[f'group{g}_calibration_bias']=wg@(pg-yg)/gd
        out[f'group{g}_auroc']=weighted_aucs(yg,pg,wg)
        out[f'group{g}_auprc']=weighted_ap(yg,pg,wg)
        for name,mask in [('ppr',m),('tpr',m&(y==1)),('fpr',m&(y==0))]:
            out[f'group{g}_{name}']=w[:,mask]@q[mask]/w[:,mask].sum(1)
    out['fpr_gap']=abs(out['group0_fpr']-out['group1_fpr'])
    out['ece_gap']=abs(out['group0_ece']-out['group1_ece'])
    out['worst_group_auprc']=np.minimum(out['group0_auprc'],out['group1_auprc'])
    return out

def weighted_ap(y,p,w):
    order=np.argsort(-p,kind='stable');sy=np.asarray(y)[order];sp=np.asarray(p)[order];sw=w[:,order]
    ends=np.r_[np.flatnonzero(sp[:-1]!=sp[1:]),len(sp)-1]
    tp=np.cumsum(sw*sy,axis=1)[:,ends];den=np.cumsum(sw,axis=1)[:,ends]
    precision=np.divide(tp,den,out=np.zeros_like(tp),where=den>0)
    increments=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
    return (increments*precision).sum(1)/np.maximum(tp[:,-1],1e-12)

def evaluate(index,smoke=False):
    c=grid()[index];folder=ROOT/'runs'/c['id'];out=folder/'evaluation';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    assert (folder/'complete.json').exists()
    sets=frames();device='cuda' if torch.cuda.is_available() else 'cpu';start=time.time()
    before={s:sha(folder/f'{s}_z.npy') for s in SPLITS}
    p={s:np.load(folder/f'{s}_p.npy') for s in SPLITS}
    assert all(len(sets[s])==len(p[s]) for s in SPLITS)
    t=finite_probability_threshold(sets['validation'].y.to_numpy(),p['validation'])
    base={s:np.load(ROOT/'baseline'/f'{s}_p.npy') for s in SPLITS}
    bt=finite_probability_threshold(sets['validation'].y.to_numpy(),base['validation'])
    tests=['selection','test']
    pred={s:sets[s][['row_id','unit','a','y']].copy() for s in tests}
    for s in tests:pred[s]['task_probability']=p[s];pred[s]['erm_probability']=base[s]
    # Fixed 40,000 training rows sampled without using representation/probe outcomes.
    ti=np.random.default_rng(42).choice(len(sets['train']),min(40000,len(sets['train'])),replace=False)
    vi=np.arange(len(sets['probe_validation']))
    if smoke:ti=ti[:320];vi=vi[:320]
    np.savez(out/'probe_row_indices.npz',train=ti,validation=vi)
    variants=['representation','score']
    if (folder/'train_mean_z.npy').exists():variants.append('mean')
    if (folder/'train_repeat100_z.npy').exists():variants.append('repeat100')
    probe_meta=[]
    for variant in variants:
        if variant=='score':zs={s:p[s][:,None] for s in SPLITS}
        else:
            suffix='z' if variant=='representation' else f'{variant}_z'
            zs={s:np.load(folder/f'{s}_{suffix}.npy',mmap_mode='r') for s in SPLITS}
        # Score-release representation equals its single score; reuse exactly rather than fit identical probes twice.
        if variant=='score' and c['method'] in ['score_release','constant_control']:
            for s in tests:
                for col in list(pred[s]):
                    if col.startswith('representation__'):pred[s][col.replace('representation__','score__')]=pred[s][col]
            continue
        for label in [None,0,1]:
            suffix='marginal' if label is None else f'Y{label}'
            tm=np.ones(len(ti),bool) if label is None else sets['train'].y.to_numpy()[ti]==label
            vm=np.ones(len(vi),bool) if label is None else sets['probe_validation'].y.to_numpy()[vi]==label
            masks={s:np.ones(len(sets[s]),bool) if label is None else sets[s].y.to_numpy()==label for s in tests}
            probe_dir=out/'probes'/variant/suffix
            cache=probe_dir/'scores.npz'
            if cache.exists():
                npz=np.load(cache);scores={s:{fam:npz[f'{s}__{fam}'] for fam in FAMILIES+['permuted_linear']} for s in tests}
            else:
                scores=fit(np.asarray(zs['train'][ti[tm]]),sets['train'].a.to_numpy()[ti[tm]],
                    np.asarray(zs['probe_validation'][vi[vm]]),sets['probe_validation'].a.to_numpy()[vi[vm]],
                    {s:np.asarray(zs[s][masks[s]]) for s in tests},probe_dir,SEED+(0 if label is None else 100*(label+1)),device,smoke)
                np.savez_compressed(cache,**{f'{s}__{fam}':pp for s,sc in scores.items() for fam,pp in sc.items()})
            for s in tests:
                for fam,pp in scores[s].items():
                    col=f'{variant}__{suffix}__{fam}';arr=np.full(len(sets[s]),np.nan);arr[masks[s]]=pp;pred[s][col]=arr
            probe_meta.append(dict(variant=variant,scope=suffix,training_rows=int(tm.sum()),validation_rows=int(vm.sum()),features=zs['train'].shape[1]))
            log(f'{c["id"]}: probes {variant}/{suffix} completed')
    rows=[]
    for s in tests:
        d=sets[s];a=d.a.to_numpy();y=d.y.to_numpy();metrics=extended(y,p[s],a,t);erm=extended(y,base[s],a,bt)
        row=dict(config=c,split=s,threshold=t,metrics=metrics,erm=erm,threshold_05=extended(y,p[s],a,.5),
            delta_vs_erm=metric_deltas(metrics,erm),erm_threshold=bt,
            erm_threshold_05=extended(y,base[s],a,.5),
            delta_vs_erm_threshold_05=metric_deltas(extended(y,p[s],a,.5),extended(y,base[s],a,.5)),
            utility_relative_auroc_drop=(erm['auroc']-metrics['auroc'])/erm['auroc'],
            utility_absolute_auroc_drop=erm['auroc']-metrics['auroc'],
            within_10percent_auroc_budget=metrics['auroc']>=.9*erm['auroc'],
            additional_collapse_flag=metrics['auroc']<.7 or metrics['worst_group_auroc']<.65,
            gate_is_exploratory=True,independence_proven=False,probes={})
        columns=[col for col in pred[s] if '__' in col and not col.endswith('permuted_linear')]
        task_draws={};probe_draws={col:[] for col in columns}
        ba_draws={col:[] for col in columns};ll_draws={col:[] for col in columns}
        reps=20 if smoke else 2000
        for w in cluster_weights(d.unit,reps):
            mm=detailed_weighted(y,p[s],a,t,w);bb=detailed_weighted(y,base[s],a,bt,w)
            for key in mm:
                task_draws.setdefault(key,[]).append(mm[key]);task_draws.setdefault('delta_'+key,[]).append(mm[key]-bb[key])
            for col in columns:
                m=pred[s][col].notna().to_numpy();probe_draws[col].append(weighted_aucs(a[m],pred[s][col].to_numpy()[m],w[:,m]))
                aa=a[m];pp=np.clip(pred[s][col].to_numpy()[m],1e-7,1-1e-7);ww=w[:,m]
                ba=np.zeros(len(w));ll=np.zeros(len(w))
                for g in [0,1]:
                    gm=aa==g;gw=ww[:,gm];den=gw.sum(1).clip(1)
                    ba+=.5*(gw@((pp[gm]>=.5)==g))/den
                    losses=-(g*np.log(pp[gm])+(1-g)*np.log1p(-pp[gm]))
                    ll+=.5*(gw@losses)/den
                ba_draws[col].append(np.maximum(ba,1-ba));ll_draws[col].append(np.log(2)-ll)
        row['task_intervals']={key:interval(np.concatenate(v)) for key,v in task_draws.items()}
        for col in columns:
            m=pred[s][col].notna().to_numpy();aa=a[m];pp=pred[s][col].to_numpy()[m];auc=roc_auc_score(aa,pp)
            vals=np.concatenate(probe_draws[col]);ci=interval(vals,.05/(len(FAMILIES)*(2 if '__Y' in col else 1)))
            balanced_ll=.5*sum(log_loss(aa[aa==g],pp[aa==g],labels=[0,1]) for g in [0,1])
            row['probes'][col]=dict(auroc=float(auc),oriented_auroc=float(max(auc,1-auc)),
                auprc=float(average_precision_score(aa,pp)),balanced_accuracy=float(balanced_accuracy_score(aa,pp>=.5)),
                balanced_log_loss=float(balanced_ll),balanced_log_loss_gain=float(np.log(2)-balanced_ll),
                natural_log_loss=float(log_loss(aa,pp)),**ci,
                balanced_accuracy_interval=interval(np.concatenate(ba_draws[col]),.05/(len(FAMILIES)*(2 if '__Y' in col else 1))),
                balanced_logloss_gain_interval=interval(np.concatenate(ll_draws[col]),.05/(len(FAMILIES)*(2 if '__Y' in col else 1))),
                oriented_upper=max(ci['high'],1-ci['low']) if ci['high'] is not None else None)
        row['gates']={}
        for variant in variants:
            row['gates'][variant]={}
            for scope,suffixes in [('marginal',['marginal']),('conditional',['Y0','Y1'])]:
                details=[row['probes'][f'{variant}__{suffix}__{fam}'] for suffix in suffixes for fam in FAMILIES]
                uppers=[v['oriented_upper'] for v in details]
                row['gates'][variant][scope]=dict(auroc_gate=all(u is not None and u<=.55 for u in uppers),
                    max_oriented_auroc=max(v['oriented_auroc'] for v in details),max_oriented_upper=max(uppers) if None not in uppers else None,
                    balanced_accuracy_point_check=all(max(v['balanced_accuracy'],1-v['balanced_accuracy'])<=.55 for v in details),
                    balanced_logloss_point_check=all(v['balanced_log_loss_gain']<=.01 for v in details),
                    balanced_accuracy_interval_check=all(v['balanced_accuracy_interval']['high'] is not None and v['balanced_accuracy_interval']['high']<=.55 for v in details),
                    balanced_logloss_interval_check=all(v['balanced_logloss_gain_interval']['high'] is not None and v['balanced_logloss_gain_interval']['high']<=.01 for v in details),
                    correction='Bonferroni across probe families within scope, not across sweep')
        reference_file=ROOT/'runs'/'00_raw_erm'/'evaluation'/f'{s}_metrics.json'
        if reference_file.exists():
            reference=json.loads(reference_file.read_text())
            row['probe_delta_vs_erm']={key:row['probes'][key]['oriented_auroc']-reference['probes'][key]['oriented_auroc'] for key in row['probes']}
        write(out/f'{s}_metrics.json',row);pred[s].to_csv(out/f'{s}_predictions.csv.gz',index=False,compression='gzip')
        np.savez_compressed(out/f'{s}_bootstrap.npz',**{**{key:np.concatenate(v) for key,v in task_draws.items()},**{key:np.concatenate(v) for key,v in probe_draws.items()}})
        curves=[]
        for th in np.linspace(0,1,21):
            cm=point_metrics(y,p[s],a,float(th));cb=point_metrics(y,base[s],a,float(th))
            curves.append(dict(threshold=float(th),**cm,**{'delta_'+k:v for k,v in metric_deltas(cm,cb).items()}))
        pd.DataFrame(curves).to_csv(out/f'{s}_thresholds.csv',index=False)
        sub=[]
        if 'Age' in d:
            for name,mask in [('age_lt40',d.Age<40),('age_40_64',d.Age.between(40,64)),('age_ge65',d.Age>=65),
                              ('certain_labels',~d.uncertain.astype(bool))]:
                if all(len(np.unique(y[np.asarray(mask)&(a==g)]))==2 for g in [0,1]):
                    sub.append(dict(subset=name,**extended(y[mask],p[s][mask],a[mask],t)))
        pd.DataFrame(sub).to_csv(out/f'{s}_subgroups.csv',index=False);rows.append(row)
    after={s:sha(folder/f'{s}_z.npy') for s in SPLITS};assert before==after
    write(out/'complete.json',dict(elapsed_seconds=time.time()-start,probe_manifest=probe_meta,feature_hashes=after,
        model_updated_by_probes=False,bootstrap_replicates=20 if smoke else 2000,smoke=smoke))
    log(f'Evaluated {c["id"]} in {time.time()-start:.1f}s')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    evaluate(args.index,args.smoke)
