"""Paired patient bootstrap of held-out demographic-probe AUC differences.
Uses saved predictions only. Does not retrain any encoder or probe.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.rq4_ptbxl.prepare import OUT,CACHE

B=500

def weighted_aucs(y,p,w):
    """Exact weighted rank AUC including tied scores, vectorized over replicates."""
    order=np.argsort(p,kind='stable');y=y[order];p=p[order];w=w[:,order]
    starts=np.r_[0,np.flatnonzero(np.diff(p)!=0)+1]
    pos=np.add.reduceat(w*(y==1),starts,axis=1)
    neg=np.add.reduceat(w*(y==0),starts,axis=1)
    numerator=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1)
    denominator=pos.sum(axis=1)*neg.sum(axis=1)
    return np.divide(numerator,denominator,out=np.full(len(w),np.nan),where=denominator>0)

def main():
    results=[]
    for f in (CACHE/'trials').rglob('complete.json'):results+=json.loads(f.read_text())
    assert len(results)==48
    frame=pd.DataFrame(results);rows=[];rng=np.random.default_rng(20260908)
    for m in frame[frame.method!='erm'].to_dict('records'):
        base=frame[(frame.target==m['target'])&(frame.training_condition==m['training_condition'])&(frame.test_condition==m['test_condition'])&(frame.method=='erm')].iloc[0]
        name=f"predictions_{m['test_condition']}.csv"
        a=pd.read_csv(Path(m['run_dir'])/name).sort_values('ecg_id').reset_index(drop=True)
        b=pd.read_csv(Path(base.run_dir)/name).sort_values('ecg_id').reset_index(drop=True)
        for key in ['ecg_id','patient_id','recorded_sex','y']:assert np.array_equal(a[key],b[key])
        patients,inverse=np.unique(a.patient_id.to_numpy(),return_inverse=True)
        draws=rng.multinomial(len(patients),np.ones(len(patients))/len(patients),size=B)
        weights=draws[:,inverse];sex=a.recorded_sex.to_numpy();metrics={}
        for suffix in ['', '_Y0','_Y1']:
            for kind in ['linear','mlp']:
                col='probe_'+kind+suffix;mask=np.isfinite(a[col])&np.isfinite(b[col]);yy=sex[mask]
                aa=a.loc[mask,col].to_numpy();bb=b.loc[mask,col].to_numpy()
                point_a=roc_auc_score(yy,aa);point_b=roc_auc_score(yy,bb)
                assert abs(weighted_aucs(yy,aa,np.ones((1,len(yy))))[0]-point_a)<1e-10
                boot_a=weighted_aucs(yy,aa,weights[:,mask]);boot_b=weighted_aucs(yy,bb,weights[:,mask]);delta=boot_a-boot_b
                low,high=np.nanquantile(delta,[.025,.975])
                a_low,a_high=np.nanquantile(boot_a,[.025,.975]);b_low,b_high=np.nanquantile(boot_b,[.025,.975])
                row={k:m[k] for k in ['target','training_condition','test_condition','method','lambda_adv']}
                row.update(probe=col,erm_auc=point_b,method_auc=point_a,delta_auc=point_a-point_b,delta_ci_low=low,delta_ci_high=high,method_ci_low=a_low,method_ci_high=a_high,erm_ci_low=b_low,erm_ci_high=b_high,n_ecgs=int(mask.sum()),n_patients=int(a.loc[mask,'patient_id'].nunique()),bootstrap_replicates=B,valid_bootstrap_replicates=int(np.isfinite(delta).sum()))
                rows.append(row)
        print('Compared',m['target'],m['training_condition'],m['method'],m['test_condition'],flush=True)
    r=pd.DataFrame(rows).sort_values(['target','training_condition','test_condition','method','probe'])
    r.to_csv(OUT/'probe_reduction_analysis.csv',index=False)
    from experiments.rq4_ptbxl.report import table
    parts=['''# Does adversarial training reduce demographic information in RQ4?

## What is being measured

**The probes do not remove information.** In `trial.py`, encoder feature extraction runs under `torch.no_grad()`, and `fit_probes` receives detached NumPy feature arrays. Its optimizer contains only probe parameters. The encoder is never updated by the audit. DANN/CDANN are the interventions intended to change the representation; the independently fitted probes measure how much recorded sex remains recoverable afterward.

A lower held-out sex-probe AUROC means **lower decodability for that fitted probe and protocol**. It does not directly measure mutual information, prove information destruction, or establish independence against every possible decoder. An underfit probe can appear near chance even when information remains. Both a linear and nonlinear probe are therefore checked, overall and within Y=0/Y=1. A reduction in one probe family while another remains strong is evidence of incomplete erasure, not successful invariance. No acceptable-invariance equivalence margin was prespecified in this trial.

## Matched comparison and uncertainty

Each DANN/CDANN prediction file is matched to ERM by disease, training condition, test condition and ECG ID. Sex, pathology label and patient identities are asserted equal. We compare **method minus ERM demographic AUROC**; a negative difference means reduced measured decodability.

Intervals are 95% percentile intervals from 500 paired patient-cluster bootstrap replicates. Patients are resampled with replacement, and all of their ECG records receive the same resampling multiplicity. The same draw is used for ERM and the intervention. Conditional probes are restricted to their label class after the patient resampling. This accounts for repeated ECGs when estimating test-cohort sampling variability. It does **not** account for model-training, probe-fitting, hyperparameter-selection or different-split variability; this remains one fitted trial per configuration. Intervals are exploratory and unadjusted for multiple comparisons. Overlapping intervals around zero do not establish equal representations or absence of an effect.

The complete per-probe point estimates, interval bounds, sample support and all comparisons are in [probe_reduction_analysis.csv](probe_reduction_analysis.csv). The compact tables show `method AUROC; change [95% paired interval]`. Absolute method/ERM AUROC intervals are also saved in the CSV.\n''']
    marginal=r[r.probe=='probe_mlp'];linear=r[r.probe=='probe_linear']
    parts.append(f"Across the 32 marginal MLP comparisons, {int((marginal.delta_auc<0).sum())} point estimates decrease and {int((marginal.delta_auc>0).sum())} increase relative to ERM. {int((marginal.delta_ci_high<0).sum())} paired intervals lie below zero, {int((marginal.delta_ci_low>0).sum())} above zero, and {int(((marginal.delta_ci_low<=0)&(marginal.delta_ci_high>=0)).sum())} include zero. These counts summarize heterogeneous exploratory comparisons and are not a combined significance test.\n")
    parts.append(f"Adversarial marginal MLP AUCs remain {marginal.method_auc.min():.3f}–{marginal.method_auc.max():.3f}; marginal linear AUCs remain {linear.method_auc.min():.3f}–{linear.method_auc.max():.3f}. Within-label MLP AUCs remain {r[r.probe.isin(['probe_mlp_Y0','probe_mlp_Y1'])].method_auc.min():.3f}–{r[r.probe.isin(['probe_mlp_Y0','probe_mlp_Y1'])].method_auc.max():.3f}. Thus the trial does not demonstrate erased demographic representations. For CDANN, the within-label results are the relevant test of its objective.\n")
    for conditional,cols in [('Marginal',['probe_linear','probe_mlp']),('Conditional',['probe_linear_Y0','probe_mlp_Y0','probe_linear_Y1','probe_mlp_Y1'])]:
        out=[]
        for key,g in r.groupby(['target','training_condition','test_condition','method']):
            row=dict(zip(['target','training_condition','test_condition','method'],key))
            for col in cols:
                x=g[g.probe==col].iloc[0];row[col]=f'{x.method_auc:.3f}; {x.delta_auc:+.3f} [{x.delta_ci_low:+.3f}, {x.delta_ci_high:+.3f}]'
            out.append(row)
        parts.append(f'## {conditional} probe comparisons\n\n'+table(pd.DataFrame(out)))
    parts.append('''## What this supports, and what it does not

The relevant question is not whether adversarial loss approaches log(2), but whether independent held-out probes become less capable. The [main report](rq4_report.md) adds held-out scores of the original training adversaries, which may be weak even when refitted probes recover sex. Below-chance adversary AUROC must also be checked after reversing its ranking.

The observed reductions are method-, pathology-, class- and condition-dependent. Read negative changes jointly with absolute AUC: for example, a change from 0.83 to 0.80 is partial reduction at most, not invariance. Read the paired intervals before attributing a small difference to a reliable intervention effect. Also compare both linear and nonlinear probes; neither is guaranteed to be the stronger fitted classifier on every task.

For MI under shifted training/testing, DANN improves EOdds while its marginal sex MLP AUC remains close to ERM. That is a direct example of output disparity improvement without strong evidence of demographic erasure. More broadly, sex decodability does not by itself establish that the disease head uses sex unfairly. These single-trial results cannot establish that DANN/CDANN are incapable of erasure on PTB-XL, or that ECGs inherently permit more erasure than chest radiographs.

A stronger future erasure claim would require a prespecified practical near-chance criterion, probe-capacity/optimization checks and repeated fitting, uncertainty over training runs, class-conditional diagnostics for CDANN, and retained diagnostic utility. None of those additional model experiments is inferred from these bootstrap intervals.

Reproduce this analysis with `python experiments/rq4_ptbxl/probe_analysis.py` from the project directory. It reads saved predictions only and leaves encoder and probe outputs unchanged.
''')
    parts.append('## Concrete checks against misleading erasure claims\n\nFor natural-training/natural-test conduction disturbance, the CDANN training adversaries have within-class AUROCs 0.467 (Y=0) and 0.505 (Y=1). After allowing ranking reversal, the first is 0.533. Independent conditional MLP probes nevertheless reach 0.811 and 0.711. The near-chance training-adversary scores therefore do not establish conditional invariance. This is evidence that the trained adversaries did not expose the demographic signal that an independently fitted audit could recover, not a diagnosis of which optimization or capacity limitation caused the failure.\n\nFor shifted-training/shifted-test MI, the marginal MLP change for DANN is −0.003, with a paired patient-bootstrap interval of approximately [−0.016, +0.010]. Meanwhile, ΔEOdds drops from 0.151 to 0.071. This experiment supports **smaller error-rate disparity without a resolved reduction in marginal demographic decodability**. It does not prove the representations are identical, and it does not attribute the disparity change causally to demographic erasure.\n\nThe clearest modest marginal reductions occur mainly in conduction-disturbance trials: for shifted training with natural testing, CDANN reduces MLP AUROC from 0.819 to 0.796 (change −0.022, paired interval [−0.037, −0.007]). The representation still carries substantial recoverable sex information. Read this as partial reduction under the specified probe protocol; the interval is unadjusted and conditional on the fitted models/probes.\n')
    (OUT/'probe_reduction_analysis.md').write_text('\n\n'.join(parts))
    print('Wrote probe reduction analysis',flush=True)
if __name__=='__main__':main()
