"""Generate RQ4 report from metadata and completed trial artifacts."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.rq4_ptbxl.prepare import OUT,CACHE,CLASSES,DISEASES,load,shifted

def table(df):
    df=df.copy()
    for c in df:
        if pd.api.types.is_float_dtype(df[c]):df[c]=df[c].map(lambda x:f'{x:.3f}' if pd.notna(x) else 'NA')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))+'\n'

def main():
    p=pd.read_csv(OUT/'prevalence_all_strata.csv');d=load();audit=json.loads((OUT/'data_audit.json').read_text())
    parts=['''# RQ4: PTB-XL ECG invariance trial

## Question and scope

Does changing the modality from chest radiographs to ECG make recorded-sex information easier to remove, and do fairness-gap reductions track actual marginal/conditional invariance?

This is a **single-replicate exploratory trial**, not a seed sweep or a benchmark replication. ECGs may retain demographic information too. ERM is the necessary comparison for DANN/CDANN. Absence of leakage from the tested probes does not prove exact independence; strong held-out leakage rejects the corresponding practical erasure claim. DANN targets marginal independence; true-label CDANN targets within-label independence. Neither a small DP nor EOdds gap establishes erased representations. The broader conceptual motivation remains [Petersen et al.](https://arxiv.org/abs/2305.01397), as discussed in the [RQ1–RQ3 interpretation](../../interpretation/experiment_interpretation.md).

## Data and provenance

The supplied directory is named 1.0.3 but contains 1.0.1 metadata and waveforms. This trial downloads official **1.0.3 metadata**, which removes 38 earlier records and revises duplicate labels, and verifies all retained 100-Hz waveform/header files against the 1.0.3 checksum manifest. No source dataset files are overwritten. The prepared cohort has 21,799 ECGs from 18,869 patients. Training uses folds 1–8, validation fold 9, test fold 10; patient separation is asserted. Recorded sex is PTB-XL 0=Male, 1=Female, remapped to the project’s Female=0/Male=1. Ages ≥90, including deidentified values around 300, are grouped as “90+ (deidentified),” not treated as literal ages. These are 10-second, 12-lead ECGs. [Official dataset documentation and citation](https://physionet.org/content/ptb-xl/1.0.3/).

Label construction follows the official diagnostic-superclass mapping: membership in MI, STTC, CD, HYP or NORM is based on diagnostic SCP code presence, including zero likelihood (unknown likelihood, not code absence). Labels overlap; prevalences do not sum to 100%. Label prevalence describes annotated ECG records rather than clinical population disease prevalence. Age/sex comparisons are descriptive and do not establish a causal demographic effect.

The metadata audit below includes all records. Modeling excludes 411 records with no diagnostic superclass; their disease status should not be assumed negative. Per-run cohort manifests give the exact model populations, so prevalence denominators can be reconciled. Multiple ECGs per patient remain in a split. The report gives record-level prevalence and distinct patient support, not patient-level disease incidence.

## Natural prevalence by sex

Numbers below are proportions, not percentages; `positive` is an ECG count.
''']
    parts.append(table(p.query("condition=='natural' and split=='all' and stratification=='sex'")[['sex','label','n_ecgs','n_patients','positive','prevalence']]))
    parts.append('MI and conduction-disturbance labels are more common in male ECGs (30.0% versus 19.7%, and 26.4% versus 18.2%). ST/T changes show the opposite direction (22.6% male versus 25.6% female). These are aggregate record-level comparisons; the age tables below expose a potential compositional explanation rather than establishing sex as a cause.\n')
    parts.append('## Natural prevalence by age\n\n')
    parts.append(table(p.query("condition=='natural' and split=='all' and stratification=='age'")[['age','label','n_ecgs','n_patients','positive','prevalence']]))
    parts.append('Age differences are substantial: MI-label prevalence rises from 4.6% below age 40 to 37.1% at ages 80–89; normal-ECG labels fall from 80.8% to 17.6% across those groups. The oldest bin contains only 293 records overall, so subgroup estimates require particular caution.\n')
    parts.append('## Natural prevalence jointly by sex and age\n\n')
    parts.append(table(p.query("condition=='natural' and split=='all' and stratification=='sex_age'")[['sex','age','label','n_ecgs','positive','prevalence']]))
    parts.append('''## Added prevalence shift and trial protocol

For each target disease independently, randomly remove 50% of Female-positive records (rounded down) from the training split, and independently construct a similarly shifted test subset. No labels are flipped. All other sex×label cells remain intact. This shifts group prevalence and sample sizes and can shift correlated labels/age composition. Validation always remains natural. Removal manifests are shared across methods. A fixed RNG value 42 makes this one trial reproducible; it is not a multi-seed experiment.

Four diagnostic targets × two training conditions × ERM/DANN/CDANN = **24 planned model runs**, each evaluated on **both** natural and shifted test cohorts: 48 evaluations. This includes natural-trained→shifted-test evaluation, which was missing in RQ1/RQ2. The shift is synthetic and does not simulate every form of clinical distribution change.

The train/validation/test normalization uses per-lead means and standard deviations fitted only on the natural training cohort, shared across conditions. The model is a compact 1D CNN (32/64/128 channels, kernels 11/7/5, stride 2, batch normalization, global pooling), a 128-dimensional representation and a linear binary task head. Inputs are waveforms only; age and sex are not task-head inputs. Adam LR 0.001, weight decay 0.0001, batch 128, 20 epochs. DANN uses one adversarial MLP; CDANN uses two, conditioned on true labels, with sex-balanced BCE within each available class. Missing-sex minibatch cells are skipped and counted. λ=1, with linear warmup over the first five epochs. Checkpoints minimize natural-validation task BCE **only among epochs ≥6**, when full λ is active. Training always runs all 20 epochs and saves both selected and last checkpoints. Reported evaluations use the selected checkpoint.

The independent probe audit uses standardized features, a balanced logistic probe and a separately fitted nonlinear 128/64 MLP. Probes are trained on the actual training cohort, including its downsampling where applicable, and the MLP’s 30-epoch selection uses natural-validation sex loss. Test labels never train/select a probe or diagnostic threshold. Marginal and separate Y=0/Y=1 probes are reported. This audit MLP has the same nominal widths as the adversary; its independent fitting is a useful test, not an exhaustive decoder search.

Task metrics: AUROC, AUPRC, Brier score, ECE, worst-sex-group AUROC, and group support. Fairness: absolute TPR gap (EOpp), maximum of absolute TPR/FPR gaps (EOdds), and positive-prediction-rate gap (DP), all using one global natural-validation Youden threshold per model. That threshold is reused on both test sets. Age and sex×age task/rate tables are saved separately. DP is descriptive when prevalences differ. PPR = TPR×prevalence + FPR×(1−prevalence), so EOdds and DP need not move together. Near-constant features and diagnostic collapse are checked alongside probe AUC.

The main trial tables have no multi-seed SDs or significance claims. The separate probe-reduction analysis adds paired patient-bootstrap intervals for demographic AUROC, conditional on these fitted models/probes; it does not estimate training-run uncertainty. Small age×sex cells can have unstable AUROC/rates; undefined quantities are NA. A selected checkpoint does not automatically satisfy invariance. No λ optimum is claimed.

### Exact model-cohort prevalence before/after injection

The following includes only diagnostically annotated records and reports each target on its own shifted cohort. This differs from the all-record metadata audit above.
''')
    model=d[d[CLASSES].sum(axis=1)>0];co=[]
    for target in DISEASES:
        for split,g in model.groupby('split'):
            for cond,sub in [('natural',g),('prevalence_shift',g if split=='validation' else shifted(g,target))]:
                for sex,z in sub.groupby('sex_name'):
                    co.append(dict(target=target,split=split,condition=cond,sex=sex,n=len(z),positive=int(z[target].sum()),prevalence=z[target].mean()))
    co=pd.DataFrame(co);co.to_csv(OUT/'model_cohort_prevalence.csv',index=False);parts.append(table(co))
    complete=[]
    for f in (CACHE/'trials').rglob('complete.json'):complete+=json.loads(f.read_text())
    parts.append('## Trial results and current status\n\nFor a direct before/after assessment with paired patient-bootstrap intervals, see [Does adversarial training reduce demographic information?](probe_reduction_analysis.md).\n\n')
    if complete:
        r=pd.DataFrame(complete).sort_values(['target','training_condition','method','test_condition']);r.to_csv(OUT/'trial_results.csv',index=False)
        parts.append(f'{len(r)//2} completed model runs; {len(r)} test evaluations. Values are one trial each.\n')
        for target in sorted(r.target.unique()):
            z=r[r.target==target];parts.append(f'### {target}: task and fairness\n\n'+table(z[['training_condition','method','lambda_adv','test_condition','best_epoch','auroc','auprc','worst_group_auroc','delta_eopp','delta_fpr','delta_eodds','delta_dp','ece']]))
            parts.append(f'### {target}: independent sex probes\n\n'+table(z[['training_condition','method','lambda_adv','test_condition','probe_linear_auroc','probe_mlp_auroc','probe_linear_Y0_auroc','probe_mlp_Y0_auroc','probe_linear_Y1_auroc','probe_mlp_Y1_auroc','feature_mean_std']]))
        adv=r[r.method!='erm']
        if len(adv):
            parts.append(f"Across the completed adversarial evaluations, marginal MLP sex AUC ranges from {adv.probe_mlp_auroc.min():.3f} to {adv.probe_mlp_auroc.max():.3f}; within-Y MLP AUC ranges from {min(adv.probe_mlp_Y0_auroc.min(),adv.probe_mlp_Y1_auroc.min()):.3f} to {max(adv.probe_mlp_Y0_auroc.max(),adv.probe_mlp_Y1_auroc.max()):.3f}. These are descriptive ranges across different tasks/conditions, not uncertainty estimates.\n")
        if len(adv) and adv.probe_mlp_auroc.min()>.6:
            parts.append('**Interpretation of the completed trial:** every evaluated adversarial representation retains substantial marginal sex decodability. The conditional probe columns must be used to assess CDANN; high marginal leakage alone would not refute conditional independence. These observations do not establish successful demographic erasure, even where prediction disparities improve. They concern this CNN, λ and optimization/selection protocol, not an impossibility result for ECG.\n')
        comparisons=[]
        for row in adv.to_dict('records'):
            base=r[(r.target==row['target'])&(r.training_condition==row['training_condition'])&(r.test_condition==row['test_condition'])&(r.method=='erm')]
            if len(base)!=1:continue
            b=base.iloc[0];comparisons.append({**{k:row[k] for k in ['target','training_condition','test_condition','method']},**{'change_'+k:row[k]-b[k] for k in ['auroc','worst_group_auroc','delta_eodds','delta_dp','probe_mlp_auroc']}})
        if comparisons:
            comp=pd.DataFrame(comparisons);comp.to_csv(OUT/'changes_vs_erm.csv',index=False);parts.append('### Changes versus matching ERM\n\nNegative gap changes indicate smaller disparities; positive AUROC changes indicate improved discrimination. These single-trial comparisons do not isolate a causal effect of erasure.\n\n'+table(comp))
        mi=r[(r.target=='MI')&(r.training_condition=='prevalence_shift')&(r.test_condition=='prevalence_shift')]
        if {'erm','dann'}.issubset(set(mi.method)):
            b=mi[mi.method=='erm'].iloc[0];a=mi[mi.method=='dann'].iloc[0]
            parts.append(f'### Concrete interpretation: MI under shifted training and testing\n\nDANN changes ΔEOdds from {b.delta_eodds:.3f} to {a.delta_eodds:.3f}, while independent marginal sex-probe AUROC changes from {b.probe_mlp_auroc:.3f} to {a.probe_mlp_auroc:.3f}. Diagnostic AUROC changes from {b.auroc:.3f} to {a.auroc:.3f}. This example illustrates a smaller output disparity without demographic erasure; it does not establish that erasure caused the disparity change. The full tables include outcomes where disparities worsen.\n')
        hyp=r[(r.target=='HYP')&(r.training_condition=='prevalence_shift')&(r.test_condition=='prevalence_shift')]
        if {'erm','cdann'}.issubset(set(hyp.method)):
            b=hyp[hyp.method=='erm'].iloc[0];a=hyp[hyp.method=='cdann'].iloc[0]
            parts.append(f'For hypertrophy in the same training/test conditions, CDANN changes ΔDP from {b.delta_dp:.3f} to {a.delta_dp:.3f} but ΔEOdds from {b.delta_eodds:.3f} to {a.delta_eodds:.3f}. A method cannot be summarized as simply “fairer” when different parity criteria move in opposite directions. Both are threshold-dependent descriptions here.\n')
        # Gather every group-level result with identifiers, preserving small-cell support.
        gr=[]
        for row in r.to_dict('records'):
            g=pd.read_csv(Path(row['run_dir'])/f"groups_{row['test_condition']}.csv")
            for k in ['target','training_condition','method','test_condition']:g[k]=row[k]
            gr.append(g)
        pd.concat(gr).to_csv(OUT/'all_group_metrics.csv',index=False)
    else:parts.append('Training/evaluation has not yet produced completed model results. Prevalence audit and protocol are available; no erasure result is claimed.\n')
    diagnostics=OUT/'invariance_diagnostics.csv'
    if diagnostics.exists():
        diag=pd.read_csv(diagnostics)
        parts.append('### Training-adversary and distribution diagnostics\n\nThese held-out training-adversary scores are distinct from independently refitted probes. Below-chance adversary AUROC can be inverted, so it is not evidence of erasure. CDANN scores are reported within true labels. Gaussian biased MMD² and HSIC use a fixed random subsample of at most 1,000 test records and a pooled median squared-distance bandwidth. Values have no permutation-null calibration and are not independence tests. Conditional kernel diagnostics and oriented adversary AUCs are included in [the full CSV](invariance_diagnostics.csv).\n\n')
        parts.append(table(diag[['target','training_condition','method','test_condition','training_adversary_auroc','training_adversary_auroc_Y0','training_adversary_auroc_Y1','mmd2_biased','hsic_biased']]))
    wave=CACHE/'waveform_audit.json'
    if wave.exists():
        w=json.loads(wave.read_text());parts.append(f"Waveform audit: {w['checked_files']} files checked against official checksums; {len(w['checksum_mismatches'])} mismatches; {len(w['duplicate_waveforms'])} duplicate waveform pairs.\n")
    parts.append('''The final checkpoint audit reproduces predictions and fairness metrics from saved artifacts. It uses the recorded float32 normalization and, where necessary, disables Ampere TF32 convolution to match the original Volta baseline. Per-evaluation reproduction differences and backend settings are saved in `invariance_diagnostics.csv`; [verification.json](verification.json) records completed checks.

## Files, verification and reproduction

- [All metadata prevalence strata](prevalence_all_strata.csv): natural and target-specific shifted cohorts, full dataset and each fold split, sex, age and sex×age.
- [Model-cohort prevalence](model_cohort_prevalence.csv), [data audit](data_audit.json), [cohort with labels](cohort.csv).
- Completed outputs: `trial_results.csv`, `changes_vs_erm.csv`, `all_group_metrics.csv`.
- Heavy outputs: `/path/to/results/rq4/trials/`, containing full configuration/source hash, per-epoch histories, selected/last checkpoints, exact cohort manifests, record-level predictions, demographic probe probabilities and subgroup tables.
- Source: [prepare.py](../../experiments/rq4_ptbxl/prepare.py), [trial.py](../../experiments/rq4_ptbxl/trial.py), [basic tests](../../experiments/rq4_ptbxl/test_trial.py), [report.py](../../experiments/rq4_ptbxl/report.py), [Slurm entry point](../../experiments/rq4_ptbxl/run_trial.sbatch).

Run from `.`:

```bash
python experiments/rq4_ptbxl/prepare.py
python -m unittest experiments.rq4_ptbxl.test_trial
sbatch experiments/rq4_ptbxl/run_trial.sbatch
python experiments/rq4_ptbxl/report.py
```

The dedicated environment adds WFDB 4.2.0 and reads the existing qwen-eval scientific stack through a `.pth` file; it does not change that environment. Existing RQ1–RQ3 results are untouched. Completed trials are skipped on rerun. Use a fresh run directory for a changed epoch/model protocol; changing λ gives a separate adversarial directory. These trials do not establish that chest radiographs in general cannot be made invariant, or that ECG is inherently easier; that comparison requires matched experimental designs and stronger replication.
''')
    from experiments.rq4_ptbxl.plots import make_plots
    make_plots()
    parts.append('## Figures\n\n![Natural prevalence by sex and age](prevalence_sex_age.png)\n')
    if complete:parts.append('![Fairness gaps versus marginal demographic decodability](fairness_vs_decodability.png)\n')
    (OUT/'rq4_report.md').write_text('\n\n'.join(parts))
    print('Wrote',OUT/'rq4_report.md',flush=True)
if __name__=='__main__':main()
