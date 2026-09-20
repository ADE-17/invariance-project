# RQ8 extension: deployable class-conditional FNF on CheXpert

Edema and Cardiomegaly, recorded sex, RQ8 shift50 cohorts, seeds 42/123/456,
gamma 0.05/0.5/0.95: 18 new fits, with six unchanged RQ8 ERM references.
Reuse the exact patient splits and frozen DenseNet121 features. The original
RQ8 artifacts remain unchanged. Results live at
`/path/to/results/rq8/conditional_shift50_v1`.

## Objective and inference

Following the continuous FNF density-matching construction of
[Balunovic et al.](https://arxiv.org/abs/2106.05937), this extension fits
train-only four-component Gaussian mixtures p(x | A=a,Y=y) for each of four
strata after train-fit standardized, whitened 8-D PCA. Two sex-specific
four-block RealNVP flows f_a are **shared across outcome classes**. A shared
64-unit classification head predicts from z=f_a(x). It needs sex, not the
outcome, at inference. It is not the RQ6 true-label oracle or proxy routing.

For each class y define K_y = [KL(q_0y || q_1y) + KL(q_1y || q_0y)] / 2.
Optimize (1-gamma) BCE + gamma (K_0+K_1)/(2d), with d=8. Estimate each
direction with 128 independent prior samples per minibatch. The inverse
change-of-variables calculation is log q_ay(z)=log p_ay(f_a^-1(z))
+ log|det Df_a^-1(z)|. Conditional KL matching is our adaptation of FNF,
not a claim of reproducing a published CheXpert protocol.

Use 60 epochs, batch 256, AdamW lr=.001, weight decay=.0001, clip norm=5,
and a 10-epoch gamma ramp. Choose the minimum **worst-label conditional
validation MMD**, with balanced group samples per class, after warmup and
subject to validation AUROC >=90% of matched ERM. If no checkpoint meets
that constraint, retain minimum-BCE fallback and explicitly mark failure
to satisfy selection. Report all settings; no test selection. Priors may
be imperfect, so fitted-density alignment does not certify population
conditional invariance. Conditional training need not remove marginal
sex information when label prevalence differs.

## Evaluation

Retain the RQ8 independent frozen probes: linear, boosted tree, RBF-SVM,
and three MLP capacities, with held-out tuning, separately for representation
and score, marginally and in each true class. Test labels are used to
evaluate conditional probes and metrics, never to encode test inputs.
Near-chance criteria retain the RQ8 multiplicity-adjusted uncertainty checks.
Report DP, EO, EOpp, TPR/FPR, AUROC/AUPRC overall and by sex, worst-group
utility, 15-bin ECE, Brier, threshold sensitivity, and age diagnostics.
Paired uncertainty uses 2,000 patient-cluster bootstrap draws.

## Calibration bias in the main tables

Adapt the user-supplied GLM implementation (unaltered reference is retained
as `calibration_reference.py`). Fit diagnostics on evaluation predictions;
do not apply those fitted recalibrations to predictions.

* **Calibration bias score / calibration-in-the-large**: intercept alpha
  in logit P(Y=1)=alpha+offset(logit R), slope fixed to one. Ideal zero;
  positive means underprediction, negative means overprediction.
* **Signed mean bias**: mean(R-Y). Ideal zero; positive means overprediction.
* Joint calibration intercept and slope: ideal (0,1), with the joint
  intercept not interpreted as an isolated prevalence-bias measure.
* Wald intervals use patient-clustered covariance, unlike the reference's
  default independent-row covariance. Undefined, separated, or failed fits
  are flagged, not silently reported as successful.

Backfill the same diagnostics for all 60 saved RQ8 runs (test and development)
without retraining or modifying their predictions. The supplied prior-shift
correction is not applied: these are raw-score calibration comparisons,
not a post-hoc recalibration experiment. The reference example's comment
that sigmoid(2z+1) should yield a positive joint intercept is reversed;
the correct expected intercept is -0.5, slope 0.5.

`tables/main_results_per_seed.csv` and `main_results_summary.csv` include
ECE, both sex-specific calibration bias scores, worst absolute bias score,
signed mean biases, utility, fairness and probes. Full fit diagnostics and
clustered intervals are in `tables/calibration_diagnostics.csv`.

## Run and audit

`python -m unittest experiments.rq8_conditional.tests -v`

Submit `smoke.sbatch`, then `python -m experiments.rq8_conditional.submit
--smoke-job JOBID`. Production is an 18-element Slurm array throttled to
eight one-GPU tasks, blocked until GPU smoke succeeds. Source and inputs
are recorded in the immutable run snapshot and manifests. CPU calibration
backfill and dependent table collection are submitted automatically.

Numerical tests cover Jacobians, analytic symmetric KL, conditional versus
marginal discrepancy, label-free inference, calibration sign and agreement
with the attached implementation. GPU integration exercises all three
strengths and a complete probe/uncertainty evaluation on synthetic data.

Theoretical context: [Petersen et al.](https://arxiv.org/abs/2305.01397).
Conditional invariance guarantees EO only when achieved, and does not
guarantee calibration or marginal independence. No result is presumed.
