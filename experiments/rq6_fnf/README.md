# RQ6: Fair Normalizing Flows, independence versus equalized odds

Execution documentation only. No results narrative, report or plots are generated here.

## Question

Does removing demographic information from the representation with an established
method (Fair Normalizing Flows, Balunovic, Ruoss & Vechev, ICLR 2022) achieve
independence (demographic parity, low probe decodability) while degrading
equalized odds when group base rates differ? This adds empirical evidence for
https://arxiv.org/abs/2305.01397 using a method whose independence bound rests on
estimated densities rather than on an adversary, after RQ5 showed DANN/CDANN
rarely achieve the invariance premise.

## Datasets and prevalence conditions

- `adult`: the RQ5 sweep cohort reused verbatim (48,842 rows, sex as attribute,
  income >50K). Categorical FNF as in the paper: six categorical features
  (`paper6`) or the same plus quantile-binned age (`paper6_age`).
- `health`: Heritage Health Prize release 3 (GitHub mirror of the original
  archive), aggregated to member-years. Attribute: age at first claim >=60.
  Label: any claim with Charlson index >0 in the member-year. Continuous FNF on
  14 numeric aggregates (`core`) or those plus specialty/place/procedure-group
  counts (`extended`). Splits are member-disjoint (60/10/10/20, RQ5 splitter).

Both datasets have natural base-rate gaps, so four conditions are run on the
source splits (train, validation, probe_validation) and, independently, as four
test variants on the natural test split: `natural`, `equalized` (higher-prevalence
group subsampled to the lower rate), `shift50` and `shift75` (50% / 75% of
A=0,Y=1 rows removed). Seeds are fixed per split. Every fit is evaluated on all
four test variants; `prepared/<dataset>/conditions.csv` records the resulting
group prevalences.

## Methods and hyperparameters

Model seeds 42, 123, 456 on fixed splits. Per shard (dataset x seed x condition):

- ERM per feature set (plain cross-entropy MLP on the input features).
- Health continuous FNF: gamma in {0, .05, .1, .5, .95}; prior in {GMM (4 comps,
  full covariance), RealNVP flow}; encoder depth in {4, 6} coupling blocks;
  80 epochs, KL warm-up over the first 10, paper preprocessing (affine to
  (alpha/2, 1-alpha/2), uniform dequantisation, logit). 40 FNF fits per shard.
- Adult categorical FNF: gamma in {0, .02, .1, .2, .5, .9, 1}; MADE hidden width
  in {50, 100}; rank-matching maps within logistic-regression predicted class
  (probability 1-gamma) or globally (probability gamma). 28 FNF fits per shard.

Priors are fitted once per (dataset, seed, condition, feature set, prior kind)
and cached under `priors/`. The paper's own post-hoc adversary is recorded for
comparability; the primary decodability measure is the independent probe battery.
FNF encodings require the sensitive attribute at encoding time; the summary
records this. Checkpoints are final-epoch, as in the paper.

## Evaluation

Independent linear, tree and three MLP probes (RQ5 code) fitted on train
representations with separate probe validation, marginally and within each
label, on every test variant. Task AUROC, accuracy, Brier, ECE, group AUROC,
DP, EOpp, EOdds, FPR gap at the validation Youden threshold and at .5, score KS
distances, 1,000 paired cluster-bootstrap draws for probe AUROC and task/fairness
metrics, and the RQ5 gate (corrected oriented upper AUROC <= .55) marginally and
conditionally. Statistical distance is saved from prior samples (paper) and
from held-out latents; the categorical variant records its exact MADE-based
distance and empirical total variation.

## Commands

```bash
cd .
PY=python
$PY -m unittest experiments.rq6_fnf.test_rq6
$PY experiments/rq6_fnf/data.py                       # cohorts (health needs raw/health/data.zip)
$PY experiments/rq6_fnf/run.py --dataset adult --condition shift50 --smoke --allow-cpu
$PY experiments/rq6_fnf/evaluate.py --dataset adult --condition shift50 --smoke
$PY experiments/rq6_fnf/submit.py --check-only
$PY experiments/rq6_fnf/submit.py                     # 24 GPU shards + dependent CPU evaluation
$PY experiments/rq6_fnf/status.py
$PY experiments/rq6_fnf/aggregate.py                  # CSV tables into results/rq6/
```

Storage: `/path/to/results/rq6/fnf_v1/` (`prepared/`, `priors/`,
`runs/<dataset>/seed<seed>/<condition>/<run>/`, `logs/`, `source_snapshot/`,
`submission.json`). Per run: `config.json`, `runtime.json`, `history.csv`,
`model.pt` or `classifier.pt` (+ `mapping.npz`), `representations.npz`,
`training_summary.json`, `evaluation/{metrics.json, predictions_*.csv.gz,
threshold_curve_*.csv, probe_audit.json, probes/}`.

## Change log

- 2026-09-12: `fnf.symmetric_kl` used `log p_X(x) + log|det dz/dx|` for the density of the mapped prior sample; the
  correct latent density is `log p_X(x) - log|det dz/dx|`. The sign error rewarded volume contraction, so Health
  (continuous variant) flows collapsed latent volume by about e^-28 per group, the logged KL reached -55 nats while the
  true symmetric KL was ~0.45, and group information was not removed (probe AUROC .87-.93). Adult (categorical variant)
  never calls `symmetric_kl` and is unaffected. Health outputs of the first submission are archived under
  `BASE/archive/runs_health_v1_klsign_bug/`; Health shards 12-23 were resubmitted (jobs 563002/563003) with the fixed
  code; priors were reused. Regression test: `test_symmetric_kl_matches_gaussian_shift_and_penalises_contraction`.
  `submit.py --shards a-b --note ...` performs targeted resubmissions.

- 2026-09-14: five further datasets added for the categorical variant (all a=1 = higher natural positive rate, kept as
  reference group): `compas` (ProPublica two-year recidivism, Caucasian vs African-American, 7 categorical features,
  5,278 rows), `lawschool` (LSAC bar passage, Non-White vs White; paper2 = LSAT/UGPA quantile-binned, extended6 adds
  family income, full-time, sex, tier; 20,798 rows), `crime` (UCI Communities & Crime, violent crime above median,
  a = racepctblack >= median, six socio-economic features binned into 5; 1,994 rows), `dutch` (Dutch Census 2001,
  high-level occupation, female vs male, 6-7 categorical features, 60,420 rows), `credit` (Taiwan credit default,
  female vs male, 5-8 features with quantile-binned amounts, 30,000 rows). Raw files under `BASE/raw/<dataset>/`
  (sources recorded in each manifest). Numeric features are quantile-binned on the training split via
  `manifest['binned_features']` (`run.categorical_codes`). Shards 24-83 submitted as jobs 564752/564753 from
  snapshot `source_snapshot_v3` with 6 concurrent GPUs. Same sweep as Adult: γ ∈ {0,.02,.1,.2,.5,.9,1} × MADE {50,100}
  × 2 feature sets + 2 ERM = 30 fits per shard, 1,800 fits.

- 2026-09-15: class-conditional variant `categorical_conditional` (`run.py --conditional`): per-(group, stratum) MADE
  densities and rank matching within each stratum of the conditioning variable, so Z ⊥ A | C. `condition_on='y'` uses the
  true label on every split (oracle upper bound for equalized odds; needs the label at encoding, including test);
  `condition_on='yhat'` uses the logistic-regression predicted class (deployable). Runs are named
  `cfnf_<fs>_cond-<y|yhat>_gamma<g>_made<h>`; training_summary records per-stratum exact TV and within-label empirical TV.
  Trial submitted (jobs 565429-565444, snapshot `source_snapshot_v4`): adult + dutch, seeds 42/123, all conditions,
  γ ∈ {.5, 1}.
  Stratum-size guard relaxed to >= 20 train / >= 5 validation rows per (group, stratum) after the cond-yhat runs on Adult
  shift75 failed (57 train / 13 validation female rows predicted positive); the snapshot was patched identically and the
  two Adult shift75 jobs re-run.

- 2026-09-18 — Calibration bias diagnostics: `calibration.py` adds Cox (1958) logistic recalibration slope and intercept (joint fit and fixed-slope "calibration-in-the-large" intercept, Wald CIs, mean bias), overall and per group, with group gaps; `evaluate.py` records them for new evaluations and `calibration_bias.py` fills them post hoc for all existing runs from the saved predictions (`evaluation/calibration_bias.json`, aggregated to `results/rq6/paper/tables/calibration_bias_all_runs.csv.gz`). Pure NumPy/SciPy Newton fits (no statsmodels in the venv); regression test `TestCalibration`.
- 2026-09-18 — Class-conditional FNF extended to all five paper datasets and three seeds (`conditional.sbatch`, array 571199, 6 concurrent; code from `source_snapshot_v4`): crime/credit/compas × seeds 42/123/456 and adult/dutch seed 456, all four prevalence conditions. Recorded in `submission.json['conditional_trial'][1]`.
- 2026-09-18 — Class-conditional extension complete (array 571199: 44 shards, 0 failures; 240 conditional fits in total). `CONDITIONAL_TRIAL.md` regenerated for five datasets × three seeds with calibration columns; `PAPER_REPORT.md` §8 rewritten (8.1 cross-dataset summary, 8.2 per-group calibration, 8.3 stratum diagnostics, 8.4 interpretation). Draft LaTeX rows with three-seed means and per-group calibration intercepts: `interpretation/paper_draft/fnf_main_table_rows_3seeds.tex`.
- 2026-09-18 — `paper_summary_figures.py`: three paper figures for the marginal vs class-conditional comparison (`results/rq6/paper/figures/summary_criteria_planes`, `summary_prevalence_gap`, `summary_criteria_dots`, PNG+PDF) and the per-seed source table `paper/tables/summary_figures_per_seed.csv` (adds ΔPPV/ΔNPV and base-rate gap). LaTeX figure environments with captions: `interpretation/paper_draft/fnf_summary_figures.tex`.
- 2026-09-19 — `cross_condition_figure.py`: deployment-shift analysis (train condition × test variant) for Dutch and Adult → `paper/figures/deployment_shift.{png,pdf}`, `deployment_shift_heatmaps.{png,pdf}`, `paper/tables/cross_condition_per_seed.csv`, LaTeX table `interpretation/paper_draft/fnf_deployment_shift_table.tex`. Revised results section draft: `interpretation/paper_draft/results_fnf_revised.tex`.
- 2026-09-19 — Paper-style restyle: `cross_condition_figure.py` deployment-shift figure now three methods (ERM, Z⊥A, Z⊥A|Y), STIX serif, 13–16 pt, compact layout. New `distribution_lines.py`: Petersen-style continuous KDE score densities p(r|A) and p(r|A,Y) for ERM / Z⊥A / Z⊥A|Y per dataset → `paper/figures/score_densities_<ds>.{png,pdf}`; caption appended to `interpretation/paper_draft/fnf_summary_figures.tex`.
