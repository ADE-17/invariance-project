# RQ8: fixed protocol for shifted CheXpert experiments

This protocol fixes the grid before launching RQ8. It is an execution specification, not a results report.

## Scope and cohorts

- Pathologies: Edema and Cardiomegaly; one binary task per model.
- Sensitive attribute: recorded sex, Female = 0 and Male = 1. This measures recorded sex, not self-identified gender.
- CheXpert-small training CSV, frontal views only, known sex, U-ones labels (uncertain = positive; blank = negative), 256 × 256 RGB images.
- Patient partition: 60% training, 5% checkpoint/threshold validation, 5% development selection, 10% independent probe validation, 20% test. Partition seed 42, fixed across model seeds and pathologies before deletion.
- Shift50 means deleting `floor(N_female_positive / 2)` female-positive **images**, separately in every split. Male records and female negatives remain. The final prevalence is not half the original prevalence because the denominator also changes. Deletion seeds are 20260914 + split index; model seeds do not change cohorts.
- No natural-prevalence model or evaluation branch is scheduled. Before-deletion counts are retained only to audit cohort construction.
- Image IDs, patient IDs, removed rows, split hashes, support counts and input metadata hash are saved under `prepared/`.
- Training seeds: 42, 123, 456. These estimate optimization variation on fixed cohorts, not variation across randomly resampled datasets.

## Selected methods and fixed grid

| Method | Representation | Three settings | Reason for inclusion |
|---|---|---|---|
| Fair Normalizing Flows (FNF) | 8-D whitened PCA followed by sex-specific invertible flows | gamma = 0.05, 0.5, 0.95 | Requested method; RQ7 retained strong task utility, although residual sex leakage remained. |
| MMD adapter | 1024 → 256 → 2, shared linear task head | initial gradient ratios = 0.1, 0.3, 1 | RQ7 2-D/0.3 setting passed both development and test probe-AUROC gates with limited utility loss. |
| Alternating adversarial ensemble | 1024 → 256 → 32, shared linear task head | initial gradient ratios = 0.3, 1, 3 | RQ7 ratio 1 retained the most task utility among promising erasers; its development confidence gate still failed. |

This is 54 erasure fits and six matched ERM references, with 60 complete evaluations. Every setting is retained, including budget failures, divergence and incomplete runs. FNF is not assumed to be a successful eraser in advance.

All methods transform frozen features from the matched shifted DenseNet121. MMD and adversarial training update their adapter and task head; FNF updates flows and its task head. The CNN is frozen during these stages. Demographic erasure claims therefore concern the **released final representation or task score**, not every internal CNN layer.

### Matched ERM

The six references use ImageNet DenseNet121, AdamW learning rate 1e-4, weight decay 1e-5, batch 32, maximum 30 epochs. Select minimum validation BCE; stop with patience seven after at least ten epochs. Features and probabilities are cached and shared by all nine erasers for that pathology/seed.

RQ7's ERM used natural-prevalence Edema training. Earlier RQ1 shifted ERMs used seed-specific 70/15/15 patient splits, included lateral views, shifted training only, and do not provide matching cohort manifests. Reusing them would confound baseline differences with data differences. RQ8 therefore needs six fresh references; it does not repeat ERM per hyperparameter.

### FNF objective and implementation

Refer to [Fair Normalizing Flows](https://arxiv.org/abs/2106.05937) and the [authors' implementation](https://github.com/eth-sri/fnf). The local flow primitives come from `experiments/rq6_fnf/fnf.py`; the corrected density objective is local to `methods.py` and numerically tested. RQ7 recorded upstream FNF commit `f57401262dc12b258353b26b308ec02f618433d6`.

Fit the feature scaler, 8-D whitened PCA and two four-component full-covariance Gaussian-mixture priors on training data only. Use four affine coupling blocks, two hidden layers of width 128 and matched initial flow weights. The shared head is 8 → 64 → 1. Each sex uses its own flow at training **and inference**, so FNF requires recorded sex to select the transformation.

For group flows f_a, minimize `(1-gamma_t) * BCE + gamma_t * symmetric_KL / 8`, where gamma ramps to its specified value over ten epochs. The symmetric KL is half the sum of the two directional latent-density KL divergences, estimated using prior samples. If encode returns `log|det Df(x)|`, the correct transformed density is `log p_Z(f(x)) = log p_X(x) - log|det Df(x)|`. Analytic Gaussian and round-trip tests check this sign.

Use AdamW 1e-3, weight decay 1e-4, batch 256, 60 epochs, 128 prior samples per group/update, gradient clipping at norm five. Save GMM convergence indicators and KL history. The prior-model KL objective does not certify empirical test independence; held-out probes remain necessary. This is an adaptation of FNF to cached chest-X-ray features, not an exact reproduction of the original experiments.

### MMD objective

Refer to the MMD dependence penalty in [The Variational Fair Autoencoder](https://arxiv.org/abs/1511.00830). RQ8 uses a deterministic supervised adapter, not the complete VFAE architecture. Minimize `BCE + lambda_t * MMD²(Z_female, Z_male)`.

The biased empirical MMD estimate includes diagonal kernel terms. The kernel averages five Gaussian bandwidth multipliers (0.25, 0.5, 1, 2, 4), using detached batch median squared distance after coordinate standardization. AdamW 1e-3, weight decay 1e-4, batch 256, 50 epochs, norm-five clipping.

### Adversarial ensemble objective

Refer to [Learning Adversarially Fair and Transferable Representations](https://proceedings.mlr.press/v80/madras18a.html). This implementation is an ensemble/alternating-update adaptation. The sex adversaries are linear, a one-hidden-layer 128-unit MLP, and a 256 → 128 → 64 MLP. Use five discriminator updates per adapter update. Average their sex-balanced BCE losses; the adapter minimizes task BCE minus the weighted mean adversarial BCE. Freeze adversary parameters during the adapter update, while allowing gradients through them into the adapter. These training adversaries are distinct from all evaluation probes.

For MMD and adversarial training, set lambda once using the first minibatch: `ratio * ||grad_encoder(task BCE)|| / max(||grad_encoder(penalty)||, 1e-8)`. Ramp it over five epochs. Record the actual numeric lambda and both gradient norms. The three grid values are gradient **ratios**, not raw lambdas. Training lengths, optimizer and clipping match the MMD adapter.

## Selection and evaluation

- Use validation checkpoint AUROC at least 90% of matched ERM AUROC as the utility budget. Among eligible checkpoints after warmup, select lowest validation MMD; otherwise save the lowest-BCE fallback and flag it. A validation budget pass does not guarantee a test pass. Save absolute and relative AUROC loss, AUPRC, and subgroup utility, so the AUROC budget cannot conceal AUPRC collapse.
- Report all three hyperparameter configurations. Any subsequent single-configuration choice must use development selection results, not test performance.
- Fit independent logistic probes (three C settings), histogram gradient-boosted trees (three leaf settings), RBF-SVM probes and three separately trained MLP capacities. Tune on the independent probe-validation patient set, never test.
- Probe training uses a fixed random subset of up to 40,000 training images; RBF-SVM fitting is capped at 2,000 per sex. Audit marginal and Y=0/Y=1 conditional decoding of both final representations and task scores. Keep permuted-label controls and frozen-feature hashes. Probes never update the representation.
- The exploratory near-chance criterion is an oriented probe-AUROC upper bound ≤ 0.55 across all six probe families. Correct within the marginal family set or the two conditional class/family sets, not across the entire experimental sweep. Also save balanced accuracy and balanced log-loss gains with intervals. Passing finite probes is not proof of independence.
- Threshold-dependent metrics use each model's validation-selected threshold. Also save fixed-threshold 0.5 metrics and same-threshold paired curves at 21 thresholds. No thresholds are selected on test.
- DP gap: absolute difference in positive prediction rates. EOpp: absolute TPR difference. EOdds (EO in this protocol): maximum of TPR and FPR gaps. Save each group's signed rates and changes to identify leveling down.
- Calibration: overall ECE (15 equal-width bins; 10/20-bin sensitivity), each group's ECE, maximum group ECE, absolute group-ECE gap, Brier score, calibration bias, calibration-in-the-large, intercept and slope. Undefined/unstable slope fits are represented as null rather than invented estimates.
- `delta_vs_erm` always means method minus matched ERM. Thus negative delta DP/EOdds/ECE means improvement in that gap/error; negative delta AUROC/AUPRC means lost utility. This separates the fairness gap itself from the change in that gap.
- Save 2,000 paired patient-bootstrap replicates/intervals for task utility, DP/EOdds, group rates, ECE and Brier metrics, and demographic probe AUROC. Save overall and worst-group AUROC/AUPRC, age subsets and a certain-label sensitivity subset. Calibration slopes/CITL are point estimates; they are not bootstrap intervals.

## Relation to the paper and limits

[Petersen et al.](https://arxiv.org/abs/2305.01397) motivates testing the relationship among invariance, fairness and utility. This experiment tests achieved demographic decodability rather than treating a method name as evidence of invariance. Exact marginal independence with a shared task head implies demographic parity; exact label-conditional independence implies equalized odds on the same distribution. Finite-probe measurements and these practical interventions do not establish either exact premise, nor do they prove invariant models are always unfair.

RQ8 reuses the master patient partition already inspected during exploratory RQ7, with a newly fixed prevalence shift. Its test is therefore not an untouched external confirmatory cohort. Keep this limitation explicit in the final paper; the three new optimization seeds do not restore an unseen test set.

## Execution and artifacts

Six GPU shards run concurrently (one pathology/seed each), with one GPU, eight allocated CPUs and 48 GB host memory each, a 48-hour wall-time limit, four compute threads and four training data workers. Each shard runs its baseline, raw-ERM audit and nine train/audit pairs sequentially. A CPU collection job runs after all shards terminate, including after failures. No other experiment's jobs are modified.

Results root: `/path/to/results/rq8/shift50_v1/`. Immutable `source_snapshot/`, `source_hashes.json`, `environment.txt`, `configurations.json` and `submission.json` record provenance. Each run saves configuration, models, training history, per-image representations/probabilities, frozen probe models, subgroup/threshold tables, metrics, paired bootstrap arrays and completion/failure markers. Aggregate CSVs are numerical outputs, not a manuscript.

```bash
python experiments/rq8_final/status.py
python experiments/rq8_final/aggregate.py
```
