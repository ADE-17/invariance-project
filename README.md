# Representational Invariance and Fairness Criteria: RQ6 and RQ8 experiments

Anonymous code release accompanying the paper submission. It contains the code, dataset
preparation records, seeds and per-run configurations for the two experiment families
reported in the paper:

* **RQ6, tabular Fair Normalizing Flows (FNF).** Marginal (Z ⊥ A) and class-conditional
  (Z ⊥ A | Y or Z ⊥ A | Ŷ) FNF encoders on seven tabular fairness datasets under four
  prevalence conditions, evaluated with an independent probe battery, fairness criteria
  (DP, EOpp, EOdds), score-distribution distances and calibration-bias diagnostics.
* **RQ8, CheXpert with frozen DenseNet-121 features.** Edema and Cardiomegaly with a 50 %
  female-positive deletion in every patient-disjoint split; FNF, MMD and adversarial erasers
  against matched ERM references, plus a deployable class-conditional FNF extension.

Everything else from the wider project (RQ1 to RQ5, RQ7) is excluded except the shared
helper modules that RQ6/RQ8 import.

## Layout

```
experiments/rq6_fnf/          RQ6: data.py (cohorts + conditions), fnf.py (flows, MADE, KL),
                              run.py (training shards), evaluate.py (probes, fairness,
                              calibration), calibration.py / calibration_bias.py (Cox slope
                              and intercept), aggregate.py, status.py, submit.py,
                              *.sbatch, figure and report scripts, test_rq6.py
experiments/rq8_final/        RQ8: prepare.py (patient-disjoint splits + shift), baseline.py
                              (DenseNet-121 ERM, frozen features), methods.py (FNF, MMD,
                              adversarial erasers), train.py, probes.py, evaluate.py, worker.py,
                              submit.py, aggregate.py, smoke.py, tests.py, METHODOLOGY.md
experiments/rq8_conditional/  RQ8 extension: class-conditional FNF on the same cohorts and
                              frozen features; calibration.py, backfill.py, tests.py
experiments/rq5_sweep/, rq5_guarantees/, rq5_removal/, rq4_ptbxl/
                              Shared utilities imported by RQ6/RQ8 (dataset loaders and
                              splitter, probe battery, metrics, bootstrap audit)
src/                          Shared metrics, calibration, probes, backbone and heads
configs/rq6/                  protocol.json, submission.json (job records), per-dataset
                              prepared/<dataset>/{manifest.json, conditions.csv, support.csv},
                              run_configs.csv (all 2,901 run configurations)
configs/rq8/shift50_v1/       configurations.json (54 erasers + 6 ERM), submission.json,
                              environment.txt (frozen pip environment), source_hashes.json,
                              prepared/<pathology>/{complete.json, support.csv, splits.csv.gz},
                              <pathology>/seed<k>/{baseline,runs/*}/config.json, shard_summary.json
configs/rq8/conditional_shift50_v1/
                              configurations.json (18 fits), submission.json, input_manifest.json,
                              preflight_validation.json, per-run config.json
```

The two experiment READMEs, `experiments/rq6_fnf/README.md` and
`experiments/rq8_final/README.md` (with `METHODOLOGY.md`), describe the protocols, the
hyper-parameter grids and the change log of the runs.

## Datasets, splits and seeds

Datasets are not redistributed. Placeholder paths in the code are `/path/to/...`; the
results root is `/path/to/results` (`RQ8_BASE`, `RQ8C_BASE` and `RQ6_CODE_ROOT` environment
variables override it where supported).

**RQ6.** `configs/rq6/prepared/<dataset>/manifest.json` records the source, row count,
SHA-256 of the prepared cohort, the attribute mapping, feature sets, quantile binning and
the fixed per-split condition seeds. `conditions.csv` lists the group sizes and prevalences
of every (condition, split). Model seeds are 42, 123 and 456 on fixed splits (60/10/10/20,
unit-disjoint). `run_configs.csv` flattens the `config.json` of every one of the 2,901 fits
(dataset, seed, condition, variant, γ, MADE width, feature set, conditioning variable,
optimiser settings).

**RQ8.** `configs/rq8/shift50_v1/prepared/<pathology>/complete.json` records the partition
seed (42), the shift seed (20260914), the deletion rule and the SHA-256 of every split file.
`splits.csv.gz` gives, for each CheXpert image path, its patient id, sex, label, uncertainty
flag and split (train, validation, probe_validation, selection, test, or removed_by_shift), so
the exact cohorts can be reconstructed from the public CheXpert-v1.0-small release without
re-running the sampler. Model seeds 42, 123, 456. Each `config.json` under
`<pathology>/seed<k>/` holds the eraser method, latent dimension, γ or ratio, and the
DenseNet-121 training settings of the matched ERM reference.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m unittest experiments.rq6_fnf.test_rq6 experiments.rq8_final.tests experiments.rq8_conditional.tests
```

## Running

RQ6 (run from the repository root):

```bash
python experiments/rq6_fnf/data.py                                  # build cohorts and conditions
python experiments/rq6_fnf/run.py --dataset adult --condition shift50 --smoke --allow-cpu
python experiments/rq6_fnf/run.py --index 0                         # one training shard
python experiments/rq6_fnf/run.py --index 0 --conditional           # class-conditional variant
python experiments/rq6_fnf/evaluate.py --index 0
python experiments/rq6_fnf/aggregate.py
```

RQ8:

```bash
python experiments/rq8_final/prepare.py                             # patient-disjoint shifted splits
python experiments/rq8_final/smoke.py --audit                       # synthetic end-to-end check
python experiments/rq8_final/worker.py --index 0                    # one (pathology, seed) shard
python experiments/rq8_final/aggregate.py
python -m experiments.rq8_conditional.worker --index 0              # class-conditional FNF extension
python -m experiments.rq8_conditional.backfill                      # calibration-bias diagnostics
python -m experiments.rq8_conditional.aggregate
```

The `*.sbatch` files are the Slurm array jobs used for the reported runs; `submit.py` in each
package snapshots the source, checks the smoke tests and launches the arrays. Adjust the
`#SBATCH` headers and paths for your cluster.
