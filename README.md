# Marginal vs. Conditional Representation Alignment

Code, configs and seeds for the two experiments in the paper:

| Experiment | Criterion | Tabular (7 datasets) | CheXpert (Edema, Cardiomegaly) |
|---|---|---|---|
| **Marginal alignment** | Z ⊥ A | `experiments/rq6_fnf/run.py` | `experiments/rq8_final/` (FNF, MMD, adversarial vs. ERM) |
| **Conditional alignment** | Z ⊥ A \| Y | `experiments/rq6_fnf/run.py --conditional` | `experiments/rq8_conditional/` (class-conditional FNF) |

Both are evaluated with the same independent probe battery, fairness metrics (DP, EOpp, EOdds),
calibration-bias diagnostics and paired cluster bootstrap.

## Where is what

```
experiments/rq6_fnf/           tabular experiments (both criteria)
  data.py                        build cohorts + prevalence conditions (natural, equalized, shift50, shift75)
  fnf.py                         flows, MADE densities, KL / TV objectives
  run.py                         train one shard   (--conditional → conditional alignment)
  evaluate.py                    probes, fairness, calibration for one shard
  aggregate.py                   CSV tables
  calibration.py                 Cox calibration slope / intercept
  train.sbatch, conditional.sbatch, evaluate.sbatch, submit.py   cluster jobs
  test_rq6.py                    unit tests
experiments/rq8_final/         CheXpert, marginal alignment
  prepare.py                     patient-disjoint splits + 50 % female-positive deletion
  baseline.py                    DenseNet-121 ERM, frozen features
  methods.py / train.py          FNF, MMD, adversarial erasers
  evaluate.py / probes.py        evaluation
  worker.py, run.sbatch, submit.py, aggregate.py, smoke.py, tests.py
  METHODOLOGY.md                 fixed protocol
experiments/rq8_conditional/   CheXpert, conditional alignment (same splits and features)
  train.py, methods.py, evaluate.py, worker.py, backfill.py (calibration), aggregate.py, tests.py
experiments/rq4_ptbxl/, rq5_*/  shared helpers (loaders, splitter, probes, metrics, bootstrap)
src/                           shared metrics, calibration, probes, model heads

configs/rq6/                   tabular
  protocol.json                  grids: datasets, seeds, conditions, γ, MADE widths, feature sets
  prepared/<dataset>/            manifest.json (source, SHA-256, split seeds, features), conditions.csv (group sizes, prevalences)
  run_configs.csv                every run configuration (2,901 rows)
  submission.json                job records
configs/rq8/shift50_v1/        CheXpert, marginal
  configurations.json            54 erasers + 6 ERM (method, dim, γ / ratio, seed)
  prepared/<pathology>/          complete.json (partition seed 42, shift seed 20260914, hashes),
                                 splits.csv.gz (image path → patient, sex, label, split)
  <pathology>/seed<k>/           config.json of the baseline and of every run
  environment.txt                frozen pip environment
configs/rq8/conditional_shift50_v1/   CheXpert, conditional: configurations.json (18 fits), per-run config.json
```

Seeds: 42, 123, 456 for all models. Splits are fixed (seeds recorded in the manifests above).
Datasets are not included; replace `/path/to/...` in the code or set `RQ8_BASE` / `RQ8C_BASE`.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m unittest experiments.rq6_fnf.test_rq6 experiments.rq8_final.tests experiments.rq8_conditional.tests
```

## Run

Tabular, from the repository root:

```bash
python experiments/rq6_fnf/data.py                        # 1. cohorts and conditions
python experiments/rq6_fnf/run.py --index 0               # 2. marginal alignment, shard 0
python experiments/rq6_fnf/run.py --index 0 --conditional #    conditional alignment, shard 0
python experiments/rq6_fnf/evaluate.py --index 0          # 3. evaluate the shard
python experiments/rq6_fnf/aggregate.py                   # 4. tables
```

CheXpert:

```bash
python experiments/rq8_final/prepare.py                   # 1. splits with shift
python experiments/rq8_final/worker.py --index 0          # 2. marginal: ERM + erasers for one (pathology, seed)
python experiments/rq8_final/aggregate.py
python -m experiments.rq8_conditional.worker --index 0    # 3. conditional: reuses the splits and features
python -m experiments.rq8_conditional.backfill            #    calibration diagnostics
python -m experiments.rq8_conditional.aggregate
```

Shard indices follow `configs/*/configurations.json` and `configs/rq6/protocol.json`.
The `*.sbatch` files are the Slurm arrays used for the reported runs.
