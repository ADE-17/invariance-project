# RQ8 final shifted CheXpert sweep

See [the fixed methodology and evaluation protocol](METHODOLOGY.md).

- Edema and Cardiomegaly; 50% female-positive deletion in every patient-disjoint split.
- FNF (8-D), MMD (2-D), alternating adversarial ensemble (32-D).
- Three settings × three seeds × two pathologies = 54 erasers; six matched shifted ERM references.
- Six simultaneous GPU shards; full independent demographic probes and paired utility/fairness/calibration evaluation.
- `tests.py` contains numerical regression checks; `smoke.py` tests all selected method families and complete evaluation on isolated synthetic data.
- `submit.py` requires successful GPU smoke checks and prepared cohorts, snapshots source, then launches the sweep and dependent CPU collection.
- Outputs: `/path/to/results/rq8/shift50_v1/`.

All 54 erasers and six ERM references completed, with 60 evaluations and no recorded execution failures. See the [completed report with interpretation, tables and plots](../../interpretation/rq8_report/report.md).


## Conditional FNF extension and calibration-bias diagnostics

See [the conditional extension](../rq8_conditional/README.md) for the new
18-run deployable conditional FNF study on the same RQ8 cohorts. Existing
RQ8 predictions are reused unchanged. ECE and sex-specific calibration-in-the-large
intercepts are included together in the new comparison tables at
`/path/to/results/rq8/conditional_shift50_v1/tables/`.
