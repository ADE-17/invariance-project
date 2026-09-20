"""Freeze a code snapshot and submit GPU training shards plus dependent CPU evaluation shards."""
from pathlib import Path
import argparse
import datetime
import json
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rq5_sweep.data import sha
from experiments.rq5_sweep.run import write_json
from experiments.rq6_fnf.data import BASE, DATASETS, CONDITIONS, load
from experiments.rq6_fnf.run import SHARDS, SEEDS, CONTINUOUS_GAMMAS, CATEGORICAL_GAMMAS, PRIORS, N_BLOCKS, MADE_HIDDEN, CONTINUOUS, CATEGORICAL, configurations


def sbatch(args):
    result = subprocess.run(['sbatch', '--parsable', *args], check=True, capture_output=True, text=True)
    return result.stdout.strip().split(';')[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-gpus', type=int, default=8)
    p.add_argument('--max-cpu-jobs', type=int, default=12)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--shards', help='Targeted resubmission, e.g. 12-23 (requires --note); a fresh snapshot is frozen')
    p.add_argument('--note', help='Reason for the resubmission, recorded in submission.json')
    p.add_argument('--snapshot', default='source_snapshot', help='Snapshot folder name under BASE (use a new name while older jobs still run from the default)')
    args = p.parse_args()
    if (BASE / 'submission.json').exists() and not args.shards:
        raise RuntimeError('Already submitted; use --shards a-b --note ... for a targeted resubmission.')
    if args.shards and not args.note:
        raise RuntimeError('--shards requires --note')
    array = args.shards or f'0-{len(SHARDS)-1}'
    root = Path(__file__).resolve().parents[2]
    manifests = {}
    for d in DATASETS:
        _, manifests[d] = load(d)
    runs = {d: len(configurations(d, list(manifests[d]['feature_sets']))) for d in DATASETS}
    plan = dict(datasets=DATASETS, seeds=SEEDS, conditions=CONDITIONS, shards=len(SHARDS),
                runs_per_shard=runs, total_fits=sum(runs[s['dataset']] for s in SHARDS),
                test_variants=[f'test_{c}' for c in CONDITIONS],
                continuous=dict(gammas=CONTINUOUS_GAMMAS, priors=PRIORS, n_blocks=N_BLOCKS, **CONTINUOUS),
                categorical=dict(gammas=CATEGORICAL_GAMMAS, made_hidden=MADE_HIDDEN, **CATEGORICAL),
                checkpoint_policy='final epoch (paper protocol); full validation history saved',
                shift='shiftXX removes XX% of A=0,Y=1 rows from train/validation/probe_validation with fixed seeds; '
                      'equalized subsamples the higher-prevalence group to the lower group rate; test variants built identically on the natural test split',
                probe_families=['linear', 'tree', 'mlp0', 'mlp1', 'mlp2'], bootstrap_replicates=1000, gate='corrected oriented upper AUROC <= .55',
                encoding_requires_sensitive_attribute=True,
                scope='Empirical pilot; no guarantee of achieved independence; FNF bounds hold only for the estimated priors',
                limitations=['Three seeds measure fitting variability on fixed splits.',
                             'Adult has a natural base-rate gap; equalized removes it by subsampling.',
                             'Health label derives from claim-level Charlson codes; Charlson is not a feature.',
                             'Health feature dimensionality (core/extended) tests prior estimation difficulty; both are study variants.',
                             'Categorical Adult follows the paper: continuous attributes are dropped or quantile-binned.',
                             'Probes establish family-limited decodability, not statistical independence.'],
                sources=['https://arxiv.org/abs/2106.05937', 'https://github.com/eth-sri/fnf',
                         'https://github.com/truongkhanhduy95/Heritage-Health-Prize (HHP release 3 mirror)',
                         'https://archive.ics.uci.edu/dataset/2/adult'],
                automated_reports=False, automatic_plots=False, max_concurrent_gpus=args.max_gpus)
    if args.check_only:
        print(json.dumps(plan, indent=2))
        return
    snapshot = BASE / args.snapshot
    if snapshot.exists():
        raise RuntimeError('Snapshot exists; archive it (mv to archive/) or pass a new --snapshot name before resubmitting.')
    for rel in ['src', 'experiments/rq4_ptbxl', 'experiments/rq5_guarantees', 'experiments/rq5_sweep', 'experiments/rq5_removal', 'experiments/rq6_fnf']:
        shutil.copytree(root / rel, snapshot / rel, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {str(f.relative_to(snapshot)): sha(f) for f in snapshot.rglob('*') if f.is_file()}
    subprocess.run([sys.executable, str(snapshot / 'experiments/rq6_fnf/run.py'), '--help'], check=True, stdout=subprocess.DEVNULL)
    write_json(BASE / f'{args.snapshot}_hashes.json', hashes)
    write_json(BASE / 'protocol.json', plan)
    (BASE / 'logs').mkdir(exist_ok=True)
    env = f'ALL,RQ6_CODE_ROOT={snapshot}'
    scripts = snapshot / 'experiments/rq6_fnf'
    training = sbatch([f'--array={array}%{args.max_gpus}', '--export=' + env, str(scripts / 'train.sbatch')])
    record = dict(training_job=training, code_snapshot=str(snapshot), shards=array,
                  submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), expected_fits=plan['total_fits'])
    if args.shards:
        submission = json.loads((BASE / 'submission.json').read_text())
        record['note'] = args.note
        submission.setdefault('resubmissions', []).append(record)
    else:
        submission = record
    write_json(BASE / 'submission.json', submission)
    evaluation = sbatch([f'--array={array}%{args.max_cpu_jobs}', f'--dependency=aftercorr:{training}', '--export=' + env, str(scripts / 'evaluate.sbatch')])
    record['evaluation_job'] = evaluation
    write_json(BASE / 'submission.json', submission)
    print(json.dumps(submission, indent=2))


if __name__ == '__main__':
    main()
