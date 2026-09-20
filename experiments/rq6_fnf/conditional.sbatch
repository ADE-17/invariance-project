#!/bin/bash
#SBATCH --job-name=rq6-cfnf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/results/rq6/fnf_v1/logs/cfnf-array-%A_%a.log
# Class-conditional FNF extension: remaining datasets and seeds. Index -> (dataset, seed, condition).
set -euo pipefail
SHARDS=(
  adult:456 dutch:456
  crime:42 crime:123 crime:456
  credit:42 credit:123 credit:456
  compas:42 compas:123 compas:456
)
CONDS=(natural equalized shift50 shift75)
i=$SLURM_ARRAY_TASK_ID
entry=${SHARDS[$((i / 4))]}; cond=${CONDS[$((i % 4))]}
ds=${entry%%:*}; seed=${entry##*:}
echo "shard $i -> $ds seed $seed $cond"
cd "${RQ6_CODE_ROOT:-/path/to/results/rq6/fnf_v1/source_snapshot_v4}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=python
$PY -u experiments/rq6_fnf/run.py --dataset "$ds" --seed "$seed" --condition "$cond" --conditional --allow-cpu
$PY -u experiments/rq6_fnf/evaluate.py --dataset "$ds" --seed "$seed" --condition "$cond"
