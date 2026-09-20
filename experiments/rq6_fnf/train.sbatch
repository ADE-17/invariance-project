#!/bin/bash
#SBATCH --job-name=rq6-fnf-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/results/rq6/fnf_v1/logs/train-%A_%a.log
set -euo pipefail
cd "${RQ6_CODE_ROOT:-.}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
trap 'rc=$?; printf "{\"exit_code\": %s, \"array_index\": %s}\n" "$rc" "$SLURM_ARRAY_TASK_ID" > "/path/to/results/rq6/fnf_v1/logs/train-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.status.json"; exit "$rc"' EXIT
python -u experiments/rq6_fnf/run.py --index "$SLURM_ARRAY_TASK_ID" "$@"
