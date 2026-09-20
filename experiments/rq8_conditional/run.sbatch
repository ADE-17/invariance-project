#!/bin/bash
#SBATCH --job-name=rq8c-fnf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/results/rq8/conditional_shift50_v1/logs/run-%A_%a.log
set -euo pipefail
cd /path/to/results/rq8/conditional_shift50_v1/source_snapshot
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -m experiments.rq8_conditional.worker --index "$SLURM_ARRAY_TASK_ID"
