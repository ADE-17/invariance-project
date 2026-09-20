#!/bin/bash
#SBATCH --job-name=rq8-shift50-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/results/rq8/shift50_v1/logs/run-%A_%a.log
set -euo pipefail
cd /path/to/results/rq8/shift50_v1/source_snapshot
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -u experiments/rq8_final/worker.py --index "$SLURM_ARRAY_TASK_ID"
