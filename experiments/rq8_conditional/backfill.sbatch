#!/bin/bash
#SBATCH --job-name=rq8c-calibration
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/results/rq8/conditional_shift50_v1/logs/calibration-%j.log
set -euo pipefail
cd /path/to/results/rq8/conditional_shift50_v1/source_snapshot
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
python -m experiments.rq8_conditional.backfill
