#!/bin/bash
#SBATCH --job-name=rq8c-collect
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/results/rq8/conditional_shift50_v1/logs/collect-%j.log
set -euo pipefail
cd /path/to/results/rq8/conditional_shift50_v1/source_snapshot
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
python -m experiments.rq8_conditional.aggregate
