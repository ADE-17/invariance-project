#!/bin/bash
#SBATCH --job-name=rq8c-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/results/rq8/conditional_shift50_v1/logs/smoke-%j.log
set -euo pipefail
cd .
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
python -m experiments.rq8_conditional.smoke
