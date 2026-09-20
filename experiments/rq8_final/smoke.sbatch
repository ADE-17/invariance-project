#!/bin/bash
#SBATCH --job-name=rq8-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/path/to/results/rq8/shift50_v1/logs/smoke-%A_%a.log
set -euo pipefail
cd .
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -u experiments/rq8_final/smoke.py --audit
