#!/bin/bash
#SBATCH --job-name=rq8-collect
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/results/rq8/shift50_v1/logs/collect-%j.log
set -euo pipefail
cd /path/to/results/rq8/shift50_v1/source_snapshot
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python experiments/rq8_final/aggregate.py
