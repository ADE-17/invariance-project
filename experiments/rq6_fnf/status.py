"""Per-shard completion counts for training and evaluation."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rq6_fnf.data import BASE, load
from experiments.rq6_fnf.run import SHARDS, configurations

for i, s in enumerate(SHARDS):
    root = BASE / 'runs' / s['dataset'] / f"seed{s['seed']}" / s['condition']
    expected = len(configurations(s['dataset'], list(json.loads((BASE / 'prepared' / s['dataset'] / 'manifest.json').read_text())['feature_sets'])))
    trained = len(list(root.glob('*/complete.json')))
    evaluated = len(list(root.glob("*/evaluation/complete.json")))
    failed = len(list(root.glob("*/failed.json")))
    print(f"{i:2d} {s['dataset']:7s} seed{s['seed']:<5d} {s['condition']:10s} trained {trained:2d}/{expected} evaluated {evaluated:2d}/{expected} failed {failed}")
