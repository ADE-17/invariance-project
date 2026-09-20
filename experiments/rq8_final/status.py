import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.rq8_final.common import BASE
if (BASE/'submission.json').exists():
    sub=json.loads((BASE/'submission.json').read_text());subprocess.run(['squeue','-j',sub['sweep_job'],'-o','%.18i %.10T %.10M %.20R'])
for p in ['Edema','Cardiomegaly']:
    for s in [42,123,456]:
        root=BASE/p/f'seed{s}';base=root/'baseline';hist=base/'history.csv'
        print(p,s,'baseline_complete=',(base/'complete.json').exists(),'epochs=',max(0,len(hist.read_text().splitlines())-1) if hist.exists() else 0,'evaluated=',len(list(root.glob('runs/*/evaluation/complete.json'))),'/10','failures=',len(list(root.glob('runs/*/failure.json'))))
