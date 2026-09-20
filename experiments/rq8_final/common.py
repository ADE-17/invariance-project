from pathlib import Path
import json,hashlib,os
import numpy as np
import pandas as pd

BASE=Path(os.environ.get('RQ8_BASE','/path/to/results/rq8/shift50_v1'))
SEED=int(os.environ.get('RQ8_SEED','42'))
PATHOLOGY=os.environ.get('RQ8_PATHOLOGY','Edema')
assert PATHOLOGY in ['Edema','Cardiomegaly']
ROOT=BASE/PATHOLOGY/f'seed{SEED}'
DATA=Path('/path/to/CheXpert-v1.0-small')
SPLITS=['train','validation','selection','probe_validation','test']

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def cv(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=cv,allow_nan=False));tmp.replace(path)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def frames():return {s:pd.read_csv(ROOT/'prepared'/f'{s}.csv') for s in SPLITS}
def load_features():return {s:np.load(ROOT/'baseline'/f'{s}_z.npy',mmap_mode='r') for s in SPLITS}
def log(m):
    from datetime import datetime,timezone
    print(datetime.now(timezone.utc).isoformat(),m,flush=True)
