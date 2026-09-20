"""Collect numerical artifacts, retain failures; no manuscript or interpretation."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
from experiments.rq8_final.common import BASE,write

def main():
    out=BASE/'tables';out.mkdir(exist_ok=True);rows=[];probes=[];failures=[]
    for p in BASE.glob('*/seed*/runs/*/evaluation/*_metrics.json'):
        d=json.loads(p.read_text());c=d['config'];row=dict(**c,split=d['split'],source=str(p))
        for key in ['metrics','erm','delta_vs_erm','threshold_05','delta_vs_erm_threshold_05']:
            row.update({f'{key}__{k}':v for k,v in d[key].items()})
        row.update({k:d[k] for k in ['utility_relative_auroc_drop','within_10percent_auroc_budget','additional_collapse_flag']})
        for variant,gates in d['gates'].items():
            for scope,values in gates.items():row.update({f'{variant}__{scope}__{k}':v for k,v in values.items()})
        for key,ci in d['task_intervals'].items():row.update({f'ci__{key}__{k}':v for k,v in ci.items()})
        rows.append(row)
        for name,values in d['probes'].items():
            probes.append(dict(**c,split=d['split'],probe=name,**{k:v for k,v in values.items() if not isinstance(v,dict)},delta_oriented_auroc=d.get('probe_delta_vs_erm',{}).get(name)))
    for p in BASE.glob('*/seed*/**/*failure.json'):
        failures.append(dict(path=str(p),**json.loads(p.read_text())))
    pd.DataFrame(rows).to_csv(out/'all_metrics.csv',index=False)
    pd.DataFrame(probes).to_csv(out/'all_probes.csv',index=False)
    write(out/'failures.json',failures)
    write(out/'status.json',dict(expected_fits=54,expected_references=6,completed_evaluations=len(list(BASE.glob('*/seed*/runs/*/evaluation/complete.json'))),expected_evaluations=60,metric_rows=len(rows),failures=len(failures)))
    print((out/'status.json').read_text())

if __name__=='__main__':main()
