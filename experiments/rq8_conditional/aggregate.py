"""Numerical comparison tables with ECE and calibration bias; no manuscript text."""
import json
from pathlib import Path
import pandas as pd
from .common import BASE,write
from .backfill import REFERENCE


def main():
    out=BASE/'tables';out.mkdir(exist_ok=True)
    rows=[];details=[]
    files=[('rq8',p) for p in REFERENCE.glob('*/seed*/runs/*/evaluation/*_metrics.json')]
    files += [('conditional',p) for p in BASE.glob('*/seed*/runs/*/evaluation/*_metrics.json')]
    for experiment,f in sorted(files,key=lambda pair:str(pair[1])):
        d=json.loads(f.read_text());c=d['config'];metrics=dict(d['metrics'])
        if experiment=='rq8':
            cf=BASE/'reference_calibration'/c['pathology']/f'seed{c["seed"]}'/c['id']/f'{d["split"]}.json'
            if not cf.exists():raise RuntimeError(f'Missing calibration supplement: {cf}')
            cd=json.loads(cf.read_text());metrics.update(cd['metrics']);cal=cd['weak_calibration']
        else:cal=d['weak_calibration']
        row=dict(experiment=experiment,**c,split=d['split'],source=str(f),**metrics)
        for variant in ['representation','score']:
            for scope in ['marginal','conditional']:
                gate=d['gates'][variant][scope]
                row[f'{variant}_{scope}_probe_auroc']=gate['max_oriented_auroc']
                row[f'{variant}_{scope}_gate']=gate['auroc_gate']
        # Signed changes in comparable model metrics against the exact same ERM.
        for key,value in metrics.items():
            baseline=d['erm'].get(key)
            if isinstance(value,(int,float)) and isinstance(baseline,(int,float)):
                row['delta_'+key]=value-baseline
        for group,cd in cal.items():
            details.append(dict(experiment=experiment,**c,split=d['split'],group=group,
                                **{k:v for k,v in cd.items() if not isinstance(v,list)}))
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(out/'all_metrics.csv',index=False)
    pd.DataFrame(details).to_csv(out/'calibration_diagnostics.csv',index=False)
    main_cols=['auroc','auprc','worst_group_auroc','dp','eopp','eodds','ece','max_group_ece',
        'group0_calibration_bias_score','group1_calibration_bias_score','worst_abs_calibration_bias_score',
        'group0_calibration_bias','group1_calibration_bias','group0_calibration_slope','group1_calibration_slope',
        'representation_marginal_probe_auroc','representation_conditional_probe_auroc',
        'score_marginal_probe_auroc','score_conditional_probe_auroc']
    cols=[k for k in main_cols if k in df]
    test=df[df.split=='test']
    test[['experiment','pathology','seed','id','method','gamma']+cols].to_csv(out/'main_results_per_seed.csv',index=False)
    summary=test.groupby(['experiment','pathology','id','method'],dropna=False)[cols].agg(['mean','std','count'])
    summary.columns=['__'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/'main_results_summary.csv',index=False)
    failures=[dict(path=str(f),**json.loads(f.read_text())) for f in BASE.glob('*/seed*/runs/*/failure.json')]
    done=len(list(BASE.glob('*/seed*/runs/*/pipeline_complete.json')))
    write(out/'status.json',dict(expected_new_runs=18,completed_new_runs=done,
                               failures=failures,reference_runs=60,metric_rows=len(df)))
    print((out/'status.json').read_text())
    if done!=18 or failures:raise SystemExit('Incomplete sweep; partial tables preserved')


if __name__=='__main__':main()
