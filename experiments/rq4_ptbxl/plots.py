"""Small, condition-separated visual summaries of saved RQ4 trial results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments.rq4_ptbxl.prepare import OUT,CLASSES,AGE_BINS

def make_plots():
    p=pd.read_csv(OUT/'prevalence_all_strata.csv')
    p=p.query("condition=='natural' and split=='all' and stratification=='sex_age'")
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True,layout='constrained')
    for ax,sex in zip(axes,['Female','Male']):
        z=p[p.sex==sex].pivot(index='age',columns='label',values='prevalence').reindex(index=AGE_BINS[:-1],columns=CLASSES)
        im=ax.imshow(z,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_xticks(range(5),CLASSES);ax.set_yticks(range(5),AGE_BINS[:-1]);ax.set_title(sex)
        for (i,j),v in np.ndenumerate(z.to_numpy()):
            ax.text(j,i,f'{v:.1%}',ha='center',va='center',color='white' if v>.6 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes,label='ECG label prevalence');fig.suptitle('PTB-XL natural prevalence by recorded sex and age')
    fig.savefig(OUT/'prevalence_sex_age.png',dpi=160);plt.close(fig)
    f=OUT/'trial_results.csv'
    if not f.exists():return
    r=pd.read_csv(f);targets=sorted(r.target.unique())
    fig,axes=plt.subplots(2,len(targets),figsize=(4*len(targets),7),squeeze=False,layout='constrained')
    colors={'erm':'#4c78a8','dann':'#f58518','cdann':'#54a24b'}
    for j,target in enumerate(targets):
        for i,test in enumerate(['natural','prevalence_shift']):
            ax=axes[i,j];z=r[(r.target==target)&(r.test_condition==test)]
            for d in z.to_dict('records'):
                ax.scatter(d['probe_mlp_auroc'],d['delta_eodds'],color=colors[d['method']],marker='o' if d['training_condition']=='natural' else '^',s=60)
                ax.annotate(d['method']+(' N' if d['training_condition']=='natural' else ' S'),(d['probe_mlp_auroc'],d['delta_eodds']),xytext=(3,3),textcoords='offset points',fontsize=7)
            ax.axvline(.5,color='gray',ls='--');ax.set_xlim(.45,1);ax.set_ylim(0,max(.3,r.delta_eodds.max()+.04));ax.set_title(f'{target}: {test} test');ax.set_xlabel('Independent marginal sex MLP AUROC');ax.set_ylabel('Equalized-odds gap')
    fig.suptitle('RQ4: disparities versus decodability\nN/circle = natural training; S/triangle = shifted training. One trial per point.')
    fig.savefig(OUT/'fairness_vs_decodability.png',dpi=160);plt.close(fig)
if __name__=='__main__':make_plots()
