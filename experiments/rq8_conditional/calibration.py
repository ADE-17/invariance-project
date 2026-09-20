"""Weak calibration following the user's GLM code, with patient-clustered Wald CIs.

Bias score = offset-only logistic intercept: positive means underprediction.
Mean bias = mean(p-y): positive means overprediction. No recalibration is applied.
"""
import warnings
import numpy as np
import statsmodels.api as sm
from scipy.special import logit


def weak_calibration(y, p, units=None):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if y.ndim!=1 or y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Expected equal finite one-dimensional arrays')
    if not np.isin(y,[0,1]).all() or ((p<0)|(p>1)).any(): raise ValueError('Invalid labels/probabilities')
    out=dict(n=len(y),n_pos=int(y.sum()),mean_bias=float(np.mean(p-y)),
             intercept=None,slope=None,intercept_fixed_slope=None,
             joint_converged=False,fixed_slope_converged=False,
             ci_method='patient_clustered_wald' if units is not None else 'wald')
    for key in ['intercept','slope','intercept_fixed_slope']:
        out[key+'_ci_low']=None;out[key+'_ci_high']=None
    if len(y)<3 or len(np.unique(y))<2:
        out['failure']='single_class_or_insufficient_rows';return out
    lp=logit(np.clip(p,1e-6,1-1e-6))
    kwargs={}
    if units is not None:
        units=np.asarray(units)
        if units.shape!=y.shape: raise ValueError('Cluster/label shape mismatch')
        _,groups=np.unique(units,return_inverse=True)
        if len(np.unique(groups))<2:
            out['failure']='insufficient_patient_clusters';return out
        kwargs=dict(cov_type='cluster',cov_kwds={'groups':groups})
    for joint in (False,True):
        mode='joint' if joint else 'fixed_slope'
        if joint and np.ptp(lp)<1e-10:
            out['joint_failure']='constant_predictions';continue
        x=np.column_stack([np.ones(len(y)),lp]) if joint else np.ones((len(y),1))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit=sm.GLM(y,x,family=sm.families.Binomial(),offset=None if joint else lp).fit(**kwargs)
            out[mode+'_warnings']=[str(w.message) for w in caught]
            separated=any('separation' in str(w.message).lower() for w in caught)
            if not fit.converged or not np.isfinite(fit.params).all() or separated:
                out[mode+'_failure']='nonconverged_or_separated';continue
            ci=np.asarray(fit.conf_int(alpha=.05))
            keys=['intercept','slope'] if joint else ['intercept_fixed_slope']
            for i,key in enumerate(keys):
                out[key]=float(fit.params[i])
                for j,side in enumerate(['low','high']):
                    out[key+'_ci_'+side]=float(ci[i,j]) if np.isfinite(ci[i,j]) else None
            out[mode+'_converged']=True
        except (ValueError,np.linalg.LinAlgError,RuntimeError) as exc:
            out[mode+'_failure']=repr(exc)
    return out


def diagnostics(y,p,a,units=None):
    y,p,a=map(np.asarray,(y,p,a));units=None if units is None else np.asarray(units)
    result={};details={}
    for name,mask in [('overall',np.ones(len(y),bool)),('group0',a==0),('group1',a==1)]:
        d=weak_calibration(y[mask],p[mask],None if units is None else units[mask]);details[name]=d
        result[name+'_calibration_bias']=d['mean_bias']
        result[name+'_calibration_bias_score']=d['intercept_fixed_slope']
        result[name+'_citl']=d['intercept_fixed_slope']
        result[name+'_calibration_slope']=d['slope']
        result[name+'_calibration_intercept']=d['intercept']
    biases=[result[f'group{g}_calibration_bias_score'] for g in (0,1)]
    result['worst_abs_calibration_bias_score']=None if None in biases else max(abs(b) for b in biases)
    result['worst_abs_mean_calibration_bias']=max(abs(result[f'group{g}_calibration_bias']) for g in (0,1))
    return result,details
