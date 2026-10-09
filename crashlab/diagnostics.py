import numpy as np
from scipy.stats import norm,chi2,kstest,t
from .estimation import evt,gbm

def residual_diagnostics(z,nu=None,lags=10):
    z=np.asarray(z);z=z-z.mean();n=len(z)
    def ljung(x):
        ac=np.array([np.dot(x[k:],x[:-k])/np.dot(x,x) for k in range(1,lags+1)])
        q=n*(n+2)*np.sum(ac**2/(n-np.arange(1,lags+1)))
        return {'Q':float(q),'p_approx':float(chi2.sf(q,lags)),'acf':ac}
    dist=(lambda x:t.cdf(x*np.sqrt(nu/(nu-2)),nu)) if nu else norm.cdf
    ks=kstest(z,dist)
    return {'returns':ljung(z),'squared_returns':ljung(z*z-np.mean(z*z)),'KS_statistic':float(ks.statistic),'KS_p_naive':float(ks.pvalue),'caveat':'Approximate diagnostics: fitted parameters and dependence invalidate naive exact p-values.'}

def evt_uncertainty(r,q=.95,confidence=.99,replicates=100,seed=42,block=10):
    rng=np.random.default_rng(seed);r=np.asarray(r);values=[]
    for _ in range(replicates):
        starts=rng.integers(len(r),size=int(np.ceil(len(r)/block)))
        sample=r[(starts[:,None]+np.arange(block))%len(r)].ravel()[:len(r)]
        try:
            p=evt(sample,q,confidence);values.append([p['VaR'],p['ES'],p['shape']])
        except ValueError:pass
    if len(values)<replicates*.8:raise ValueError('Too few successful EVT bootstrap fits.')
    return {'VaR_ES_shape_ci95':np.quantile(values,[.025,.975],axis=0),'replicates':len(values),'method':'Circular block bootstrap; conditional on selected tail threshold.'}

def stability(r,window=252,stride=63):
    return [dict(end=i,**gbm(r[i-window:i])) for i in range(window,len(r)+1,stride)]
