import numpy as np
from scipy.stats import norm

def wilson(successes,n,z=1.96):
    p=successes/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]

def events(paths,threshold,definition='starting loss'):
    if definition=='starting loss':return np.min(paths,axis=1)<=paths[:,0]*(1-threshold)
    if definition=='peak drawdown':return np.max(1-paths/np.maximum.accumulate(paths,axis=1),axis=1)>=threshold
    raise ValueError('Unknown event definition.')

def summarize(paths,c):
    hit=events(paths,c.threshold,c.definition);n=len(hit);p=float(hit.mean())
    # Pair-mean uncertainty when GBM antithetic paths are used; Wilson is IID only.
    if c.antithetic:
        k=n//2;pair=(hit[:k].astype(float)+hit[(n+1)//2:(n+1)//2+k])/2
        se=float(np.std(pair,ddof=1)/np.sqrt(k));ci=[max(0,p-1.96*se),min(1,p+1.96*se)]
    else:se=float(np.sqrt(p*(1-p)/n));ci=wilson(hit.sum(),n)
    loss=1-paths[:,-1]/paths[:,0];var=float(np.quantile(loss,.95));es=float(loss[loss>=var].mean())
    dd=np.max(1-paths/np.maximum.accumulate(paths,axis=1),axis=1)
    q=np.quantile(paths[:,-1],[.05,.5,.95])
    return dict(probability=p,mc_se=se,mc_ci95=ci,terminal_p05=float(q[0]),terminal_median=float(q[1]),terminal_p95=float(q[2]),terminal_mean=float(paths[:,-1].mean()),terminal_VaR95=var,terminal_ES95=es,max_drawdown_mean=float(dd.mean()),max_drawdown_p95=float(np.quantile(dd,.95)),threshold_probabilities={str(x):float(events(paths,x,c.definition).mean()) for x in [.1,.15,.2,.3,.4]},convergence=[dict(paths=k,p=float(hit[:k].mean())) for k in sorted(set([min(1000,n),min(5000,n),n]))])

def gbm_terminal_probability(p,threshold,horizon):
    t=horizon/252;sig=p['sigma'];m=p['mu']-.5*sig*sig
    return float(norm.cdf((np.log(1-threshold)-m*t)/(sig*np.sqrt(t))))

def gbm_continuous_hit_probability(p,threshold,horizon):
    a=np.log(1-threshold);t=horizon/252;m=p['mu']-.5*p['sigma']**2;s=p['sigma'];d=s*np.sqrt(t)
    return float(norm.cdf((a-m*t)/d)+np.exp(2*m*a/s**2)*norm.cdf((a+m*t)/d))
