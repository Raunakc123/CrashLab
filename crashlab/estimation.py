import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm,t,genpareto


def gbm(r):
    v=float(np.var(r,ddof=1))
    return dict(mu=float(np.mean(r)*252+v*252/2),sigma=float(np.sqrt(v*252)),origin='estimated',n=len(r))

def garch(r,student=True,asymmetric=False):
    x=np.asarray(r)*100
    def decode(z):
        weights=np.exp(np.r_[z[2:5],0]-max(0,max(z[2:5])))
        weights=weights/weights.sum()*.999
        return z[0],np.exp(z[1]),weights[0],weights[1],2*weights[2] if asymmetric else 0,2.05+np.exp(z[5])
    def calc(z):
        m,w,a,b,g,nu=decode(z); e=x-m; h=np.empty(len(x));h[0]=max(np.var(x),1e-8)
        for i in range(1,len(x)): h[i]=w+a*e[i-1]**2+b*h[i-1]+g*e[i-1]**2*(e[i-1]<0)
        return e,h
    def objective(z):
        e,h=calc(z);nu=decode(z)[-1]
        lp=t.logpdf(e/np.sqrt(h)*np.sqrt(nu/(nu-2)),nu)+.5*np.log(nu/(nu-2))- .5*np.log(h) if student else norm.logpdf(e/np.sqrt(h))-.5*np.log(h)
        return -np.sum(lp)
    z=[np.mean(x),np.log(max(np.var(x)*.04,1e-6)),1,4,-5,np.log(6)]
    fit=minimize(objective,z,method='L-BFGS-B',bounds=[(-5,5),(-15,5),(-12,12),(-12,12),(-12,12),(-4,5)],options={'maxiter':250})
    if not fit.success: raise ValueError('GARCH optimisation failed: '+fit.message)
    m,w,a,b,g,nu=decode(fit.x); e,h=calc(fit.x)
    return dict(mean=m/100,omega=w/10000,alpha=a,beta=b,gamma=g,nu=nu if student else None,h=float(h[-1]/10000),epsilon=float(e[-1]/100),loglik=-fit.fun,residuals=e/np.sqrt(h),conditional_vol=np.sqrt(h)/100,origin='estimated',stationarity=a+b+g/2)

def garch_forecast(p,horizon):
    h=p['omega']+p['alpha']*p['epsilon']**2+p['beta']*p['h']+p['gamma']*p['epsilon']**2*(p['epsilon']<0)
    out=[h]
    for _ in range(horizon-1): h=p['omega']+p['stationarity']*h;out.append(h)
    return np.asarray(out)

def jump(r):
    """Daily compound-Poisson Gaussian mixture MLE, truncated at count 8."""
    r=np.asarray(r);v=max(np.std(r),.001)
    def obj(z):
        m,s,l,j,d=z; k=np.arange(9)
        from scipy.stats import poisson
        lp=poisson.logpmf(k,l)[None,:]+norm.logpdf(r[:,None],m+k*j,np.sqrt(s*s+k*d*d))
        return -logsumexp(lp,axis=1).sum()
    fits=[minimize(obj,[np.mean(r),v*.8,.02,-2*v,2*v],bounds=[(-.03,.03),(.0001,.1),(.00001,.5),(-.25,.1),(.0001,.3)],method='L-BFGS-B'),minimize(obj,[np.mean(r),v*.7,.1,-v,v],bounds=[(-.03,.03),(.0001,.1),(.00001,.5),(-.25,.1),(.0001,.3)],method='L-BFGS-B')]
    good=[f for f in fits if f.success]
    if not good: raise ValueError('Jump mixture MLE failed.')
    fit=min(good,key=lambda f:f.fun);m,s,l,j,d=fit.x
    return dict(log_drift=m,sigma=s,lambda_day=l,jump_mean=j,jump_sd=d,mu=252*(m+s*s/2+l*np.expm1(j+d*d/2)),loglik=-fit.fun,origin='estimated; weakly identified from daily closes',boundary=bool(l<.00002 or l>.49))

def regime(r,seed=42):
    """Three-state Gaussian HMM, Baum-Welch EM with scaled forward/backward."""
    r=np.asarray(r);n=len(r);best=None
    for trial in range(3):
        rng=np.random.default_rng(seed+trial);means=np.quantile(r,[.25,.5,.75])+rng.normal(0,np.std(r)*.1,3);sd=np.std(r)*np.array([.6,1.1,2.]);A=np.full((3,3),.025);np.fill_diagonal(A,.95);pi=np.ones(3)/3;old=-np.inf
        for iteration in range(200):
            B=np.maximum(norm.pdf(r[:,None],means,sd),1e-250)
            f=np.empty((n,3));c=np.empty(n);f[0]=pi*B[0];c[0]=f[0].sum();f[0]/=c[0]
            for i in range(1,n): f[i]=(f[i-1]@A)*B[i];c[i]=f[i].sum();f[i]/=c[i]
            b=np.ones((n,3))
            for i in range(n-2,-1,-1): b[i]=A@(B[i+1]*b[i+1])/c[i+1]
            g=f*b;g/=g.sum(axis=1,keepdims=True)
            xi=np.zeros((3,3))
            for i in range(n-1):
                q=f[i,:,None]*A*(B[i+1]*b[i+1])[None,:];xi+=q/q.sum()
            A=(xi+1e-8);A/=A.sum(axis=1,keepdims=True)
            means=(g*r[:,None]).sum(axis=0)/g.sum(axis=0)
            sd=np.sqrt(np.maximum((g*(r[:,None]-means)**2).sum(axis=0)/g.sum(axis=0),1e-8));pi=g[0];ll=np.log(c).sum()
            if abs(ll-old)<1e-5: break
            old=ll
        if best is None or ll>best['loglik']: best=dict(means=means,sd=sd,A=A,current=f[-1],loglik=ll,iterations=iteration+1,origin='estimated')
    if best['iterations']>=200: raise ValueError('Regime EM did not converge; model disabled.')
    # Labels describe distributions, not known crisis dates.
    crisis=int(np.argmin(best['means']/best['sd'])); best['crisis_state']=crisis
    best['duration']=1/np.maximum(1-np.diag(best['A']),1e-12)
    return best

def evt(r,q=.95,confidence=.99):
    loss=-np.asarray(r);u=np.quantile(loss,q);ex=loss[loss>u]-u
    if len(ex)<40: raise ValueError('EVT needs at least 40 threshold exceedances.')
    xi,_,scale=genpareto.fit(ex,floc=0);pu=len(ex)/len(loss)
    if confidence<=1-pu: raise ValueError('VaR confidence must be above threshold quantile.')
    var=u+genpareto.ppf(1-(1-confidence)/pu,xi,scale=scale)
    es=(var+scale-xi*u)/(1-xi) if xi<1 else np.inf
    return dict(threshold=u,shape=xi,scale=scale,exceedances=len(ex),tail_fraction=pu,VaR=var,ES=es,confidence=confidence,normal_VaR=float(norm.ppf(confidence,np.mean(loss),np.std(loss,ddof=1))),normal_ES=float(np.mean(loss)+np.std(loss,ddof=1)*norm.pdf(norm.ppf(confidence))/(1-confidence)),origin='estimated; daily log-loss tail')

def tail_probability(p,loss):
    if loss<p['threshold']: raise ValueError('Requested loss lies below fitted tail threshold.')
    return p['tail_fraction']*genpareto.sf(loss-p['threshold'],p['shape'],scale=p['scale'])

def calculus(prices,window=21):
    from scipy.signal import savgol_filter
    window=min(window,len(prices)//2*2-1);window=max(5,window|1)
    return {k:savgol_filter(np.asarray(prices),window,3,deriv=d,delta=1.) for k,d in [('smooth',0),('velocity',1),('acceleration',2)]}

def fit_all(r,student=True,asymmetric=False):
    if len(r)<250 or not np.isfinite(r).all(): raise ValueError('Need at least 250 finite returns.')
    out={'GBM':gbm(r),'Bootstrap':dict(returns=np.asarray(r),origin='empirical')};errors={}
    for name,fn in [('GARCH',lambda:garch(r,student,asymmetric)),('Jump',lambda:jump(r)),('Regime',lambda:regime(r))]:
        try: out[name]=fn()
        except (ValueError,FloatingPointError) as e: errors[name]=str(e)
    return out,errors
