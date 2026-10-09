from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    paths:int=10000
    horizon:int=90
    seed:int=42
    threshold:float=.20
    definition:str='starting loss'
    block:int=10
    vol_stress:float=1.
    jump_stress:float=1.
    crisis_stress:float=1.
    antithetic:bool=False
    def __post_init__(self):
        if not 100<=self.paths<=100000 or not 1<=self.horizon<=756: raise ValueError('Paths 100–100,000; horizon 1–756.')
        if not 0<self.threshold<1: raise ValueError('Threshold must be between 0 and 1.')
        if self.definition not in ['starting loss','peak drawdown']: raise ValueError('Unknown event definition.')
        if self.block<1 or min(self.vol_stress,self.jump_stress,self.crisis_stress)<=0: raise ValueError('Invalid block or stress parameters.')

def simulate(name,p,s0,c):
    """Vectorised across paths; daily monitoring, float32 path storage."""
    rng=np.random.default_rng(c.seed);n=c.paths;h=c.horizon
    paths=np.empty((n,h+1),dtype=np.float32);paths[:,0]=s0
    if c.antithetic and name!='GBM': raise ValueError('Antithetic variance reduction is supported only for GBM.')
    logprice=np.full(n,np.log(s0));states=None;state_hit=None;counts=0
    if name=='GARCH':
        variance=np.full(n,p['h']);eps=np.full(n,p['epsilon'])
    if name=='Regime':
        states=rng.choice(3,n,p=p['current']);A=p['A'].copy();A[:,p['crisis_state']]*=c.crisis_stress;A/=A.sum(axis=1,keepdims=True);state_hit=states==p['crisis_state']
    if name=='Heston':
        variance=np.full(n,p['v0'])
        if not (-1<=p['rho']<=1 and min(p['kappa'],p['theta'],p['xi'],p['v0'])>0): raise ValueError('Invalid Heston parameters.')
    if name=='Bootstrap': indices=rng.integers(len(p['returns']),size=n)
    for i in range(h):
        z=rng.standard_normal((n+1)//2) if c.antithetic else rng.standard_normal(n)
        if c.antithetic:z=np.r_[z,-z][:n]
        if name=='GBM':
            sig=p['sigma']*c.vol_stress;ret=(p['mu']-.5*sig*sig)/252+sig/np.sqrt(252)*z
        elif name=='GARCH':
            variance=p['omega']+p['alpha']*eps**2+p['beta']*variance+p['gamma']*eps**2*(eps<0)
            if p['nu'] is not None:z=rng.standard_t(p['nu'],n)*np.sqrt((p['nu']-2)/p['nu'])
            eps=np.sqrt(variance)*z
            ret=p['mean']+eps*c.vol_stress
        elif name=='Jump':
            k=rng.poisson(p['lambda_day']*c.jump_stress,n);counts+=int(k.sum())
            sig=p['sigma']*c.vol_stress;j=p['jump_mean']*c.jump_stress;sd=p['jump_sd']
            # Keep estimated arithmetic drift fixed under compensated jump stress.
            drift=p['mu']/252-.5*sig*sig-p['lambda_day']*c.jump_stress*np.expm1(j+sd*sd/2)
            ret=drift+sig*z+k*j+np.sqrt(k)*sd*rng.standard_normal(n)
        elif name=='Regime':
            u=rng.random(n);states=(u[:,None]>np.cumsum(A[states],axis=1)).sum(axis=1);states=np.minimum(states,2)
            state_hit|=states==p['crisis_state'];ret=p['means'][states]+p['sd'][states]*c.vol_stress*z
        elif name=='Heston':
            vplus=np.maximum(variance,0);dt=1/252
            z2=p['rho']*z+np.sqrt(1-p['rho']**2)*rng.standard_normal(n)
            ret=(p['mu']-.5*vplus*c.vol_stress**2)*dt+np.sqrt(vplus*dt)*c.vol_stress*z
            variance=variance+p['kappa']*(p['theta']-vplus)*dt+p['xi']*np.sqrt(vplus*dt)*z2
        elif name=='Bootstrap':
            if i%c.block==0:indices=rng.integers(len(p['returns']),size=n)
            ret=p['returns'][indices]*c.vol_stress;indices=(indices+1)%len(p['returns'])
        else:raise ValueError('Unknown simulator: '+name)
        logprice+=ret
        if not np.isfinite(logprice).all() or np.max(logprice)>80: raise ValueError('Unstable simulation: overflow risk.')
        paths[:,i+1]=np.exp(logprice)
    extra={'jump_count':counts} if name=='Jump' else {}
    if state_hit is not None:extra['crisis_entry_probability']=float(state_hit.mean())
    return paths,extra
