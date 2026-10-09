import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from crashlab.simulation import Config,simulate
from crashlab.estimation import gbm,garch,garch_forecast,regime,evt,tail_probability,calculus,jump
from crashlab.risk import events,summarize,gbm_terminal_probability,gbm_continuous_hit_probability,wilson
from crashlab.backtest import walk_forward,stack,historical_frequency
from crashlab.data import validate

def returns(seed=8,n=1800):return np.random.default_rng(seed).normal(.0002,.012,n)

def test_gbm_analytical_mean_variance_and_tail():
    p=dict(mu=.07,sigma=.24);c=Config(paths=100000,horizon=90,seed=3);a,_=simulate('GBM',p,100.,c);t=90/252
    expected=100*np.exp(p['mu']*t);variance=10000*np.exp(2*p['mu']*t)*np.expm1(p['sigma']**2*t)
    assert abs(a[:,-1].mean()-expected)<4*np.sqrt(variance/c.paths)
    assert np.var(a[:,-1])/variance==pytest.approx(1,rel=.025)
    analytical=gbm_terminal_probability(p,.2,90);observed=np.mean(a[:,-1]<=80)
    assert abs(analytical-observed)<4*np.sqrt(analytical*(1-analytical)/c.paths)
    assert events(a,.2).mean()<=gbm_continuous_hit_probability(p,.2,90)+.005

def test_definitions():
    a=np.array([[100,150,115],[100,90,85]])
    assert list(events(a,.2))==[False,False]
    assert list(events(a,.2,'peak drawdown'))==[True,False]

def test_reproducibility_and_convergence():
    p=gbm(returns());c=Config(paths=10000,horizon=30)
    a,_=simulate('GBM',p,100,c);b,_=simulate('GBM',p,100,c);assert np.array_equal(a,b)
    small=summarize(a[:1000],replace(c,paths=1000));large=summarize(a,c)
    assert large['mc_ci95'][1]-large['mc_ci95'][0]<small['mc_ci95'][1]-small['mc_ci95'][0]
    assert wilson(0,10000)[1]>0

def test_antithetic_mean_and_error():
    c=Config(paths=10000,horizon=10,antithetic=True);p=dict(mu=.05,sigma=.2);a,_=simulate('GBM',p,100,c)
    assert a[:,-1].mean()==pytest.approx(100*np.exp(.05*10/252),rel=.001)
    assert summarize(a,c)['mc_se']>=0

def test_garch_constraint_and_forecast():
    rng=np.random.default_rng(19);r=[];h=.0001;e=0
    for _ in range(2200):
        h=.000005+.09*e*e+.86*h;e=np.sqrt(h)*rng.standard_t(8)*np.sqrt(6/8);r.append(e)
    p=garch(np.asarray(r));assert p['omega']>0 and p['stationarity']<1 and p['nu']>2
    assert p['alpha']==pytest.approx(.09,abs=.055);assert p['beta']==pytest.approx(.86,abs=.09)
    f=garch_forecast(p,10);assert np.all(f>0)
    assert f[1]==pytest.approx(p['omega']+p['stationarity']*f[0])
    c=Config(paths=100000,horizon=5);a,_=simulate('GARCH',p,100,c)
    assert np.var(np.log(a[:,1]/100))==pytest.approx(f[0],rel=.04)

def test_poisson_frequency():
    p=dict(mu=.05,sigma=.008,lambda_day=.025,jump_mean=-.04,jump_sd=.02)
    c=Config(paths=10000,horizon=90);_,extra=simulate('Jump',p,100,c);expected=c.paths*c.horizon*p['lambda_day']
    assert abs(extra['jump_count']-expected)<5*np.sqrt(expected)

def test_jump_fit_finite():
    rng=np.random.default_rng(7);k=rng.poisson(.04,1800);r=rng.normal(0,.008,1800)+k*(-.04)+np.sqrt(k)*rng.normal(0,.02,1800)
    p=jump(r);assert np.isfinite(p['loglik']) and p['lambda_day']>0

def test_regime_fit_and_transition():
    rng=np.random.default_rng(9);r=np.r_[rng.normal(.001,.005,500),rng.normal(-.002,.03,500),rng.normal(0,.015,500)]
    p=regime(r);assert np.allclose(p['A'].sum(axis=1),1);assert np.all(p['A']>=0);assert p['current'].sum()==pytest.approx(1)
    p=dict(A=np.eye(3),current=np.array([0,1,0]),means=np.array([0,0,0]),sd=np.array([.01,.01,.01]),crisis_state=1)
    _,ex=simulate('Regime',p,100,Config(paths=1000,horizon=10));assert ex['crisis_entry_probability']==1

def test_evt_tail_formula_and_quantile():
    p=dict(threshold=.01,shape=.2,scale=.02,tail_fraction=.05)
    assert tail_probability(p,.01)==pytest.approx(.05)
    assert tail_probability(p,.03)==pytest.approx(.05*1.2**(-5))
    rng=np.random.default_rng(3);r=-rng.exponential(.01,10000);p=evt(r,.9,.99)
    assert p['VaR']==pytest.approx(.01*np.log(100),rel=.08)
    assert p['ES']>p['VaR']

def test_heston_nonnegative_prices_and_bootstrap():
    p=dict(mu=.05,kappa=2,theta=.04,xi=.8,rho=-.7,v0=.04)
    a,_=simulate('Heston',p,100,Config(paths=1000,horizon=90));assert np.all(np.isfinite(a)) and np.all(a>0)
    r=returns();a,_=simulate('Bootstrap',dict(returns=r),100,Config(paths=1000,horizon=10))
    assert np.all(a>0)

def test_derivatives_polynomial():
    t=np.arange(100);d=calculus(2*t*t+3*t+20,21)
    assert np.allclose(d['velocity'][15:-15],4*t[15:-15]+3)
    assert np.allclose(d['acceleration'][15:-15],4)

def test_data_validation():
    dates=pd.date_range('2020-01-01',periods=300);d=pd.DataFrame({'Date':dates,'Close':100})
    assert len(validate(d))==300
    d.loc[0,'Close']=-1
    with pytest.raises(ValueError):validate(d)

def test_walk_forward_no_future_leakage(monkeypatch):
    import crashlab.backtest as b
    # Mock only expensive fitting; verify exact training slice and actual event labels.
    captured=[]
    def fitting(r,*args):captured.append(r.copy());return {'GBM':gbm(r)},{}
    monkeypatch.setattr(b,'fit_all',fitting)
    prices=pd.Series(100*np.exp(np.r_[0,np.cumsum(returns(n=700))]),index=pd.bdate_range('2000',periods=701))
    d1=walk_forward(prices,window=252,horizon=30,stride=30,paths=100,models=['GBM'],max_origins=2)
    first=captured[0].copy();changed=prices.copy();changed.iloc[253:]*=.1;captured.clear()
    d2=walk_forward(changed,window=252,horizon=30,stride=30,paths=100,models=['GBM'],max_origins=1)
    assert np.array_equal(first,captured[0]);assert d1.GBM.iloc[0]==d2.GBM.iloc[0];assert d2.actual.iloc[0]==1
    assert all(d1.train_end==d1.date);assert all(d1.label_end>d1.date)

def test_purged_stacking_and_disable():
    dates=pd.bdate_range('2000',periods=30);y=np.array([0,1]*15)
    d=pd.DataFrame({'date':dates,'label_end':dates+pd.Timedelta(days=1),'actual':y,'GBM':.1+.8*y,'GARCH':np.full(30,.5),'baseline':np.full(30,.5)})
    result=stack(d,['GBM','GARCH']);assert sum(result['weights'].values())==pytest.approx(1)
    assert result['weights']['GBM']>.99;assert result['test_metrics']['Brier']<.02
    d['actual']=0
    with pytest.raises(ValueError):stack(d,['GBM','GARCH'])

def test_gjr_gaussian_and_stress_response():
    r=returns(n=1000);p=garch(r,student=False,asymmetric=True)
    assert p['gamma']>=0 and p['stationarity']<1 and p['nu'] is None
    c=Config(paths=10000,horizon=90);gp=dict(mu=.06,sigma=.2)
    a,_=simulate('GBM',gp,100,c);b,_=simulate('GBM',gp,100,replace(c,vol_stress=2))
    assert events(b,.2).mean()>events(a,.2).mean()

def test_future_horizon_metrics_use_correct_labels(monkeypatch):
    import crashlab.backtest as b
    def fitting(r,*args):return {'GBM':gbm(r),'Bootstrap':dict(returns=r)},{}
    monkeypatch.setattr(b,'fit_all',fitting)
    vals=np.r_[np.full(253,100.),np.linspace(99,70,30),np.full(50,70.)]
    prices=pd.Series(vals,index=pd.bdate_range('2000',periods=len(vals)))
    df=walk_forward(prices,252,30,.2,stride=30,paths=100,max_origins=1,models=['GBM'])
    assert df.actual.iloc[0]==1
    assert df.baseline.iloc[0]<.1
    assert 'unconditional' in df
