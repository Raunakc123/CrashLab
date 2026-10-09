import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .estimation import fit_all,garch_forecast
from .simulation import Config,simulate
from .risk import events,summarize

CRISES={'Black Monday 1987':('1987-01-01','1988-03-01'),'Dot-com 2000–2002':('2000-01-01','2003-01-01'),'Global Financial Crisis 2008':('2007-01-01','2009-07-01'),'COVID-19 2020':('2019-10-01','2021-01-01')}

def historical_frequency(prices,horizon,threshold,definition):
    """Only complete past windows in the training sample; no forward labels."""
    vals=np.asarray(prices);hits=[]
    for j in range(0,len(vals)-horizon,horizon):hits.append(events(vals[None,j:j+horizon+1],threshold,definition)[0])
    return (sum(hits)+.5)/(len(hits)+1)

def walk_forward(prices,window=1260,horizon=90,threshold=.2,definition='starting loss',stride=90,paths=1000,seed=42,start=None,end=None,max_origins=30,models=('GBM','GARCH','Jump','Regime','Bootstrap'),progress=None,student=True,asymmetric=False):
    dates=prices.index;rows=[]
    for i in range(window,len(prices)-horizon,stride):
        if start is not None and dates[i]<pd.Timestamp(start):continue
        if end is not None and dates[i]>pd.Timestamp(end):continue
        # Price at origin is known. Neither calibration nor baseline receives future observations.
        train=prices.iloc[i-window:i+1];r=np.diff(np.log(train.values));params,errors=fit_all(r,student,asymmetric)
        actual=events(prices.iloc[i:i+horizon+1].values[None,:],threshold,definition)[0]
        row={'date':dates[i],'label_end':dates[i+horizon],'actual':int(actual),'baseline':historical_frequency(train,horizon,threshold,definition),'train_end':train.index[-1],'errors':str(errors)}
        cfg=Config(paths=paths,horizon=horizon,threshold=threshold,definition=definition,seed=seed+len(rows))
        row['unconditional']=summarize(simulate('Bootstrap',params['Bootstrap'],float(train.iloc[-1]),Config(paths=paths,horizon=horizon,threshold=threshold,definition=definition,seed=seed+len(rows)+999,block=1))[0],cfg)['probability'] if 'Bootstrap' in params else np.nan
        for model in models:
            if model in params:
                try: row[model]=summarize(simulate(model,params[model],float(train.iloc[-1]),cfg)[0],cfg)['probability']
                except ValueError as e:row['errors']+=str(e)
        if 'GARCH' in params:
            row['GARCH_forecast_vol_annual']=float(np.sqrt(np.mean(garch_forecast(params['GARCH'],horizon))*252))
            future_returns=np.diff(np.log(prices.iloc[i:i+horizon+1].values))
            row['realised_forward_vol_annual']=float(np.sqrt(np.mean((future_returns-params['GARCH']['mean'])**2)*252))
        rows.append(row)
        if progress:progress(len(rows),max_origins)
        if len(rows)>=max_origins:break
    if not rows:raise ValueError('Insufficient observations for selected replay window and horizon.')
    return pd.DataFrame(rows)

def metrics(y,p):
    y=np.asarray(y);p=np.clip(np.asarray(p),1e-6,1-1e-6);pred=p>=.5
    tp=sum(pred&(y==1));fp=sum(pred&(y==0));fn=sum(~pred&(y==1));tn=sum(~pred&(y==0))
    return {'Brier':float(np.mean((p-y)**2)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),'precision':float(tp/(tp+fp)) if tp+fp else np.nan,'recall':float(tp/(tp+fn)) if tp+fn else np.nan,'false_positive_rate':float(fp/(fp+tn)) if fp+tn else np.nan,'missed_crash_rate':float(fn/(fn+tp)) if fn+tp else np.nan,'stability_mean_abs_change':float(np.mean(np.abs(np.diff(p)))) if len(p)>1 else np.nan}

def stack(df,model_names):
    """Purged chronological validation/test split, convex log-loss stacking.
    Earlier forecast labels must resolve before first test origin. Joint fitting
    handles correlated predictors; held-out score is never training performance.
    """
    names=[n for n in model_names if n in df and df[n].notna().all()]
    if len(names)<2 or len(df)<12:raise ValueError('Ensemble disabled: need 12+ complete forecasts and 2+ models.')
    split=int(len(df)*.65);test=df.iloc[split:];train=df.iloc[:split];train=train[train.label_end<test.date.iloc[0]]
    if len(train)<6 or train.actual.nunique()<2:raise ValueError('Ensemble disabled: purged validation sample needs 6+ origins and both outcomes.')
    P=train[names].to_numpy();y=train.actual.to_numpy()
    def objective(w):
        p=np.clip(P@w,1e-6,1-1e-6)
        return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
    fit=minimize(objective,np.ones(len(names))/len(names),method='SLSQP',bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}])
    if not fit.success:raise ValueError('Stacking optimisation failed.')
    prediction=test[names].to_numpy()@fit.x
    return {'weights':dict(zip(names,fit.x)),'test_metrics':metrics(test.actual,prediction),'test_predictions':prediction,'test_dates':test.date,'individual_test_metrics':{n:metrics(test.actual,test[n]) for n in names+[n for n in ['baseline','unconditional'] if n in test]},'training_origins':len(train),'test_origins':len(test),'procedure':'Convex log-loss weights; purged chronological split; untouched later test set.'}

def calibration(y,p):
    d=pd.DataFrame({'y':y,'p':p});d['bin']=pd.cut(d.p,np.linspace(0,1,6),include_lowest=True)
    return d.groupby('bin',observed=True).agg(predicted=('p','mean'),observed=('y','mean'),count=('y','size')).reset_index().astype({'bin':str})
