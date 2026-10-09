"""Run verified computations on a real CSV, emitting a machine-readable report."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import pandas as pd
from crashlab.data import validate
from crashlab.estimation import fit_all,evt
from crashlab.simulation import Config,simulate
from crashlab.risk import summarize
from crashlab.backtest import walk_forward,metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');parser.add_argument('--paths',type=int,default=10000);parser.add_argument('--replay',action='store_true');parser.add_argument('--output',default='experiment.json');a=parser.parse_args()
    prices=validate(pd.read_csv(a.csv));r=__import__('numpy').diff(__import__('numpy').log(prices.iloc[-1261:].values));models,errors=fit_all(r);c=Config(paths=a.paths);out={'last_observation':str(prices.index[-1]),'observations':len(prices),'calibration_returns':len(r),'configuration':c.__dict__,'disabled':errors,'results':{}}
    for i,(name,p) in enumerate(models.items()):
        began=time.perf_counter();paths,extra=simulate(name,p,float(prices.iloc[-1]),replace(c,seed=c.seed+i*1009));out['results'][name]={**summarize(paths,c),**extra,'seconds':time.perf_counter()-began}
    try:out['EVT_daily_log_loss']=evt(r,.95,.99)
    except ValueError as e:out['EVT_disabled']=str(e)
    if a.replay:
        bt=walk_forward(prices,window=252,horizon=90,stride=90,paths=1000,start='2019-10-01',end='2021-01-01',max_origins=12)
        out['replay_configuration']={'window':252,'horizon':90,'stride':90,'paths':1000,'period':['2019-10-01','2021-01-01']};out['replay_origins']=len(bt);out['replay_metrics']={n:metrics(bt.actual,bt[n]) for n in models if n in bt};bt.to_csv(Path(a.output).with_suffix('.replay.csv'),index=False)
    def clean(x):
        import numpy as np
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    Path(a.output).write_text(json.dumps(clean(out),indent=2,allow_nan=False));print(a.output)
