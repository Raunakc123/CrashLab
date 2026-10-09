import numpy as np
import pandas as pd
from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_dashboard_initial_load():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'advanced_app.py')).run(timeout=20)
    assert not app.exception
    assert 'Load historical' in app.info[0].value

def test_dashboard_simulation_and_stale_controls():
    rng=np.random.default_rng(77)
    prices=pd.Series(100*np.exp(np.cumsum(rng.normal(.0002,.012,1000))),index=pd.bdate_range('2020',periods=1000))
    prices.attrs['source']='SYNTHETIC TEST FIXTURE — NOT MARKET HISTORY'
    app=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'advanced_app.py'));app.session_state['prices']=prices
    app.run(timeout=60);assert not app.exception
    for b in app.button:
        if b.label=='Run Simulation / Compare Models':b.click();break
    app.run(timeout=60);assert not app.exception
    assert len(app.session_state['results'])>=3
    next(w for w in app.slider if w.label=='Forecast trading days').set_value(95);app.run(timeout=60)
    assert not app.exception
    assert any('Controls changed' in w.value for w in app.warning)

def test_diagnostic_model_panels():
    rng=np.random.default_rng(22);prices=pd.Series(100*np.exp(np.cumsum(rng.normal(.0001,.011,1300))),index=pd.bdate_range('2020',periods=1300))
    prices.attrs['source']='SYNTHETIC TEST FIXTURE — NOT MARKET HISTORY'
    app=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'advanced_app.py'));app.session_state['prices']=prices;app.run(timeout=60)
    for model in ['GARCH','Regime','EVT','Calculus','Heston']:
        next(w for w in app.selectbox if w.label=='Mathematical model').select(model)
        app.run(timeout=60);assert not app.exception,model

def test_dashboard_replay():
    rng=np.random.default_rng(91);prices=pd.Series(100*np.exp(np.cumsum(rng.normal(0,.015,700))),index=pd.bdate_range('2020',periods=700))
    prices.attrs['source']='SYNTHETIC TEST FIXTURE — NOT MARKET HISTORY'
    app=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'advanced_app.py'));app.session_state['prices']=prices;app.run(timeout=60)
    next(w for w in app.number_input if w.label=='Estimation returns').set_value(250)
    next(w for w in app.slider if w.label=='Maximum chronological origins').set_value(5)
    app.run(timeout=60)
    next(w for w in app.button if w.label=='Historical Crash Replay').click();app.run(timeout=60)
    assert not app.exception
    assert len(app.session_state['backtest'])==4
