import io,json,hashlib
from dataclasses import asdict,replace
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from crashlab.data import fetch_market,validate,SYMBOLS
from crashlab.estimation import fit_all,evt,tail_probability,calculus,garch_forecast
from crashlab.simulation import Config,simulate
from crashlab.risk import summarize,gbm_continuous_hit_probability
from crashlab.backtest import walk_forward,metrics,stack,calibration,CRISES
from crashlab.diagnostics import residual_diagnostics,evt_uncertainty,stability

st.set_page_config(page_title='CrashLab | Quantitative Risk',layout='wide')
st.markdown('''<style>.stApp{background:#080f1c} div[data-testid="stMetric"]{background:#101d30;padding:16px;border-radius:8px;border:1px solid #233752} h1{letter-spacing:-1.5px}</style>''',unsafe_allow_html=True)
st.title('CrashLab · Mathematical Risk Engine')
st.caption('Daily-observation forecasts · reproducible Monte Carlo · explicit model uncertainty')

def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    if isinstance(obj,pd.Timestamp):return obj.isoformat()
    return obj

@st.cache_data(ttl=3600)
def download(symbol,start,end,provider):return fetch_market(symbol,start,end,provider)

with st.sidebar:
    st.header('Experiment configuration')
    upload=st.file_uploader('Historical prices CSV (Date, Close)',type=['csv'])
    market=st.selectbox('Market',list(SYMBOLS)+['Custom ticker'])
    symbol=st.text_input('Yahoo ticker','AAPL') if market=='Custom ticker' else SYMBOLS[market]
    provider=st.selectbox('Data provider',['Yahoo with FRED fallback','FRED index data'])
    start=st.date_input('Data start',date(1980,1,1));end=st.date_input('Data end (exclusive)',date.today())
    load=st.button('Load real market data',type='primary')
    window=st.number_input('Estimation returns',250,20000,1260,step=252)
    horizon=st.slider('Forecast trading days',5,252,90)
    paths=st.select_slider('Simulation paths',options=[1000,10000,25000,50000,100000],value=10000)
    threshold=st.selectbox('Loss threshold',[.1,.15,.2,.3,.4],index=2,format_func=lambda x:f'{x:.0%}')
    definition=st.radio('Event definition',['starting loss','peak drawdown'])
    seed=st.number_input('Random seed',0,2147483647,42)
    student=st.checkbox('Student t GARCH innovations',True);asymmetric=st.checkbox('GJR asymmetry',False)
    block=st.slider('Bootstrap block length',1,60,10)
    model_selection=st.multiselect('Simulators',['GBM','GARCH','Jump','Regime','Bootstrap','Heston'],default=['GBM','GARCH','Jump','Regime','Bootstrap'])
    with st.expander('Conditional stress scenario'):
        vs=st.slider('Volatility multiplier',.5,3.,1.,.1);js=st.slider('Jump intensity / negative mean multiplier',.5,4.,1.,.1);cs=st.slider('Crisis transition multiplier',.5,4.,1.,.1)
    with st.expander('Heston — illustrative, not calibrated'):
        hk=st.number_input('Mean reversion κ',.01,20.,2.)
        ht=st.number_input('Long-run variance θ',.0001,1.,.04,format='%.4f')
        hx=st.number_input('Vol of variance ξ',.01,3.,.3)
        hr=st.slider('Correlation ρ',-.99,.99,-.7)
    with st.expander('Optional jump parameter override'):
        override=st.checkbox('Use user-defined jump parameters',False)
        jl=st.number_input('Jumps per trading year',.001,126.,5.)
        jm=st.number_input('Log jump mean',-.5,.2,-.03)
        jd=st.number_input('Log jump standard deviation',.0001,.5,.04)
    anti=st.checkbox('GBM antithetic sampling',False)
    if st.button('Reset Parameters'):
        st.session_state.clear();st.rerun()

if upload is not None:
    try:
        prices=validate(pd.read_csv(upload));prices.attrs['source']='User-provided CSV: provenance not independently verified'
        st.session_state['prices']=prices
    except Exception as e:st.error(str(e))
if load:
    try:
        with st.spinner('Fetching historical closes…'):st.session_state['prices']=download(symbol,start,end,provider)
    except Exception as e:st.error(f'Data download failed: {e}. Upload an authentic CSV to continue.')
if 'prices' not in st.session_state:
    st.info('Load historical market prices or upload a real CSV to calibrate the engine. No synthetic data is shown as history.')
    st.stop()
prices=st.session_state.prices
r=np.diff(np.log(prices.iloc[-(window+1):].values));s0=float(prices.iloc[-1])
st.caption(f"Source: {prices.attrs.get('source','CSV')} | {len(prices):,} closes | Latest observation: {prices.index[-1].date()} | Calibration: {len(r):,} returns")
if prices.index[-1] < pd.Timestamp(date.today())-pd.Timedelta(days=7):st.warning('Historical snapshot: data is more than seven calendar days old; these are not current market estimates.')
if len(r)<250:st.error('Insufficient estimation observations.');st.stop()
base=Config(paths=paths,horizon=horizon,seed=seed,threshold=threshold,definition=definition,block=block)
fitkey=hashlib.sha256(r.tobytes()+str((student,asymmetric,s0,str(prices.index[-1]))).encode()).hexdigest()
a,b,c=st.columns(3)
recal=a.button('Recalibrate Models');run=b.button('Run Simulation / Compare Models',type='primary');stress=c.button('Stress-Test Market')
if recal or run or stress or st.session_state.get('fitkey')!=fitkey:
    with st.spinner('Maximum likelihood / HMM calibration…'):
        fitted,errors=fit_all(r,student,asymmetric)
    st.session_state.update(fitted=fitted,fitkey=fitkey,fit_errors=errors)
for name,error in st.session_state.fit_errors.items():st.warning(name+' disabled: '+error)
fitted=st.session_state.fitted.copy()
fitted['Heston']=dict(mu=fitted['GBM']['mu'],kappa=hk,theta=ht,xi=hx,rho=hr,v0=fitted['GBM']['sigma']**2,origin='illustrative user parameters; variance initialised from realised returns')
if override and 'Jump' in fitted:
    fitted['Jump']={**fitted['Jump'],'lambda_day':jl/252,'jump_mean':jm,'jump_sd':jd,'origin':'user-defined jump override'}
experiment_key=str((fitkey,asdict(base),model_selection,hk,ht,hx,hr,override,jl,jm,jd,anti,vs,js,cs))
if run or stress:
    cfg=replace(base,vol_stress=vs,jump_stress=js,crisis_stress=cs) if stress else base
    results={};visual={};bar=st.progress(0.)
    for idx,name in enumerate(model_selection):
        if name not in fitted:continue
        try:
            mc=replace(cfg,antithetic=anti and name=='GBM')
            mc=replace(mc,seed=seed+idx*1009)
            arr,extra=simulate(name,fitted[name],s0,mc);results[name]={**summarize(arr,mc),**extra}
            # Retain only plotting paths and summaries; avoid holding all model arrays.
            bands=np.quantile(arr,[.05,.5,.95],axis=0);rng=np.random.default_rng(seed)
            sample=arr[rng.choice(len(arr),min(2000,len(arr)),replace=False)].copy()
            worst=arr[np.argsort(arr[:,-1])[:5]].copy()
            dd=np.max(1-arr/np.maximum.accumulate(arr,axis=1),axis=1)
            visual[name]=dict(drawdowns=dd,bands=bands,sample=sample,worst=worst,terminal=arr[:,-1].copy())
            del arr
        except ValueError as e:st.error(name+': '+str(e))
        bar.progress((idx+1)/max(1,len(model_selection)))
    st.session_state.update(results=results,visual=visual,result_key=experiment_key,run_config=asdict(cfg))
if st.session_state.get('result_key')!=experiment_key and 'results' in st.session_state:st.warning('Controls changed. Run Simulation to update forecasts. Previous results are hidden.');results={}
else:results=st.session_state.get('results',{})
tabs=st.tabs(['Market overview','Models & diagnostics','Crash forecast','Monte Carlo','Historical replay'])
with tabs[0]:
    st.plotly_chart(px.line(prices,title='Observed historical prices'),width='stretch')
    x,y=st.columns(2)
    x.plotly_chart(px.histogram(x=r,nbins=80,title='Historical daily log returns'),width='stretch')
    y.plotly_chart(px.line(1-prices/prices.cummax(),title='Observed peak-to-trough drawdown'),width='stretch')
    st.plotly_chart(px.line(pd.Series(r).rolling(21).std()*np.sqrt(252),title='21-day annualised realised volatility'),width='stretch')
with tabs[1]:
    selected=st.selectbox('Mathematical model',['GBM','GARCH','Jump','Regime','EVT','Calculus','Heston'])
    eq={'GBM':r'dS=\mu Sdt+\sigma SdW','GARCH':r'h_t=\omega+\alpha\epsilon_{t-1}^2+\beta h_{t-1}+\gamma\epsilon_{t-1}^2 I(\epsilon_{t-1}<0)','Jump':r'dS/S=(\mu-\lambda\kappa)dt+\sigma dW+(J-1)dN','Regime':r'P(Z_{t+1}=j\mid Z_t=i)=A_{ij}','EVT':r'P(L>x)=p_u(1+\xi(x-u)/\beta)^{-1/\xi}','Calculus':r'P^{\prime}(t),\quad P^{\prime\prime}(t)','Heston':r'dv=\kappa(\theta-v)dt+\xi\sqrt{v}dW_2'}
    st.latex(eq[selected])
    if selected in fitted:
        p=fitted[selected];st.json(clean({k:v for k,v in p.items() if k not in ['returns','residuals','conditional_vol']}))
    if selected=='GBM':st.info('Gaussian independent log returns and constant volatility can understate tail risk. Drift estimates are noisy. Daily monitoring misses between-close threshold crossings.')
    if selected=='GARCH' and selected in fitted:
        p=fitted[selected];st.plotly_chart(px.line(garch_forecast(p,horizon)**.5*np.sqrt(252),title='Forecast conditional annualised volatility'),width='stretch')
        vol=pd.DataFrame({'conditional':p['conditional_vol'],'realised_backward_21d':pd.Series(r).rolling(21).std()})
        st.plotly_chart(px.line(vol,title='Fitted volatility vs backward realised volatility (not an out-of-sample score)'),width='stretch')
        st.json(clean(residual_diagnostics(p['residuals'],p['nu'])))
        percentile=float(np.mean(p['conditional_vol']<=garch_forecast(p,1)[0]**.5))
        st.metric('Next-day volatility vs fitted historical values',f'{percentile:.0%} percentile')
    if selected=='Regime' and selected in fitted:st.info('The crisis-like label is assigned by lowest mean/volatility ratio. It is an inferred distribution, not a confirmed crisis. Entry probability includes being in that state at origin.')
    if selected=='Jump':st.info('Daily-close jump mixture estimates are weakly identified; Gaussian jumps and diffusion volatility can substitute for one another. Parameter uncertainty is not included in Monte Carlo intervals.')
    if selected=='Heston':st.warning('Illustrative physical-measure simulation only. No option calibration or claim that closing prices identify κ, θ, ξ and ρ. Full-truncation Euler; Feller condition not required for this discretisation.')
    if selected=='Calculus':
        sw=st.slider('Savitzky–Golay smoothing observations',5,101,21,step=2);d=calculus(prices.values,sw)
        st.plotly_chart(px.line(pd.DataFrame({'price':prices.values,'smooth':d['smooth']},index=prices.index)),width='stretch')
        st.plotly_chart(px.line(pd.DataFrame({'dP/dt':d['velocity'],'d²P/dt²':d['acceleration']},index=prices.index)),width='stretch');st.caption('Price units per observation / squared observation. Centred smoothing uses neighbours: descriptive only, never used in replay forecasts.')
    if selected=='EVT':
        tq=st.slider('Tail threshold quantile',.85,.98,.95,.01);confidence=st.selectbox('Tail confidence',[.99,.995,.999]);
        try:
            p=evt(r,tq,confidence);st.json(clean(p));st.caption('One-day log-loss quantiles; not a 90-day crash probability. ES is infinite if fitted shape ≥ 1.')
            sensitivity=[]
            for q in [.85,.9,.95,.975]:
                try:sensitivity.append(dict(q=q,**evt(r,q,confidence)))
                except ValueError:pass
            st.dataframe(pd.DataFrame(sensitivity),width='stretch')
            loss=st.slider('Daily log-loss tail query',.01,.3,.05,.01)
            if loss>=p['threshold']:st.metric('Daily tail exceedance',f'{tail_probability(p,loss):.3%}')
            if st.button('Bootstrap EVT uncertainty (100 fits)'):
                with st.spinner('Block-bootstrap tail fits…'):st.json(clean(evt_uncertainty(r,tq,confidence)))
        except ValueError as e:st.warning(str(e))
    st.dataframe(pd.DataFrame(stability(r)),width='stretch',hide_index=True)
with tabs[2]:
    st.caption(f'Event: ≥{threshold:.0%} {definition} within {horizon} future observed trading closes.')
    if results:
        if any(st.session_state.run_config[k]!=1 for k in ['vol_stress','jump_stress','crisis_stress']):st.warning('Conditional stress scenario — not an estimated base forecast.')
        table=pd.DataFrame(results).T;st.dataframe(table.drop(columns=['threshold_probabilities','convergence']),width='stretch')
        st.plotly_chart(px.bar(x=list(results),y=[v['probability'] for v in results.values()],labels={'x':'Model','y':'Threshold-crossing probability'}),width='stretch')
        st.metric('Between-model probability range',f"{min(v['probability'] for v in results.values()):.1%} – {max(v['probability'] for v in results.values()):.1%}")
        distributions=pd.DataFrame({n:st.session_state.visual[n]['terminal'][:5000]/s0-1 for n in results})
        st.plotly_chart(px.violin(distributions,y=list(distributions.columns),box=True,labels={'value':'Terminal return','variable':'Model'},title='Forecast distribution comparison (up to 5,000 paths/model)'),width='stretch')
        st.caption('Intervals quantify simulation sampling error conditional on fitted parameters. They do not cover calibration error, structural model error or rare-event uncertainty.')
        ensemble=st.session_state.get('ensemble')
        if ensemble and st.session_state.get('ensemble_key')==str((fitkey,horizon,threshold,definition,student,asymmetric)) and all(n in results for n in ensemble['weights']) and not override and all(st.session_state.run_config[k]==1 for k in ['vol_stress','jump_stress','crisis_stress']):
            w=ensemble['weights'];prob=sum(w[n]*results[n]['probability'] for n in w)
            se=sum(w[n]*results[n]['mc_se'] for n in w)
            st.metric('Validated-weight mixture probability',f'{prob:.2%}');st.write('Model contributions',pd.DataFrame({'weight':w,'probability':{n:results[n]['probability'] for n in w},'contribution':{n:w[n]*results[n]['probability'] for n in w}}))
            st.caption(f'Conservative mixture Monte Carlo SE bound: {se:.4f}; model and weight uncertainty omitted. The bound permits correlated Monte Carlo errors.')
        else:st.info('No eligible ensemble weights for this configuration. Run a broad walk-forward validation; insufficient samples or a single outcome disable stacking.')
        payload=dict(config=st.session_state.run_config,latest_price_date=str(prices.index[-1]),source=prices.attrs,results=results,parameters={n:{k:v for k,v in p.items() if k not in ['returns','residuals','conditional_vol']} for n,p in fitted.items()})
        st.download_button('Export Results JSON',json.dumps(clean(payload),indent=2,allow_nan=False),'crashlab_results.json','application/json')
        st.download_button('Export model summary CSV',table.to_csv(),'crashlab_summary.csv','text/csv')
    else:st.info('Run a simulation to calculate model forecasts.')
with tabs[3]:
    if results:
        chosen=st.selectbox('Trajectory model',list(results));v=st.session_state.visual[chosen];fig=go.Figure();x=np.arange(horizon+1)
        shown=st.select_slider('Displayed sample paths',options=[25,100,500,1000,2000],value=100)
        for row in v['sample'][:shown]:fig.add_trace(go.Scatter(x=x,y=row,mode='lines',line=dict(color='rgba(51,166,190,.12)',width=1),showlegend=False))
        for row in v['worst']:fig.add_trace(go.Scatter(x=x,y=row,line=dict(color='rgba(244,92,107,.6)',width=1),showlegend=False))
        for label,arr in zip(['5th percentile','Median','95th percentile'],v['bands']):fig.add_trace(go.Scatter(x=x,y=arr,name=label,line=dict(width=3)))
        if definition=='starting loss':fig.add_hline(y=s0*(1-threshold),line_dash='dash',line_color='red')
        else:st.caption('Peak drawdown has a path-dependent threshold; no single horizontal crash line applies.')
        fig.update_layout(template='plotly_dark',title=f'{min(shown,len(v["sample"]))} sampled paths, five worst terminal paths, pointwise percentile curves');st.plotly_chart(fig,width='stretch')
        st.plotly_chart(px.histogram(x=1-v['terminal']/s0,nbins=80,title='Simulated terminal loss distribution'),width='stretch')
        st.plotly_chart(px.histogram(x=v['drawdowns'],nbins=80,title='Simulated maximum drawdown distribution'),width='stretch')
        st.dataframe(pd.DataFrame(results[chosen]['convergence']),width='stretch')
        if chosen=='GBM':st.write('Continuous-monitoring analytical GBM crossing probability (daily simulation approaches from below):',gbm_continuous_hit_probability(fitted['GBM'],threshold,horizon))
with tabs[4]:
    st.caption('Each origin refits using only past observations. Daily labels use complete future horizons; recent incomplete labels are excluded. Default stride avoids overlapping event windows.')
    replay=st.selectbox('Replay period',['Broad walk-forward']+list(CRISES));stride=st.number_input('Origin stride (trading days)',1,252,horizon)
    max_origins=st.slider('Maximum chronological origins',5,100,20);bp=st.selectbox('Paths per historical origin',[1000,5000,10000])
    if st.button('Historical Crash Replay'):
        period=CRISES.get(replay,(None,None));bar=st.progress(0.)
        try:
            with st.spinner('Walk-forward refits and simulations…'):
                df=walk_forward(prices,window,horizon,threshold,definition,stride,bp,seed,*period,max_origins,progress=lambda i,n:bar.progress(min(i/n,1.)),student=student,asymmetric=asymmetric)
            st.session_state['backtest']=df;st.session_state['backtest_key']=str((fitkey,horizon,threshold,definition,student,asymmetric))
            try:
                st.session_state['ensemble']=stack(df,['GBM','GARCH','Jump','Regime','Bootstrap']);st.session_state['ensemble_key']=st.session_state['backtest_key']
            except ValueError as e:st.session_state.pop('ensemble',None);st.warning(str(e))
        except ValueError as e:st.error(str(e))
    bt_current=st.session_state.get('backtest_key')==str((fitkey,horizon,threshold,definition,student,asymmetric))
    if 'backtest' in st.session_state and not bt_current:st.warning('Replay settings changed. Run Historical Crash Replay to update historical scores.')
    if 'backtest' in st.session_state and bt_current:
        df=st.session_state.backtest;st.dataframe(df,width='stretch')
        cols=[n for n in ['GBM','GARCH','Jump','Regime','Bootstrap','baseline','unconditional'] if n in df]
        st.plotly_chart(px.line(df,x='date',y=cols+['actual']),width='stretch')
        if 'GARCH_forecast_vol_annual' in df:st.plotly_chart(px.line(df,x='date',y=['GARCH_forecast_vol_annual','realised_forward_vol_annual'],title='Out-of-sample volatility forecast vs subsequent realised RMS volatility'),width='stretch')
        scores={n:metrics(df.loc[df[n].notna(),'actual'],df[n].dropna()) for n in cols};st.dataframe(pd.DataFrame(scores).T,width='stretch')
        cn=st.selectbox('Calibration model',cols);cal=calibration(df.actual,df[cn]);st.dataframe(cal,width='stretch')
        st.plotly_chart(px.scatter(cal,x='predicted',y='observed',size='count',range_x=[0,1],range_y=[0,1]),width='stretch')
        st.download_button('Export replay CSV',df.to_csv(index=False),'crashlab_replay.csv','text/csv')
        if 'ensemble' in st.session_state:st.json(clean({k:v for k,v in st.session_state.ensemble.items() if k not in ['test_predictions','test_dates']}))
        st.caption('Classification metrics use probability ≥ 0.5; undefined rates are NaN. Famous-crisis replay is selected-case analysis, not representative predictive accuracy. Overlapping origins reduce effective sample size.')
