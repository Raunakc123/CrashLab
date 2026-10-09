import numpy as np
import pandas as pd

SYMBOLS={'S&P 500':'^GSPC','NASDAQ':'^IXIC','Dow Jones':'^DJI','NIFTY 50':'^NSEI'}
def validate(frame):
    if not {'Date','Close'} <= set(frame.columns):
        raise ValueError('CSV must contain Date and Close columns.')
    d=frame[['Date','Close']].copy()
    d['Date']=pd.to_datetime(d.Date,utc=True).dt.tz_localize(None)
    if d['Date'].isna().any():raise ValueError('Missing dates are not allowed.')
    d['Close']=pd.to_numeric(d.Close,errors='raise')
    d=d.sort_values('Date')
    if d.Date.duplicated().any(): raise ValueError('Duplicate dates.')
    if not np.isfinite(d.Close).all() or (d.Close<=0).any(): raise ValueError('Prices must be finite and positive.')
    if len(d)<252: raise ValueError('At least 252 price observations required.')
    return d.set_index('Date').Close

def fetch(symbol,start,end):
    import yfinance as yf
    d=yf.download(symbol,start=str(start),end=str(end),auto_adjust=True,progress=False,threads=False)
    if d.empty: raise ValueError('Provider returned no prices. Upload a real Date,Close CSV instead.')
    c=d['Close']
    if isinstance(c,pd.DataFrame): c=c.iloc[:,0]
    out=validate(pd.DataFrame({'Date':c.index,'Close':c.values}))
    out.attrs.update(source='Yahoo Finance via yfinance; adjusted closes',symbol=symbol)
    return out

def fetch_fred(symbol,start,end):
    """Explicit index-series alternative. No stock/NIFTY substitution."""
    import io
    import requests
    ids={'^GSPC':'SP500','^IXIC':'NASDAQCOM','^DJI':'DJIA'}
    if symbol not in ids:raise ValueError('FRED fallback is available only for the three US indices.')
    series=ids[symbol]
    response=requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv',params={'id':series,'cosd':str(start),'coed':str(end)},timeout=30)
    response.raise_for_status();frame=pd.read_csv(io.StringIO(response.text))
    frame=frame.rename(columns={'observation_date':'Date',series:'Close'})
    frame['Close']=pd.to_numeric(frame['Close'],errors='coerce');frame=frame.dropna(subset=['Close'])
    out=validate(frame);out=out.loc[(out.index>=pd.Timestamp(start))&(out.index<pd.Timestamp(end))]
    if len(out)<252:raise ValueError('FRED returned insufficient observations.')
    out.attrs.update(source=f'FRED {series}: daily index close; not total return; available history may be limited',symbol=symbol,url=f'https://fred.stlouisfed.org/series/{series}')
    return out

def fetch_market(symbol,start,end,provider='Yahoo with FRED fallback'):
    if provider=='FRED index data':return fetch_fred(symbol,start,end)
    try:return fetch(symbol,start,end)
    except Exception as error:
        try:
            p=fetch_fred(symbol,start,end);p.attrs['provider_note']='Yahoo unavailable; explicit FRED index-series fallback';return p
        except Exception:raise ValueError(f'Historical download unavailable: {error}. Upload a real CSV.') from error
