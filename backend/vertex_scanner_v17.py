"""VertexCapital v17: independent, broad-universe, time-aware research scanner.
No live orders. Research-only. Prices are Yahoo auto-adjusted and may be revised.
"""
import argparse, json, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

EXTRA = "ADSK ANSS CDNS SNPS PANW CRWD FTNT DDOG NET MDB PLTR UBER ABNB DASH SHOP SQ PYPL COIN HOOD RBLX SPOT ROKU ZM DOCU TEAM HUBS OKTA TWLO SNOW DELL HPE HPQ SMCI ARM TSM ASML KLAC LRCX MRVL ON NXPI MCHP MPWR TER WDC STX FSLR ENPH VRT CEG VST NRG ETR PWR GEV URI FAST GWW CARR OTIS JCI EMR ROK RSG WM CTAS PAYX ADP CB SPGI MCO ICE CME CBOE MMC AON AJG PGR TRV ALL AFL MET PRU KKR APO BX TROW COF USB PNC TFC FITB KEY NTRS STT BK HIG HUM CI CVS ELV REGN VRTX BIIB ZTS SYK BSX MDT EW DHR IDXX DXCM MRNA HCA DG DLTR TGT KR GIS HSY KDP MNST STZ YUM CMG DPZ MAR HLT RCL CCL DAL UAL LUV EXPE EBAY ETSY PINS SNAP RDDT BABA PDD JD BIDU TCEHY NVO SAP SONY TM HMC BHP RIO VALE GOLD AEM WPM FNV SCCO TECK.TO SHOP.TO CSU.TO CNR.TO CP.TO ENB.TO TRP.TO FTS.TO BCE.TO T.TO ATD.TO WCN.TO BAM.TO BN.TO NA.TO CM.TO BNS.TO MFC.TO SLF.TO".split()

def universe(root, max_stocks):
    import sys
    sys.path.insert(0, str(root/'backend'))
    from market_data import DEFAULT_TICKERS
    tickers=list(dict.fromkeys([t for t in DEFAULT_TICKERS if t not in ('SPY','XEQT.TO')]+EXTRA))
    return tickers[:max_stocks]

def features(close, volume):
    r=close.pct_change(fill_method=None)
    m=pd.DataFrame(index=close.index)
    for days in (5,20,60,120): m[f'mom_{days}']=close.pct_change(days,fill_method=None)
    m['vol20']=r.rolling(20).std()*np.sqrt(252)
    m['vol60']=r.rolling(60).std()*np.sqrt(252)
    m['reversal5']=-m['mom_5']
    m['trend50']=close/close.rolling(50).mean()-1
    m['trend200']=close/close.rolling(200).mean()-1
    m['range60']=close/close.rolling(60).max()-1
    m['volume20']=volume/volume.rolling(20).mean()-1
    return m.replace([np.inf,-np.inf],np.nan)

def panel(prices, volumes, benchmark, horizon=20):
    pieces=[]
    bench_future=benchmark.shift(-horizon)/benchmark-1
    for ticker in prices.columns:
        if ticker=='SPY': continue
        c=prices[ticker]; v=volumes[ticker] if ticker in volumes else pd.Series(index=c.index,dtype=float)
        f=features(c,v)
        f['target']=c.shift(-horizon)/c-1-bench_future
        f['date']=f.index
        f['ticker']=ticker
        pieces.append(f.reset_index(drop=True))
    return pd.concat(pieces,ignore_index=True)

def models():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.svm import SVR
    return {
        'ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=20)),
        'random_forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=100,max_depth=7,min_samples_leaf=15,n_jobs=-1,random_state=42)),
        'svr':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),SVR(C=.05,epsilon=.03)),
        'hist_gradient_boosting':make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=12,min_samples_leaf=30,l2_regularization=10,learning_rate=.035,random_state=42))}

def run(root, max_stocks, period, steps, horizon, min_train, min_coverage):
    import yfinance as yf
    tickers=universe(root,max_stocks)
    print(f'Downloading {len(tickers)} requested symbols in batches; unavailable symbols will be skipped...',flush=True)
    frames=[]; volframes=[]; failed=[]
    for k in range(0,len(tickers),40):
        batch=tickers[k:k+40]+(['SPY'] if k==0 else [])
        try:
            raw=yf.download(batch,period=period,auto_adjust=True,group_by='column',threads=True,progress=False)
            if raw.empty: failed.extend(batch); continue
            for field,dest in [('Close',frames),('Volume',volframes)]:
                if isinstance(raw.columns,pd.MultiIndex):
                    part=raw[field] if field in raw.columns.get_level_values(0) else pd.DataFrame()
                else: part=raw[[field]].rename(columns={field:batch[0]})
                if not part.empty: dest.append(part)
        except Exception as exc:
            print(f'Batch {k//40+1} failed: {exc}',flush=True);failed.extend(batch)
    if not frames: raise RuntimeError('No price data downloaded; check internet access')
    prices=pd.concat(frames,axis=1); prices=prices.loc[:,~prices.columns.duplicated()].sort_index()
    volumes=pd.concat(volframes,axis=1);volumes=volumes.loc[:,~volumes.columns.duplicated()].reindex(prices.index)
    if 'SPY' not in prices: raise RuntimeError('SPY benchmark download failed')
    valid=[t for t in tickers if t in prices and prices[t].notna().sum()>=min_coverage]
    if len(valid)<10: raise RuntimeError(f'Only {len(valid)} tickers meet data coverage; need >=10')
    prices=prices[valid+['SPY']];volumes=volumes.reindex(columns=valid)
    print(f'{len(valid)} symbols have sufficient history; creating forward labels...',flush=True)
    data=panel(prices,volumes,prices['SPY'],horizon)
    cols=[c for c in data if c not in ('date','ticker','target')]
    dates=sorted(pd.to_datetime(data.date.dropna().unique()))
    dates=[d for d in dates if d>=dates[0]+pd.Timedelta(days=min_train*1.4)]
    # Train on only rows with labels fully matured BEFORE test signal date; purge by horizon.
    usable=dates[::steps]
    # Keep at most 8 evaluation folds to limit compute; no tuning on their outcomes.
    evaluation=usable[-3:] if len(usable)>3 else usable
    diagnostics=[]; last=None
    for d in evaluation:
        eligible=data[(data.date<d-pd.Timedelta(days=int(horizon*1.6))) & data.target.notna()]
        eligible=eligible[eligible.date>=d-pd.Timedelta(days=int(min_train*2.3))]
        test=data[(data.date==d)&data[cols].notna().sum(axis=1).ge(7)].copy()
        if len(eligible)<500 or len(test)<10:continue
        # Do not train on labels whose target end date reaches test date.
        eligible=eligible.sample(n=min(len(eligible),12000),random_state=42).sort_values('date')
        x=eligible[cols]; y=eligible.target.clip(-.8,.8)
        print(f'Fold {pd.Timestamp(d).date()}: training on {len(x)} samples...',flush=True)
        preds={}
        for name,model in models().items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(x.tail(2500) if name=='svr' else x, y.tail(2500) if name=='svr' else y)
                preds[name]=model.predict(test[cols])
        test['ensemble']=np.median(np.column_stack(list(preds.values())),axis=1)
        for name,values in preds.items():test[name]=values
        if test.target.notna().sum()>=10:
            scored=test.dropna(subset=['target'])
            ic=scored['ensemble'].corr(scored['target'],method='spearman')
            diagnostics.append({'date':str(pd.Timestamp(d).date()),'n':int(len(scored)),'spearman_ic':float(ic) if pd.notna(ic) else None})
        last=test
        print(f'Fold {pd.Timestamp(d).date()}: {len(test)} stocks scored',flush=True)
    # Latest predictions trained using matured labels; do not treat final fold as current.
    latest=prices.index.max();eligible=data[(data.date<latest-pd.Timedelta(days=int(horizon*1.6))) & data.target.notna()]
    test=data[(data.date==latest)&data[cols].notna().sum(axis=1).ge(7)].copy()
    if len(test)<10:raise RuntimeError('Not enough current feature rows')
    preds={}
    for name,model in models().items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sample=eligible.sample(n=min(len(eligible),12000),random_state=42).sort_values('date')
            x=sample[cols];y=sample.target.clip(-.8,.8)
            print(f'Latest: fitting {name} on {min(len(x),2500) if name=="svr" else len(x)} samples...',flush=True)
            model.fit(x.tail(2500) if name=='svr' else x,y.tail(2500) if name=='svr' else y)
            preds[name]=model.predict(test[cols])
    arr=np.column_stack(list(preds.values()))
    for name,values in preds.items():test[name]=values
    test['ensemble']=np.median(arr,axis=1)
    test['model_spread']=arr.std(axis=1)
    test['positive_model_fraction']=(arr>0).mean(axis=1)
    test['research_flag']=np.where((test.positive_model_fraction>=.75)&(test.ensemble>0),'RESEARCH_CANDIDATE','UNCONFIRMED')
    out=root/'data'/'recommendations'/'upgrade17';out.mkdir(parents=True,exist_ok=True)
    output=test[['date','ticker','ensemble','model_spread','positive_model_fraction','research_flag']+list(preds)].sort_values('ensemble',ascending=False)
    output.to_csv(out/'expanded_universe_rankings.csv',index=False)
    icvals=[v['spearman_ic'] for v in diagnostics if v['spearman_ic'] is not None]
    report={'version':'17.0','status':'RESEARCH_ONLY_NO_TRADE_AUTHORIZATION','generated_utc':datetime.now(timezone.utc).isoformat(),'latest_price_date':str(pd.Timestamp(latest).date()),'requested_symbols':len(tickers),'downloaded_usable_symbols':len(valid),'latest_scored_symbols':len(test),'model_families':list(preds),'experimental_lstm':'NOT_INSTALLED: requires separate time-series validation and TensorFlow/PyTorch','fold_diagnostics':diagnostics,'mean_fold_spearman_ic':float(np.mean(icvals)) if icvals else None,'limitations':['Yahoo historical universe is NOT survivorship-free or point-in-time','Adjusted historical data may be revised','Validation folds overlap when steps < horizon; not independent','No costs or execution modeled in this scanner','Research candidate is NOT a buy recommendation','Latest data depends on download provider availability'],'files':['expanded_universe_rankings.csv','research_report.json']}
    (out/'research_report.json').write_text(json.dumps(report,indent=2))
    print(f'UPGRADE 17 COMPLETE: {len(test)} stocks, {len(preds)} models. Report: {out}',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default='.');a.add_argument('--max-stocks',type=int,default=300);a.add_argument('--period',default='7y');a.add_argument('--steps',type=int,default=63);a.add_argument('--horizon',type=int,default=20);a.add_argument('--min-train',type=int,default=756);a.add_argument('--min-coverage',type=int,default=500)
    v=a.parse_args();run(Path(v.root).resolve(),v.max_stocks,v.period,v.steps,v.horizon,v.min_train,v.min_coverage)
