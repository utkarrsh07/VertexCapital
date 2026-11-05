"""VertexCapital v16: offline, evidence-first validation and governance.
No live execution, no data downloading, no unvalidated buy recommendations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = '16.0'
HORIZON = 20


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def read_csv(path, required):
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'{path.name}: missing {sorted(missing)}')
    return frame


def validate_predictions(frame):
    p = frame.copy()
    p['date'] = pd.to_datetime(p['date'], errors='coerce')
    for col in ('prediction', 'linear_prediction', 'nonlinear_prediction'):
        p[col] = pd.to_numeric(p[col], errors='coerce')
    flags = []
    if p['date'].isna().any(): flags.append('INVALID_DATES')
    if p.duplicated(['date','ticker']).any(): flags.append('DUPLICATE_DATE_TICKER')
    if p[['prediction','linear_prediction','nonlinear_prediction']].isna().any().any(): flags.append('MISSING_PREDICTIONS')
    if p['date'].is_monotonic_increasing is False: flags.append('UNSORTED_DATES')
    valid = p.dropna(subset=['date','ticker','prediction','linear_prediction','nonlinear_prediction']).sort_values(['date','ticker'])
    dates = pd.DatetimeIndex(sorted(valid.date.unique()))
    latest = valid[valid.date == dates[-1]].copy() if len(dates) else valid
    agree = (np.sign(latest.linear_prediction) == np.sign(latest.nonlinear_prediction))
    disagreement = (latest.linear_prediction-latest.nonlinear_prediction).abs()
    latest['model_agreement'] = agree
    latest['model_spread'] = disagreement
    latest['spread_to_signal'] = disagreement / (latest.prediction.abs()+0.005)
    latest['research_state'] = 'UNVALIDATED'
    return valid, latest, {'rows':len(valid),'dates':len(dates),'first_signal':str(dates[0].date()) if len(dates) else None,'last_signal':str(dates[-1].date()) if len(dates) else None,'latest_universe':len(latest),'direction_agreement_fraction':number(agree.mean()) if len(latest) else None,'median_model_spread':number(disagreement.median()) if len(latest) else None,'issues':flags}


def aligned_labels(predictions, labels, horizon=HORIZON):
    """Only explicit point-in-time forward excess labels can validate alpha.
    Expected: date,ticker,realized_20d_excess; optionally label_end_date.
    """
    required = {'date','ticker','realized_20d_excess'}
    if not required.issubset(labels):
        raise ValueError('Labels require date,ticker,realized_20d_excess')
    y=labels.copy()
    y['date']=pd.to_datetime(y.date,errors='coerce')
    y['realized_20d_excess']=pd.to_numeric(y.realized_20d_excess,errors='coerce')
    if y.duplicated(['date','ticker']).any():raise ValueError('Duplicate point-in-time labels')
    if 'label_end_date' in y:
        y['label_end_date']=pd.to_datetime(y.label_end_date,errors='coerce')
        if (y['label_end_date']<=y['date']).any():raise ValueError('Invalid label end date')
    else:
        raise ValueError('Require label_end_date for leakage-safe validation')
    p=predictions.merge(y[['date','ticker','realized_20d_excess','label_end_date']],on=['date','ticker'],how='inner',validate='one_to_one')
    p=p.dropna(subset=['realized_20d_excess','label_end_date'])
    return p


def predictive_skill(p):
    if p.empty:return {'status':'INSUFFICIENT_LABELS'}
    groups=[]
    for date,g in p.groupby('date'):
        if len(g)<5 or g.prediction.nunique()<2 or g.realized_20d_excess.nunique()<2:continue
        groups.append({'date':str(date.date()),'n':len(g),'spearman_ic':number(g.prediction.corr(g.realized_20d_excess,method='spearman')),'pearson_ic':number(g.prediction.corr(g.realized_20d_excess)),'top_quintile_minus_bottom':number(g.nlargest(max(1,len(g)//5),'prediction').realized_20d_excess.mean()-g.nsmallest(max(1,len(g)//5),'prediction').realized_20d_excess.mean())})
    clean=pd.DataFrame(groups)
    if len(clean)<12:return {'status':'INSUFFICIENT_INDEPENDENT_SIGNAL_DATES','usable_dates':len(clean),'minimum_dates':12}
    ics=clean.spearman_ic.dropna().to_numpy()
    if len(ics)<12:return {'status':'INSUFFICIENT_IC_OBSERVATIONS'}
    rng=np.random.default_rng(16)
    # Resample dates, not rows, to avoid cross-sectional pseudo-replication.
    boot=np.mean(rng.choice(ics,size=(2000,len(ics)),replace=True),axis=1)
    return {'status':'DESCRIPTIVE_OOS_NOT_PROOF','usable_dates':len(clean),'observations':len(p),'mean_spearman_ic':number(ics.mean()),'median_spearman_ic':number(np.median(ics)),'positive_ic_fraction':number(np.mean(ics>0)),'bootstrap_95pct_mean_ic':[number(x) for x in np.quantile(boot,[.025,.975])],'mean_top_minus_bottom_20d_excess':number(clean.top_quintile_minus_bottom.mean()),'note':'Overlapping 20-session labels induce serial dependence; date bootstrap is descriptive only. Full inference requires purged/embargoed retraining.'}


def stress_test(backtest):
    d=backtest.copy();d['Date']=pd.to_datetime(d.Date,errors='coerce')
    d=d.sort_values('Date')
    if d.Date.isna().any() or d.Date.duplicated().any():raise ValueError('Invalid/duplicate backtest dates')
    for c in ('net_return','benchmark_return','trading_cost'):
        d[c]=pd.to_numeric(d[c],errors='coerce')
    if d[['net_return','benchmark_return','trading_cost']].isna().any().any():raise ValueError('Missing backtest returns or costs')
    if (d.net_return<=-1).any():raise ValueError('Impossible returns')
    r=d.net_return.to_numpy();b=d.benchmark_return.to_numpy();cost=d.trading_cost.to_numpy()
    def metrics(x):
        equity=np.cumprod(1+x);peaks=np.maximum.accumulate(np.r_[1,equity])[1:]
        years=len(x)/252
        return {'cagr':number(equity[-1]**(1/years)-1) if len(x) else None,'max_drawdown':number(np.min(equity/peaks-1)) if len(x) else None,'annual_volatility':number(np.std(x,ddof=1)*np.sqrt(252)) if len(x)>1 else None}
    scenarios={}
    for multiple in (0,1,2,3,5):
        # net returns already include baseline trading cost. Additional cost = (multiple-1)*recorded cost.
        adjusted=r-(multiple-1)*cost
        scenarios[f'{multiple}x_recorded_cost']=metrics(adjusted)
    excess=r-b
    windows={}
    for n in (63,126,252):
        if len(excess)>=n:
            rolling=pd.Series(excess).rolling(n).sum().dropna()
            windows[str(n)]={'positive_excess_fraction':number((rolling>0).mean()),'worst_arithmetic_excess':number(rolling.min()),'best_arithmetic_excess':number(rolling.max())}
    # Moving-block bootstrap on paired daily excess, descriptive not p-value.
    rng=np.random.default_rng(160);length=min(20,len(excess));reps=2500
    samples=[]
    for _ in range(reps):
        starts=rng.integers(0,len(excess)-length+1,size=math.ceil(len(excess)/length))
        sample=np.concatenate([excess[s:s+length] for s in starts])[:len(excess)]
        samples.append(sample.mean()*252)
    return {'period':{'start':str(d.Date.iloc[0].date()),'end':str(d.Date.iloc[-1].date()),'sessions':len(d)},'strategy':metrics(r),'benchmark':metrics(b),'annualized_arithmetic_excess':number(excess.mean()*252),'block_bootstrap_95pct_excess':[number(v) for v in np.quantile(samples,[.025,.975])],'cost_sensitivity':scenarios,'rolling_excess_windows':windows,'caution':'Historical simulation; cost multipliers are sensitivity tests, not actual fills. Bootstrap does not remove backtest selection bias.'}


def trade_audit(trades):
    if trades is None or trades.empty:return {'status':'NOT_AVAILABLE'}
    d=trades.copy()
    if not {'signal_date','ticker','net_return','outcome'}.issubset(d):return {'status':'UNSUPPORTED_SCHEMA'}
    d.net_return=pd.to_numeric(d.net_return,errors='coerce')
    closed=d[d.net_return.notna()].copy()
    if closed.empty:return {'status':'NO_CLOSED_TRADES'}
    per=closed.groupby('ticker').net_return.agg(['count','mean']).sort_values('count',ascending=False)
    return {'status':'DESCRIPTIVE_NOT_INDEPENDENT_VALIDATION','closed_trades':len(closed),'mean_net_return':number(closed.net_return.mean()),'median_net_return':number(closed.net_return.median()),'positive_fraction':number((closed.net_return>0).mean()),'worst_net_return':number(closed.net_return.min()),'best_net_return':number(closed.net_return.max()),'by_ticker':[{'ticker':str(t),'count':int(r['count']),'mean_return':number(r['mean'])} for t,r in per.iterrows()],'warning':'Trades are strategy-selected, not a complete set of model forward labels; cannot estimate model IC from them.'}


def portfolio_audit(weights):
    w=weights.copy();w.date=pd.to_datetime(w.date,errors='coerce');w.weight=pd.to_numeric(w.weight,errors='coerce')
    if w.duplicated(['date','ticker']).any():raise ValueError('Duplicate portfolio positions')
    w=w.dropna(subset=['date','ticker','weight'])
    matrix=w.pivot(index='date',columns='ticker',values='weight').fillna(0).sort_index()
    latest=matrix.iloc[-1]
    return {'latest_date':str(matrix.index[-1].date()),'gross_exposure':number(latest.abs().sum()),'largest_absolute_weight':number(latest.abs().max()),'effective_number_positions':number(1/(latest**2).sum()) if (latest**2).sum()>0 else None,'average_one_way_target_turnover':number(matrix.diff().abs().sum(axis=1).iloc[1:].mean()/2) if len(matrix)>1 else None,'max_sector_exposure':'UNAVAILABLE_WITHOUT_POINT_IN_TIME_SECTOR_CLASSIFICATION'}


def run(root):
    root=Path(root);data=root/'data';required={'predictions':data/'walk_forward_predictions.csv','backtest':data/'backtest_results.csv','weights':data/'portfolio_weights.csv'}
    for name,path in required.items():
        if not path.is_file():raise FileNotFoundError(f'Missing {name}: {path}')
    p=read_csv(required['predictions'],['date','ticker','prediction','linear_prediction','nonlinear_prediction'])
    b=read_csv(required['backtest'],['Date','net_return','benchmark_return','trading_cost'])
    w=read_csv(required['weights'],['date','ticker','weight'])
    valid,latest,pdiag=validate_predictions(p)
    optional=data/'realized_20d_excess_labels.csv'
    if optional.is_file():
        labeled=aligned_labels(valid,pd.read_csv(optional))
        skill=predictive_skill(labeled)
        skill['matched_fraction']=number(len(labeled)/len(valid)) if len(valid) else None
    else:
        skill={'status':'NOT_MEASURABLE','reason':'Missing independently reconstructed forward 20-session excess-return labels with label_end_date','expected_optional_path':str(optional.relative_to(root)),'warning':'Do not infer prediction accuracy from model agreement or trade-selected outcomes.'}
    trades_path=data/'recommendations'/'strategy_v2'/'trade_audit.csv'
    trades=pd.read_csv(trades_path) if trades_path.is_file() else None
    report={'version':VERSION,'status':'RESEARCH_ONLY_NOT_VALIDATED','inputs':{name:{'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in required.items()},'prediction_diagnostics':pdiag,'predictive_skill':skill,'backtest_stress':stress_test(b),'portfolio':portfolio_audit(w),'historical_trade_outcomes':trade_audit(trades),'governance':{'authorized_to_trade':False,'can_claim_predictive_skill':skill.get('status')=='DESCRIPTIVE_OOS_NOT_PROOF' and False,'point_in_time_universe_verified':False,'corporate_actions_verified':False,'independent_holdout_verified':False,'cost_capacity_verified':False,'critical_next_steps':['Reconstruct forward excess-return labels from timestamped adjusted prices and benchmark','Audit point-in-time constituent membership and delisted assets','Re-train with purged/embargoed walk-forward splits and frozen holdout','Verify costs, fill timing, capacity and market impact','Run prospective paper-trading shadow validation']}}
    out=data/'recommendations'/'upgrade16';out.mkdir(parents=True,exist_ok=True)
    (out/'research_validation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    cols=['date','ticker','prediction','linear_prediction','nonlinear_prediction','model_agreement','model_spread','spread_to_signal','research_state']
    latest[cols].sort_values('prediction',ascending=False).to_csv(out/'latest_model_diagnostics.csv',index=False)
    print('UPGRADE 16 RESEARCH VALIDATION COMPLETE')
    print('Predictions:',pdiag['rows'],'| Signal dates:',pdiag['dates'],'| Latest universe:',pdiag['latest_universe'])
    print('Predictive skill:',skill['status'])
    print('Report:',out/'research_validation.json')
    print('NO LIVE TRADE AUTHORIZATION')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.')
    run(parser.parse_args().root)
