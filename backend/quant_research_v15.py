"""VertexCapital v15: evidence-first, offline quantitative research audit.
No orders, no live feeds, no training on future outcomes, no investment advice.
"""
from __future__ import annotations
import argparse, json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='15.0'
def finite(x):
 try:
  y=float(x);return y if math.isfinite(y) else None
 except (TypeError,ValueError):return None

def paired_block_bootstrap(diff, block=20, reps=1500, seed=15):
 """Moving block bootstrap for mean daily excess return, preserving local dependence."""
 x=np.asarray(diff,dtype=float);x=x[np.isfinite(x)];n=len(x)
 if n<60:return {'status':'INSUFFICIENT_OBSERVATIONS','observations':n}
 b=min(block,n);rng=np.random.default_rng(seed);means=np.empty(reps)
 for i in range(reps):
  starts=rng.integers(0,n-b+1,size=math.ceil(n/b))
  means[i]=np.concatenate([x[s:s+b] for s in starts])[:n].mean()
 lo,hi=np.quantile(means,[.025,.975]);return {'status':'DESCRIPTIVE_ONLY','observations':n,'daily_excess_mean':float(x.mean()),'annualized_arithmetic_excess_estimate':float(x.mean()*252),'bootstrap_95pct_annualized_interval':[float(lo*252),float(hi*252)],'bootstrap_probability_mean_positive':float(np.mean(means>0)),'block_length':b,'replicates':reps}

def audit_backtest(df):
 required={'Date','net_return','benchmark_return','portfolio_value'}
 if not required.issubset(df):raise ValueError('Backtest missing columns: '+str(sorted(required-set(df))))
 d=df.copy();d['Date']=pd.to_datetime(d['Date'],errors='coerce');d=d.sort_values('Date')
 issues=[]
 if d['Date'].isna().any():issues.append('INVALID_DATES')
 if d['Date'].duplicated().any():issues.append('DUPLICATE_DATES')
 for c in ['net_return','benchmark_return','portfolio_value']:
  d[c]=pd.to_numeric(d[c],errors='coerce')
  if d[c].isna().any():issues.append('MISSING_'+c.upper())
 if (d['portfolio_value']<=0).any():issues.append('NONPOSITIVE_EQUITY')
 if (d['net_return']<=-1).any():issues.append('IMPOSSIBLE_NET_RETURN')
 clean=d.dropna(subset=['Date','net_return','benchmark_return','portfolio_value'])
 if clean.empty:raise ValueError('No valid return observations')
 p=clean.net_return.to_numpy();b=clean.benchmark_return.to_numpy();ex=p-b
 equity=np.cumprod(1+p);peak=np.maximum.accumulate(np.r_[1.,equity])[1:]
 vol=np.std(p,ddof=1)*np.sqrt(252) if len(p)>1 else None
 neg=p[p<0];down=np.sqrt(np.mean(neg**2))*np.sqrt(252) if len(neg) else None
 years=len(p)/252
 def geo(r):return float(np.prod(1+r)**(1/years)-1) if years>0 and np.all(1+r>0) else None
 rolling=pd.Series(ex).rolling(63,min_periods=63).mean()*252
 return {'period':{'start':str(clean.Date.iloc[0].date()),'end':str(clean.Date.iloc[-1].date()),'sessions':len(clean)},'performance':{'strategy_cagr':geo(p),'benchmark_cagr':geo(b),'annualized_volatility':finite(vol),'sharpe_zero_rf':finite(np.mean(p)*252/vol) if vol else None,'sortino_zero_rf':finite(np.mean(p)*252/down) if down else None,'max_drawdown':float(np.min(equity/peak-1)),'daily_tracking_error_annualized':finite(np.std(ex,ddof=1)*np.sqrt(252)),'information_ratio':finite(np.mean(ex)*252/(np.std(ex,ddof=1)*np.sqrt(252))) if len(ex)>1 and np.std(ex,ddof=1)>0 else None,'positive_63d_excess_windows_fraction':finite((rolling.dropna()>0).mean()) if rolling.notna().any() else None},'paired_bootstrap':paired_block_bootstrap(ex),'data_issues':sorted(set(issues)),'warnings':['Returns are from a historical research simulation, not live execution.','Bootstrap intervals describe sampling variability, not guaranteed future performance.','Overlapping rolling windows are not independent.','A zero risk-free rate is assumed for Sharpe/Sortino.']}

def audit_predictions(df):
 req={'date','ticker','prediction','linear_prediction','nonlinear_prediction'}
 if not req.issubset(df):raise ValueError('Predictions missing columns: '+str(sorted(req-set(df))))
 d=df.copy();d['date']=pd.to_datetime(d.date,errors='coerce')
 for c in ['prediction','linear_prediction','nonlinear_prediction']:d[c]=pd.to_numeric(d[c],errors='coerce')
 issues=[]
 if d.duplicated(['date','ticker']).any():issues.append('DUPLICATE_DATE_TICKER')
 if d[['prediction','linear_prediction','nonlinear_prediction']].isna().any().any():issues.append('MISSING_MODEL_OUTPUTS')
 if d['date'].isna().any():issues.append('INVALID_SIGNAL_DATE')
 d=d.dropna(subset=['date','ticker','prediction','linear_prediction','nonlinear_prediction'])
 if d.empty:raise ValueError('No valid predictions')
 latest=d[d.date==d.date.max()].copy();latest['agreement']=(np.sign(latest.linear_prediction)==np.sign(latest.nonlinear_prediction))
 latest['disagreement_abs']=(latest.linear_prediction-latest.nonlinear_prediction).abs()
 latest['spread_to_signal']=latest.disagreement_abs/(latest.prediction.abs()+.005)
 latest=latest.sort_values('prediction',ascending=False)
 out=[]
 for r in latest.itertuples():
  reasons=[]
  if r.prediction<=0:reasons.append('NONPOSITIVE_PREDICTION')
  if not r.agreement:reasons.append('MODEL_DIRECTION_CONFLICT')
  if r.spread_to_signal>2:reasons.append('HIGH_MODEL_DISAGREEMENT')
  reasons.append('NO_POINT_IN_TIME_REALIZED_LABELS_FOR_VALIDATION')
  out.append({'ticker':str(r.ticker),'predicted_20d_excess':float(r.prediction),'linear_prediction':float(r.linear_prediction),'nonlinear_prediction':float(r.nonlinear_prediction),'direction_agreement':bool(r.agreement),'model_disagreement':float(r.disagreement_abs),'relative_disagreement':float(r.spread_to_signal),'research_status':'UNVALIDATED','diagnostics':reasons})
 return {'signal_date':str(latest.date.max().date()),'rows_all_dates':len(d),'unique_dates':int(d.date.nunique()),'latest_universe':len(latest),'model_direction_agreement_rate':float(latest.agreement.mean()),'issues':issues,'validation':{'out_of_sample_ic':None,'calibration':None,'realized_excess_returns':None,'status':'NOT_IDENTIFIABLE_FROM_PREDICTIONS_ALONE'},'rankings':out}

def audit_weights(df):
 req={'ticker','weight','date'}
 if not req.issubset(df):raise ValueError('Weights missing columns')
 d=df.copy();d.date=pd.to_datetime(d.date,errors='coerce');d.weight=pd.to_numeric(d.weight,errors='coerce');d=d.dropna(subset=['date','ticker','weight'])
 if d.empty:return {'status':'NO_VALID_WEIGHTS'}
 issues=[]
 if d.duplicated(['date','ticker']).any():issues.append('DUPLICATE_DATE_TICKER')
 if (d.weight<0).any():issues.append('NEGATIVE_WEIGHTS')
 piv=d.pivot_table(index='date',columns='ticker',values='weight',aggfunc='sum').fillna(0).sort_index()
 totals=piv.sum(axis=1);last=piv.iloc[-1];turnover=piv.diff().abs().sum(axis=1).iloc[1:]/2
 return {'latest_date':str(piv.index[-1].date()),'latest_gross_exposure':float(last.abs().sum()),'latest_net_exposure':float(last.sum()),'latest_largest_weight':float(last.abs().max()),'latest_effective_number_of_positions':float(1/(last**2).sum()) if (last**2).sum()>0 else None,'latest_positions':int((last!=0).sum()),'average_one_way_weight_turnover':finite(turnover.mean()) if len(turnover) else None,'max_total_weight':float(totals.max()),'issues':issues,'notes':['Weights do not establish realized trade executions.','Turnover is computed from successive target weight snapshots and is not trade-level realized turnover.']}

def run(root):
 root=Path(root);data=root/'data';files={'backtest':data/'backtest_results.csv','predictions':data/'walk_forward_predictions.csv','weights':data/'portfolio_weights.csv'}
 missing=[str(v) for v in files.values() if not v.is_file()]
 if missing:raise FileNotFoundError('Missing research inputs: '+', '.join(missing))
 report={'engine_version':VERSION,'status':'RESEARCH_ONLY_NOT_VALIDATED','input_files':{},'limits':['No live market data','No execution or brokerage connectivity','No prediction calibration without realized point-in-time labels','Historical simulations may be biased by survivorship, selection and data revisions']}
 for k,p in files.items():report['input_files'][k]={'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 report['backtest']=audit_backtest(pd.read_csv(files['backtest']))
 report['predictions']=audit_predictions(pd.read_csv(files['predictions']))
 report['portfolio']=audit_weights(pd.read_csv(files['weights']))
 report['research_readiness']={'can_claim_predictive_skill':False,'can_claim_calibrated_probabilities':False,'can_place_trades':False,'required_next_evidence':['Realized forward 20-session excess returns aligned by signal date and ticker','Point-in-time investable universe and delisting history','Feature availability timestamps and corporate action audit','Purged/embargoed walk-forward evaluation with untouched holdout','Transaction-cost and capacity sensitivity on reproducible fills']}
 out=data/'recommendations'/'upgrade15';out.mkdir(parents=True,exist_ok=True)
 (out/'institutional_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
 pd.DataFrame(report['predictions']['rankings']).to_csv(out/'model_diagnostics.csv',index=False)
 print('Upgrade 15 audit complete:',out)
 print('Latest universe:',report['predictions']['latest_universe'],'| Direction agreement:',round(report['predictions']['model_direction_agreement_rate']*100,1),'%')
 print('Backtest bootstrap status:',report['backtest']['paired_bootstrap']['status'])
 print('No trading recommendations generated; validation remains unverified.')
 return report
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',default='.')
 run(a.parse_args().root)
