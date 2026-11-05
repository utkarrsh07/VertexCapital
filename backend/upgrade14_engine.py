"""Upgrade 14: auditable, abstention-first cross-sectional research diagnostics.
No live feed, calibrated probabilities, or executable trading signals.
Runs on existing saved artifacts; never trains on future observations.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_RANK = {'ticker', 'predicted_20d_excess_return'}

def finite(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None

def percentile(values, q):
    return float(np.quantile(values,q)) if len(values) else None

def build_report(root: Path) -> dict:
    root=Path(root)
    source=root/'recommendations'
    rankings=pd.read_csv(source/'latest_rankings.csv')
    if not REQUIRED_RANK.issubset(rankings.columns):
        raise ValueError('Rankings missing required columns')
    if rankings.ticker.duplicated().any():
        raise ValueError('Duplicate tickers in latest rankings')
    plans=pd.read_csv(source/'trade_quality_report.csv') if (source/'trade_quality_report.csv').exists() else pd.DataFrame()
    predictions=pd.read_csv(root/'walk_forward_predictions.csv') if (root/'walk_forward_predictions.csv').exists() else pd.DataFrame()
    audit=pd.read_csv(source/'historical_trade_audit.csv') if (source/'historical_trade_audit.csv').exists() else pd.DataFrame()
    if not predictions.empty and 'date' in predictions:
        predictions['date']=pd.to_datetime(predictions['date'],errors='coerce')
    latest=str(rankings['date'].iloc[0]) if 'date' in rankings and len(rankings) else 'unknown'
    if 'date' in rankings and rankings['date'].nunique()>1:
        raise ValueError('Rankings contain multiple dates; use a single snapshot')
    plan_by={str(r['ticker']):r for _,r in plans.iterrows()} if 'ticker' in plans else {}
    records=[]
    for _,row in rankings.iterrows():
        ticker=str(row.ticker); alpha=finite(row.predicted_20d_excess_return)
        history=predictions[predictions.ticker==ticker].sort_values('date') if 'ticker' in predictions else pd.DataFrame()
        # Only use predictions STRICTLY PRIOR to the current signal date for stability diagnostics.
        signal_date=pd.to_datetime(latest,errors='coerce')
        if pd.notna(signal_date) and 'date' in history:
            history=history[history.date<signal_date]
        history=history.tail(24)
        score_hist=pd.to_numeric(history.get('prediction',pd.Series(dtype=float)),errors='coerce').dropna().to_numpy()
        agreement=None
        if {'linear_prediction','nonlinear_prediction'}.issubset(history.columns) and len(history):
            a=pd.to_numeric(history.linear_prediction,errors='coerce').to_numpy()
            b=pd.to_numeric(history.nonlinear_prediction,errors='coerce').to_numpy()
            ok=np.isfinite(a)&np.isfinite(b)
            agreement=float(np.mean(np.sign(a[ok])==np.sign(b[ok]))) if ok.any() else None
        stability=float(np.mean(score_hist>0)) if len(score_hist)>=8 else None
        volatility=float(np.std(score_hist,ddof=1)) if len(score_hist)>=8 else None
        plan=plan_by.get(ticker)
        quality=str(plan.get('quality_status','UNKNOWN')).upper() if plan is not None else 'MISSING'
        rr=finite(plan.get('reward_risk_ratio')) if plan is not None else None
        stop=finite(plan.get('stop_distance_pct')) if plan is not None else None
        historical_count=int(len(history))
        # Diagnostics are NOT calibrated success probabilities.
        reasons=[]
        if alpha is None or alpha<=0:reasons.append('NO_POSITIVE_MODEL_ALPHA')
        if quality!='PASS':reasons.append('TRADE_QUALITY_'+quality)
        if rr is None or rr<1.5:reasons.append('INSUFFICIENT_REWARD_RISK')
        if stop is None or stop>0.10:reasons.append('STOP_RISK_TOO_HIGH_OR_UNKNOWN')
        if historical_count<8:reasons.append('INSUFFICIENT_MODEL_HISTORY')
        if stability is None or stability<0.65:reasons.append('UNSTABLE_HISTORICAL_MODEL_SIGN')
        if agreement is None or agreement<0.65:reasons.append('LOW_MODEL_AGREEMENT')
        if volatility is not None and alpha is not None and alpha<volatility:reasons.append('ALPHA_BELOW_HISTORICAL_SIGNAL_DISPERSION')
        # A passing heuristic is still not an investment recommendation.
        status='RESEARCH_CANDIDATE' if not reasons else 'ABSTAIN'
        records.append({'ticker':ticker,'signal_date':latest,'predicted_20d_excess_return':alpha,
            'decision':status,'reasons':reasons,'quality_gate':quality,'reward_risk':rr,
            'stop_distance_pct':stop,'prior_signal_observations':historical_count,
            'prior_positive_signal_fraction':stability,'prior_model_sign_agreement':agreement,
            'prior_signal_std':volatility})
    records.sort(key=lambda r:(r['decision']!='RESEARCH_CANDIDATE',-(r['predicted_20d_excess_return'] or -999)))
    audit_stats={'historical_records':int(len(audit)),'completed_records':0,'mean_completed_net_return':None}
    if 'outcome' in audit and 'net_return' in audit:
        completed=audit[audit.outcome.isin(['STOP_CLOSE','TARGET_CLOSE','TIME_EXIT'])]
        values=pd.to_numeric(completed.net_return,errors='coerce').dropna()
        audit_stats.update(completed_records=int(len(values)),mean_completed_net_return=finite(values.mean()) if len(values) else None)
    return {'version':'14.0','as_of':latest,'source':'SAVED LOCAL RESEARCH ARTIFACTS',
      'candidate_count':sum(r['decision']=='RESEARCH_CANDIDATE' for r in records),
      'abstain_count':sum(r['decision']=='ABSTAIN' for r in records),
      'audit':audit_stats,'rows':records,
      'limitations':['Not a calibrated probability or validated trading recommendation',
        'Historical model sign persistence is not predictive accuracy',
        'Prior model outputs may overlap in time and are not independent',
        'Risk gates are conservative heuristics, not optimized thresholds',
        'No current market regime, fundamentals, live prices, transaction execution, or fresh model retraining',
        'Historical fills are approximate and subject to selection and survivorship bias']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[1]/'data');args=p.parse_args()
    result=build_report(args.data_root)
    target=args.data_root/'recommendations'/'upgrade14_decisions.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(f"Upgrade 14: {len(result['rows'])} reviewed, {result['candidate_count']} research candidates, {result['abstain_count']} abstentions")
    print(f'Saved: {target}')
if __name__=='__main__':main()
