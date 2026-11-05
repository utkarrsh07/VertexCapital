"""Close-only, historical research audit. No intraday fill or calibrated probabilities."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from recommendations import make_trade_plans


def quality_filter(plans, min_reward_risk=1.0, max_stop_pct=0.12):
    if plans.empty:
        return plans.copy()
    result = plans.copy()
    entry = pd.to_numeric(result['suggested_limit_entry'])
    stop = pd.to_numeric(result['illustrative_stop'])
    target = pd.to_numeric(result['illustrative_target'])
    risk = (entry-stop)/entry
    reward = (target-entry)/entry
    result['stop_distance_pct'] = risk
    result['target_upside_pct'] = reward
    result['quality_reason'] = np.where(risk > max_stop_pct, 'STOP_TOO_WIDE',
        np.where(reward/risk < min_reward_risk, 'LOW_REWARD_RISK', 'PASS'))
    result['quality_status'] = np.where(result['quality_reason']=='PASS', 'PASS', 'REJECT')
    return result


def simulate_plan(plan, prices, cost_bps=10):
    """Approximate close-only fill: first close <= entry in next 3 sessions.

    Exit checks subsequent closes only; gaps and intraday stop/target fills are not modeled.
    If both barriers are crossed intraday, this simulation cannot detect it.
    """
    date = pd.Timestamp(plan['signal_date'])
    ticker = plan['ticker']
    if ticker not in prices.columns:
        return {'outcome':'NO_DATA','net_return':np.nan}
    series = prices[ticker].dropna().sort_index()
    future = series.loc[series.index > date]
    expiry = pd.Timestamp(plan['entry_valid_through'])
    candidates = future.loc[future.index <= expiry]
    entry_level = float(plan['suggested_limit_entry'])
    fills = candidates[candidates <= entry_level]
    if fills.empty:
        return {'outcome':'NO_FILL','net_return':np.nan}
    fill_date = fills.index[0]
    fill_price = float(fills.iloc[0])
    deadline = pd.Timestamp(plan['time_exit_no_later_than'])
    path = future.loc[(future.index > fill_date) & (future.index <= deadline)]
    if path.empty or path.index[-1] < deadline and (series.index.max() < deadline):
        return {'outcome':'INCOMPLETE','net_return':np.nan}
    stop = float(plan['illustrative_stop'])
    target = float(plan['illustrative_target'])
    outcome = 'TIME_EXIT'
    exit_price = float(path.iloc[-1])
    exit_date = path.index[-1]
    for dt, px in path.items():
        if px <= stop:
            outcome, exit_price, exit_date = 'STOP_CLOSE', float(px), dt
            break
        if px >= target:
            outcome, exit_price, exit_date = 'TARGET_CLOSE', float(px), dt
            break
    net = exit_price/fill_price - 1 - 2*cost_bps/10000
    return {'outcome':outcome,'net_return':net,'fill_date':fill_date.date().isoformat(),
            'exit_date':exit_date.date().isoformat(),'fill_price':fill_price,'exit_price':exit_price}


def validate_historical_predictions(predictions, prices, max_dates=24, top_n=10):
    predictions = predictions.copy()
    predictions['date'] = pd.to_datetime(predictions['date'])
    dates = sorted(predictions['date'].dropna().unique())[-max_dates:]
    records=[]
    for date in dates:
        history = prices.loc[prices.index <= date]
        if len(history) < 320:
            continue
        subset = predictions.loc[predictions.date == date].copy()
        if 'predicted_20d_excess_return' not in subset and 'prediction' in subset:
            subset = subset.rename(columns={'prediction':'predicted_20d_excess_return'})
        plans = make_trade_plans(subset, history, top_n=top_n)
        if plans.empty:
            continue
        scored = quality_filter(plans)
        for _, row in scored.iterrows():
            rec=row.to_dict()
            if rec['quality_status']=='PASS':
                rec.update(simulate_plan(rec, prices))
            else:
                rec.update({'outcome':'QUALITY_REJECT','net_return':np.nan})
            records.append(rec)
    return pd.DataFrame(records)


def summarize(results):
    if results.empty:
        return {'plans':0,'quality_passed':0,'completed_fills':0,'win_rate':None,'mean_net_return':None}
    valid = results[results.outcome.isin(['STOP_CLOSE','TARGET_CLOSE','TIME_EXIT'])]
    return {'plans':int(len(results)),
            'quality_passed':int((results.quality_status=='PASS').sum()),
            'completed_fills':int(len(valid)),
            'win_rate':float((valid.net_return>0).mean()) if len(valid) else None,
            'mean_net_return':float(valid.net_return.mean()) if len(valid) else None}


def main():
    from market_data import download_market_data
    from recommendations import fit_current_predictions
    from pathlib import Path
    import json
    parser=argparse.ArgumentParser()
    parser.add_argument('--period', default='10y')
    parser.add_argument('--max-dates',type=int,default=24)
    args=parser.parse_args()
    print('Downloading market data for research audit...')
    prices=download_market_data(period=args.period)['close']
    folder=Path('data/recommendations'); folder.mkdir(parents=True,exist_ok=True)
    historical=Path('data/walk_forward_predictions.csv')
    if historical.exists():
        predictions=pd.read_csv(historical)
        print('Using saved walk-forward predictions (check their provenance and universe).')
        result=validate_historical_predictions(predictions,prices,max_dates=args.max_dates)
        result.to_csv(folder/'historical_trade_audit.csv',index=False)
        report=summarize(result)
        (folder/'historical_trade_audit_summary.json').write_text(json.dumps(report,indent=2))
        print('Historical audit:',report)
    else:
        print('No walk-forward prediction file found; skipping historical audit rather than fabricating out-of-sample signals.')
    print('Generating current quality report...')
    market=download_market_data(period=args.period)
    current=fit_current_predictions(market['close'],market['volume'])
    plans=make_trade_plans(current,market['close'],top_n=30)
    scored=quality_filter(plans)
    scored.to_csv(folder/'trade_quality_report.csv',index=False)
    print('Current scenarios:',len(scored),'quality passed:',int((scored.quality_status=='PASS').sum()) if len(scored) else 0)
    print('Research only. Close-only fills are approximate; no live trading validation.')

if __name__=='__main__':
    main()
