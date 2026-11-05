"""Point-in-time, OHLC-based trade-plan research. Not a validated live strategy."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def build_plan(prices: pd.DataFrame, signal_date, ticker: str, alpha: float,
               lookback=252, horizon=20):
    """Only bars at or before signal_date may influence a plan."""
    if ticker not in prices.columns or not isinstance(prices.index, pd.DatetimeIndex):
        return None
    history = prices.loc[prices.index <= pd.Timestamp(signal_date), ticker].dropna()
    if len(history) < lookback + horizon + 1:
        return None
    history = history.iloc[-(lookback + horizon + 1):]
    last = float(history.iloc[-1])
    if not np.isfinite(last) or last <= 0 or not np.isfinite(alpha):
        return None
    # Non-overlapping forward horizon observations, all resolved by signal date.
    observations = []
    for end in range(horizon, len(history), horizon):
        r = float(history.iloc[end] / history.iloc[end-horizon] - 1)
        if np.isfinite(r):
            observations.append(r)
    if len(observations) < 10:
        return None
    downside, median, upside = np.quantile(observations, [0.20, 0.50, 0.80])
    # A positive relative-to-SPY alpha is NOT an absolute positive return forecast.
    # Only evaluate positive historical absolute-return setups; do not claim confidence.
    if alpha <= 0 or median <= 0 or upside <= 0:
        return None
    daily = history.pct_change(fill_method=None).dropna()
    vol = float(daily.iloc[-63:].std())
    if not np.isfinite(vol) or vol <= 0:
        return None
    risk = min(0.08, max(0.025, 1.5 * vol * np.sqrt(horizon)))
    reward = min(0.15, max(0, float(upside)))
    if reward / risk < 1.0:
        return None
    return dict(signal_date=str(pd.Timestamp(signal_date).date()), ticker=ticker,
                signal_close=last, entry_reference=last, stop_reference=last*(1-risk),
                target_reference=last*(1+reward), holding_sessions=horizon,
                reward_risk=reward/risk, historical_20d_median=float(median),
                historical_20d_p80=float(upside), historical_20d_p20=float(downside),
                predicted_excess_return=float(alpha), status='RESEARCH_ONLY')


def simulate_plan(plan, open_prices, high_prices, low_prices, close_prices, cost_bps=10):
    """Next-session open entry; OHLC barrier exits, pessimistic on same-bar collisions."""
    ticker = plan['ticker']; date = pd.Timestamp(plan['signal_date'])
    frames = [x[ticker] for x in (open_prices, high_prices, low_prices, close_prices)]
    bars = pd.concat(frames, axis=1, keys=['open','high','low','close']).sort_index()
    bars = bars.loc[bars.index > date].iloc[:int(plan['holding_sessions'])]
    if len(bars) < int(plan['holding_sessions']) or bars.isna().any().any():
        return dict(outcome='INCOMPLETE', net_return=np.nan)
    entry = float(bars.iloc[0]['open'])
    if entry <= 0 or not np.isfinite(entry):
        return dict(outcome='INVALID_ENTRY', net_return=np.nan)
    # Rebase barriers to actual entry; signal-close references are indicative only.
    stop_fraction = 1 - plan['stop_reference']/plan['entry_reference']
    target_fraction = plan['target_reference']/plan['entry_reference'] - 1
    stop = entry * (1-stop_fraction); target = entry * (1+target_fraction)
    outcome = 'TIME_EXIT'; exit_price = float(bars.iloc[-1]['close']); exit_date = bars.index[-1]
    for day, row in bars.iterrows():
        if row['open'] <= stop:
            outcome, exit_price, exit_date = 'STOP_GAP', float(row['open']), day; break
        if row['open'] >= target:
            outcome, exit_price, exit_date = 'TARGET_GAP', float(row['open']), day; break
        if row['low'] <= stop:
            outcome, exit_price, exit_date = 'STOP', stop, day; break
        if row['high'] >= target:
            outcome, exit_price, exit_date = 'TARGET', target, day; break
    return dict(outcome=outcome, entry_date=str(bars.index[0].date()),
                exit_date=str(exit_date.date()), entry_price=entry, exit_price=exit_price,
                net_return=exit_price/entry-1-2*cost_bps/10000)


def evaluate(predictions, market, max_dates=24, top_n=10):
    close = market['close']; records = []
    predictions = predictions.copy()
    predictions['date'] = pd.to_datetime(predictions['date'])
    pred_col = 'prediction' if 'prediction' in predictions else 'predicted_20d_excess_return'
    dates = sorted(predictions['date'].dropna().unique())[-max_dates:]
    for date in dates:
        subset = predictions.loc[predictions.date == date].sort_values(pred_col, ascending=False).head(top_n)
        for row in subset.itertuples(index=False):
            plan = build_plan(close, date, row.ticker, float(getattr(row, pred_col)))
            if plan is None:
                continue
            result = simulate_plan(plan, market['open'], market['high'], market['low'], close)
            records.append({**plan, **result})
    return pd.DataFrame(records)


def main():
    from market_data import download_market_data
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-dates', type=int, default=24)
    args = parser.parse_args()
    path = Path('data/walk_forward_predictions.csv')
    if not path.exists():
        raise SystemExit('No saved walk-forward predictions. Run the original walk-forward pipeline first.')
    print('Downloading OHLC data; evaluating point-in-time plans...')
    market = download_market_data(period='10y')
    result = evaluate(pd.read_csv(path), market, max_dates=args.max_dates)
    output = Path('data/recommendations/strategy_v2'); output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output/'trade_audit.csv', index=False)
    completed = result.loc[result.outcome.isin(['STOP','STOP_GAP','TARGET','TARGET_GAP','TIME_EXIT'])] if not result.empty else result
    summary = dict(plans=int(len(result)), completed=int(len(completed)),
                   mean_net_return=float(completed.net_return.mean()) if len(completed) else None,
                   positive_fraction=float((completed.net_return > 0).mean()) if len(completed) else None,
                   warning='Exploratory only: saved prediction provenance, universe survivorship, overlapping trades, execution assumptions and model calibration not independently audited.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('STRATEGY V2 RESEARCH:', summary)
    print('Saved:', output)

if __name__ == '__main__':
    main()
