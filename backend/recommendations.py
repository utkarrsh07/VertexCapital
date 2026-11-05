"""VertexCapital research-only, end-of-day recommendation proposals.

Price levels are volatility-based heuristics, NOT trained price forecasts.
Do not use as live trading instructions without independent validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factors import RAW_FACTOR_COLUMNS, build_factor_panel
from market_data import download_market_data
from walk_forward import make_models
from market_data import SECTOR_MAP


def fit_current_predictions(close: pd.DataFrame, volume: pd.DataFrame,
                            benchmark: str = 'SPY', horizon: int = 20,
                            train_days: int = 756) -> pd.DataFrame:
    """Train on matured labels only; score the most recent feature date."""
    panel = build_factor_panel(close, volume, benchmark=benchmark, horizon=horizon)
    dates = pd.DatetimeIndex(sorted(panel.index.get_level_values('date').unique()))
    latest = dates[-1]
    latest_features = panel.xs(latest, level='date').dropna(subset=RAW_FACTOR_COLUMNS)
    if latest_features.empty:
        raise ValueError('No complete factor rows on latest date.')
    # Label at date t requires the price at t+horizon. Exclude immature labels.
    mature_cutoff = len(dates) - 1 - horizon
    if mature_cutoff < train_days:
        raise ValueError('Insufficient matured historical dates for training.')
    eligible_dates = dates[max(0, mature_cutoff - train_days + 1):mature_cutoff + 1]
    training = panel.loc[panel.index.get_level_values('date').isin(eligible_dates)]
    training = training.dropna(subset=RAW_FACTOR_COLUMNS + ['target_20d_excess'])
    if len(training) < 500:
        raise ValueError('Insufficient labeled training rows.')
    linear, nonlinear = make_models()
    linear.fit(training[RAW_FACTOR_COLUMNS], training['target_20d_excess'])
    nonlinear.fit(training[RAW_FACTOR_COLUMNS], training['target_20d_excess'])
    predictions = .4 * linear.predict(latest_features[RAW_FACTOR_COLUMNS]) + .6 * nonlinear.predict(latest_features[RAW_FACTOR_COLUMNS])
    return pd.DataFrame({'date': latest, 'ticker': latest_features.index,
                         'predicted_20d_excess_return': predictions})


def make_trade_plans(predictions: pd.DataFrame, close: pd.DataFrame,
                     horizon: int = 20, top_n: int = 10) -> pd.DataFrame:
    """Convert relative-return rankings into conditional research scenarios.

    Uses 20-day realized volatility, not learned entry/exit probabilities.
    Time exit is horizon trading sessions after signal; not a guaranteed fill date.
    """
    close = close.sort_index()
    returns = close.pct_change(fill_method=None)
    vol = returns.rolling(63, min_periods=40).std().iloc[-1]
    latest_prices = close.iloc[-1]
    date = pd.Timestamp(close.index[-1])
    rows = []
    ranked = predictions.sort_values('predicted_20d_excess_return', ascending=False)
    for rec in ranked.itertuples(index=False):
        ticker = rec.ticker
        if ticker not in close.columns:
            continue
        price = float(latest_prices[ticker])
        daily_vol = float(vol[ticker])
        # Near-perfect synthetic trends can produce almost zero measured volatility.
        # Apply a conservative floor to prevent rounded entry/stop/target collisions.
        if np.isfinite(daily_vol):
            daily_vol = max(daily_vol, 0.005)
        alpha = float(rec.predicted_20d_excess_return)
        if not (np.isfinite(price) and price > 0 and np.isfinite(daily_vol) and daily_vol > 0):
            continue
        # Only positive relative-alpha names qualify for long research proposals.
        if alpha <= 0:
            continue
        entry = price * (1 - min(daily_vol * .5, .05))
        stop = entry * (1 - min(daily_vol * np.sqrt(horizon) * 1.25, .25))
        # Historical 20-session return distribution, using only matured observations.
        historical = close[ticker].dropna()
        realized = (historical.shift(-horizon) / historical - 1).dropna().tail(756)
        if len(realized) < 252:
            continue
        upside = float(realized.quantile(0.60))
        downside = float(realized.quantile(0.10))
        # Exclude extreme/unfavorable scenarios rather than manufacture a target.
        if upside <= 0 or upside > 0.15:
            continue
        target = entry * (1 + upside)
        stop = min(stop, entry * (1 + min(downside, -0.005)))
        rows.append({'signal_date': date.date().isoformat(), 'ticker': ticker,
                     'reference_close': round(price, 4),
                     'predicted_20d_excess_return': round(alpha, 6),
                     'suggested_limit_entry': round(entry, 4),
                     'illustrative_stop': round(stop, 4),
                     'illustrative_target': round(target, 4),
                     'reward_risk_ratio': round((target-entry)/(entry-stop), 3),
                     'sector': SECTOR_MAP.get(ticker, 'Unclassified'),
                     'historical_20d_return_p60': round(upside, 6),
                     'historical_samples': len(realized),
                     'entry_valid_through': (date + pd.offsets.BDay(3)).date().isoformat(),
                     'time_exit_no_later_than': (date + pd.offsets.BDay(horizon + 3)).date().isoformat(),
                     'status': 'RESEARCH_ONLY_NOT_VALIDATED',
                     'note': 'Relative-alpha model; entry/stop/target are volatility heuristics, not calibrated forecasts.'})
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', default='10y')
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    print('Downloading market data and computing latest features...')
    market = download_market_data(period=args.period)
    predictions = fit_current_predictions(market['close'], market['volume'])
    plans = make_trade_plans(predictions, market['close'], top_n=args.top)
    output = Path('data/recommendations')
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'latest_rankings.csv', index=False)
    plans.to_csv(output / 'research_trade_plans.csv', index=False)
    (output / 'README.txt').write_text(
        'RESEARCH ONLY. Predicted values are 20-day excess returns vs SPY, not absolute returns.\n'
        'Entry, stop, target and dates are unvalidated volatility-based scenarios.\n'
        'No probabilities, confidence scores or live execution implied.\n', encoding='utf-8')
    print(f'Latest signal date: {predictions.date.iloc[0].date()}')
    print(f'Scored stocks: {len(predictions)}; conditional long scenarios: {len(plans)}')
    print('Saved outputs in data/recommendations/')
    if not plans.empty:
        print(plans[['ticker','suggested_limit_entry','illustrative_target','illustrative_stop','time_exit_no_later_than']].to_string(index=False))
    print('RESEARCH ONLY â€” not validated for real-money execution.')


if __name__ == '__main__':
    main()
