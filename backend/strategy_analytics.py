"""VertexCapital Upgrade 7: descriptive analytics for Strategy v2 trade audit.
Not a portfolio backtest: trades may overlap and capital allocation is unspecified.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

SOURCE = Path('data/recommendations/strategy_v2/trade_audit.csv')
DEST = Path('data/recommendations/strategy_v2/analytics')
VALID_OUTCOMES = {'STOP', 'STOP_GAP', 'TARGET', 'TARGET_GAP', 'TIME_EXIT'}


def summarize(returns):
    r = pd.to_numeric(pd.Series(returns), errors='coerce').dropna().astype(float)
    r = r[np.isfinite(r)]
    if r.empty:
        return {'trades': 0, 'win_rate': None, 'average_return': None, 'median_return': None,
                'average_win': None, 'average_loss': None, 'profit_factor': None,
                'payoff_ratio': None, 'best_trade': None, 'worst_trade': None}
    wins = r[r > 0]; losses = r[r < 0]
    return {'trades': int(len(r)), 'win_rate': float((r > 0).mean()),
            'average_return': float(r.mean()), 'median_return': float(r.median()),
            'average_win': float(wins.mean()) if len(wins) else None,
            'average_loss': float(losses.mean()) if len(losses) else None,
            'profit_factor': float(wins.sum() / -losses.sum()) if len(losses) and -losses.sum() > 0 else None,
            'payoff_ratio': float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
            'best_trade': float(r.max()), 'worst_trade': float(r.min())}


def analyze(frame):
    required = {'net_return', 'outcome', 'ticker', 'signal_date'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError('Missing required audit columns: ' + ', '.join(sorted(missing)))
    df = frame.copy()
    df['net_return'] = pd.to_numeric(df['net_return'], errors='coerce')
    df['signal_date'] = pd.to_datetime(df['signal_date'], errors='coerce')
    df = df[df['outcome'].isin(VALID_OUTCOMES) & df['net_return'].notna() & np.isfinite(df['net_return'])].copy()
    df = df.sort_values(['signal_date', 'ticker'], kind='stable').reset_index(drop=True)
    report = summarize(df['net_return'])
    report['excluded_or_incomplete'] = int(len(frame) - len(df))
    report['unique_signal_dates'] = int(df['signal_date'].nunique())
    report['unique_tickers'] = int(df['ticker'].nunique())
    report['outcomes'] = {str(k): int(v) for k, v in df['outcome'].value_counts().items()}
    # NOT a portfolio equity curve. This assumes trades execute sequentially with 100% reinvestment,
    # which is impossible when trades overlap. Only a descriptive illustration.
    report['portfolio_equity_curve_available'] = False
    report['portfolio_max_drawdown'] = None
    report['spy_comparison_available'] = False
    report['sector_analysis_available'] = False
    report['warning'] = ('Exploratory trade-level statistics only. Overlapping positions, position sizing, '
                         'capital constraints, benchmark matching, prediction provenance and survivorship '
                         'have not been audited. Do not interpret as portfolio performance or live results.')
    by_ticker = df.groupby('ticker', sort=True)['net_return'].agg(['count', 'mean', 'median']).reset_index()
    by_outcome = df.groupby('outcome', sort=True)['net_return'].agg(['count', 'mean']).reset_index()
    # Strategy v2 charges 10bps each way. Sensitivity is a hypothetical change in round-trip bps,
    # not an execution model or spread/market impact simulation.
    cost = []
    for bps in [0, 10, 20, 50, 100]:
        adjusted = df['net_return'] + (20 - 2 * bps) / 10000
        stats = summarize(adjusted)
        cost.append({'one_way_cost_bps': bps, 'average_net_return': stats['average_return'],
                     'positive_fraction': stats['win_rate']})
    return report, by_ticker, by_outcome, pd.DataFrame(cost), df


def main():
    if not SOURCE.exists():
        raise SystemExit('Missing trade audit. First run: python backend\\strategy_research_v2.py')
    report, ticker, outcome, cost, clean = analyze(pd.read_csv(SOURCE))
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'trade_statistics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    ticker.to_csv(DEST / 'by_ticker.csv', index=False)
    outcome.to_csv(DEST / 'by_outcome.csv', index=False)
    cost.to_csv(DEST / 'cost_sensitivity.csv', index=False)
    print('VERTEXCAPITAL UPGRADE 7 - TRADE ANALYTICS')
    print('Completed trades:', report['trades'])
    for key in ('win_rate', 'average_return', 'average_win', 'average_loss', 'profit_factor', 'payoff_ratio'):
        print(f'{key}: {report[key]}')
    print('Unique signal dates:', report['unique_signal_dates'])
    print('Portfolio drawdown / SPY comparison: NOT AVAILABLE without a portfolio-level simulation')
    print('Saved reports to:', DEST)
    print('WARNING:', report['warning'])


if __name__ == '__main__':
    main()
