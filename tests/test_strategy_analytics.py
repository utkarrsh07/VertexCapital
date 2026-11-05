import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from strategy_analytics import analyze, summarize


def test_summary():
    s = summarize([.1, -.05, .02])
    assert s['trades'] == 3
    assert s['profit_factor'] == pytest.approx(2.4)
    assert s['win_rate'] == pytest.approx(2/3)


def test_empty():
    assert summarize([])['win_rate'] is None


def test_filter_and_costs():
    f = pd.DataFrame({'net_return':[.04, -.02, None], 'outcome':['TARGET','STOP','INCOMPLETE'],
                      'ticker':['AAA','BBB','CCC'], 'signal_date':['2025-01-01']*3})
    summary, tickers, outcomes, costs, clean = analyze(f)
    assert summary['trades'] == 2
    assert summary['excluded_or_incomplete'] == 1
    assert costs.loc[costs.one_way_cost_bps == 10, 'average_net_return'].iloc[0] == pytest.approx(.01)
    assert not summary['portfolio_equity_curve_available']


def test_missing_columns():
    with pytest.raises(ValueError, match='Missing required'):
        analyze(pd.DataFrame({'net_return':[.1]}))
