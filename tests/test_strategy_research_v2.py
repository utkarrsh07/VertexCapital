import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from strategy_research_v2 import build_plan, simulate_plan


def test_future_prices_do_not_change_plan():
    d = pd.bdate_range('2024-01-01', periods=450)
    px = pd.DataFrame({'AAA': 100 * (1.001 ** np.arange(450))}, index=d)
    first = build_plan(px, d[400], 'AAA', 0.03)
    modified = px.copy(); modified.loc[d[401]:, 'AAA'] *= 10
    assert first == build_plan(modified, d[400], 'AAA', 0.03)


def test_negative_alpha_is_not_long_signal():
    d = pd.bdate_range('2024-01-01', periods=450)
    px = pd.DataFrame({'AAA': 100 * (1.001 ** np.arange(450))}, index=d)
    assert build_plan(px, d[400], 'AAA', -0.02) is None


def test_stop_and_target_same_bar_assumes_stop():
    d = pd.bdate_range('2026-01-01', periods=3)
    plan = dict(ticker='AAA', signal_date='2025-12-31', holding_sessions=3,
                entry_reference=100, stop_reference=95, target_reference=105)
    op = pd.DataFrame({'AAA':[100,100,100]}, index=d)
    hi = pd.DataFrame({'AAA':[110,101,101]}, index=d)
    lo = pd.DataFrame({'AAA':[90,99,99]}, index=d)
    assert simulate_plan(plan,op,hi,lo,op)['outcome'] == 'STOP'


def test_gap_stop_uses_open():
    d = pd.bdate_range('2026-01-01', periods=3)
    plan = dict(ticker='AAA', signal_date='2025-12-31', holding_sessions=3,
                entry_reference=100, stop_reference=95, target_reference=105)
    op = pd.DataFrame({'AAA':[100,90,90]}, index=d)
    hi = pd.DataFrame({'AAA':[101,92,92]}, index=d)
    lo = pd.DataFrame({'AAA':[99,88,88]}, index=d)
    result = simulate_plan(plan,op,hi,lo,op)
    assert result['outcome'] == 'STOP_GAP'
    assert result['exit_price'] == 90
