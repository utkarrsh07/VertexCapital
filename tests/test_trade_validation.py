import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from trade_validation import quality_filter, simulate_plan, summarize

def test_quality_rejects_bad_reward_risk():
    x=pd.DataFrame([dict(suggested_limit_entry=100,illustrative_stop=80,illustrative_target=103)])
    result=quality_filter(x)
    assert result.iloc[0].quality_status=='REJECT'

def test_quality_accepts_bounded_risk():
    x=pd.DataFrame([dict(suggested_limit_entry=100,illustrative_stop=96,illustrative_target=108)])
    assert quality_filter(x).iloc[0].quality_status=='PASS'

def test_close_only_no_fill():
    dates=pd.bdate_range('2026-01-01',periods=30)
    prices=pd.DataFrame({'AAA':[100.0]*30},index=dates)
    plan=dict(signal_date=str(dates[0].date()),ticker='AAA',entry_valid_through=str(dates[3].date()),time_exit_no_later_than=str(dates[22].date()),suggested_limit_entry=90,illustrative_stop=85,illustrative_target=105)
    assert simulate_plan(plan,prices)['outcome']=='NO_FILL'

def test_summary_empty():
    assert summarize(pd.DataFrame())['completed_fills']==0
