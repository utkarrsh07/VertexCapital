import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from quality_diagnostics import diagnose, summarize

def test_overlapping_failures():
    plans = pd.DataFrame([dict(suggested_limit_entry=100, illustrative_stop=75, illustrative_target=103)])
    scored = diagnose(plans)
    assert scored.iloc[0].fails_stop_width
    assert scored.iloc[0].fails_reward_risk
    assert scored.iloc[0].diagnostic_reasons == 'STOP_TOO_WIDE,LOW_REWARD_RISK'
    assert summarize(scored)['rejected'] == 1

def test_valid_trade():
    plans = pd.DataFrame([dict(suggested_limit_entry=100, illustrative_stop=96, illustrative_target=108)])
    assert diagnose(plans).iloc[0].diagnostic_status == 'PASS'

def test_invalid_levels():
    plans = pd.DataFrame([dict(suggested_limit_entry=100, illustrative_stop=100, illustrative_target=108)])
    assert diagnose(plans).iloc[0].diagnostic_reasons == 'INVALID_LEVELS'

def test_empty():
    scored = diagnose(pd.DataFrame(columns=['suggested_limit_entry','illustrative_stop','illustrative_target']))
    assert summarize(scored)['plans'] == 0
