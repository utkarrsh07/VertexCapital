import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from upgrade14_engine import build_report

def test_abstains_when_no_history(tmp_path):
    root=tmp_path/'data';(root/'recommendations').mkdir(parents=True)
    pd.DataFrame([{'date':'2026-09-02','ticker':'TEST','predicted_20d_excess_return':.05}]).to_csv(root/'recommendations/latest_rankings.csv',index=False)
    report=build_report(root)
    assert report['candidate_count']==0
    assert report['rows'][0]['decision']=='ABSTAIN'
    assert 'INSUFFICIENT_MODEL_HISTORY' in report['rows'][0]['reasons']

def test_duplicate_tickers_rejected(tmp_path):
    root=tmp_path/'data';(root/'recommendations').mkdir(parents=True)
    pd.DataFrame([{'ticker':'A','predicted_20d_excess_return':.1}]*2).to_csv(root/'recommendations/latest_rankings.csv',index=False)
    with pytest.raises(ValueError,match='Duplicate'):
        build_report(root)
