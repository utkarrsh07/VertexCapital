import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import numpy as np
import pandas as pd
import pytest
from research_validation_v16 import validate_predictions, aligned_labels, predictive_skill, stress_test, portfolio_audit

def sample_predictions():
    return pd.DataFrame({'date':['2025-01-02']*6,'ticker':list('ABCDEF'),'prediction':np.arange(6)/100,'linear_prediction':np.arange(6)/100,'nonlinear_prediction':np.arange(6)/100})

def test_predictions_preserve_all_rows():
    valid,latest,report=validate_predictions(sample_predictions())
    assert len(valid)==6 and report['latest_universe']==6
    assert latest.model_agreement.all()

def test_duplicate_labels_rejected():
    p,_,_=validate_predictions(sample_predictions())
    labels=pd.DataFrame({'date':['2025-01-02']*2,'ticker':['A','A'],'realized_20d_excess':[.1,.2],'label_end_date':['2025-02-01']*2})
    with pytest.raises(ValueError,match='Duplicate'):aligned_labels(p,labels)

def test_missing_end_date_rejected():
    p,_,_=validate_predictions(sample_predictions())
    labels=pd.DataFrame({'date':['2025-01-02'],'ticker':['A'],'realized_20d_excess':[.1]})
    with pytest.raises(ValueError,match='label_end_date'):aligned_labels(p,labels)

def test_small_sample_abstains():
    p,_,_=validate_predictions(sample_predictions())
    p['realized_20d_excess']=np.arange(6)/100
    assert predictive_skill(p)['status']=='INSUFFICIENT_INDEPENDENT_SIGNAL_DATES'

def test_cost_sensitivity_monotone():
    dates=pd.bdate_range('2024-01-01',periods=100)
    d=pd.DataFrame({'Date':dates,'net_return':np.full(100,.001),'benchmark_return':np.full(100,.0008),'trading_cost':np.full(100,.0001)})
    report=stress_test(d)
    assert report['cost_sensitivity']['5x_recorded_cost']['cagr'] < report['cost_sensitivity']['1x_recorded_cost']['cagr']

def test_portfolio_concentration():
    w=pd.DataFrame({'date':['2025-01-01']*2,'ticker':['A','B'],'weight':[.5,.5]})
    assert portfolio_audit(w)['effective_number_positions']==pytest.approx(2)
