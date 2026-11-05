import unittest
import numpy as np
import pandas as pd
from quant_research_v15 import paired_block_bootstrap,audit_predictions,audit_backtest
class ResearchTests(unittest.TestCase):
 def test_short_sample_not_validated(self):self.assertEqual(paired_block_bootstrap([.01]*20)['status'],'INSUFFICIENT_OBSERVATIONS')
 def test_deterministic_bootstrap(self):
  x=np.linspace(-.01,.01,100);self.assertEqual(paired_block_bootstrap(x),paired_block_bootstrap(x))
 def test_disagreement_and_unvalidated(self):
  d=pd.DataFrame([{'date':'2026-01-01','ticker':'ABC','prediction':.02,'linear_prediction':.03,'nonlinear_prediction':-.01}]);r=audit_predictions(d)
  self.assertFalse(r['rankings'][0]['direction_agreement']);self.assertEqual(r['validation']['status'],'NOT_IDENTIFIABLE_FROM_PREDICTIONS_ALONE')
 def test_backtest_rejects_missing_columns(self):
  with self.assertRaises(ValueError):audit_backtest(pd.DataFrame({'x':[1]}))
if __name__=='__main__':unittest.main()
