"""VertexCapital Upgrade 5: diagnostic-only trade quality analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ('suggested_limit_entry', 'illustrative_stop', 'illustrative_target')

def diagnose(plans: pd.DataFrame, min_reward_risk: float = 1.0, max_stop_pct: float = .12) -> pd.DataFrame:
    out = plans.copy()
    if out.empty:
        for key in ('stop_distance_pct', 'target_upside_pct', 'reward_risk_ratio_diagnostic', 'fails_stop_width', 'fails_reward_risk', 'fails_invalid_levels', 'diagnostic_status', 'diagnostic_reasons'):
            out[key] = pd.Series(dtype='object' if key in ('diagnostic_status','diagnostic_reasons') else 'float64')
        return out
    missing = set(REQUIRED) - set(out.columns)
    if missing:
        raise ValueError('Missing trade plan columns: ' + ', '.join(sorted(missing)))
    entry = pd.to_numeric(out['suggested_limit_entry'], errors='coerce')
    stop = pd.to_numeric(out['illustrative_stop'], errors='coerce')
    target = pd.to_numeric(out['illustrative_target'], errors='coerce')
    valid = (np.isfinite(entry) & np.isfinite(stop) & np.isfinite(target) & (entry > 0) & (stop > 0) & (stop < entry) & (target > entry))
    with np.errstate(divide='ignore', invalid='ignore'):
        risk = (entry - stop) / entry
        reward = (target - entry) / entry
        rr = reward / risk
    out['stop_distance_pct'] = risk.where(valid)
    out['target_upside_pct'] = reward.where(valid)
    out['reward_risk_ratio_diagnostic'] = rr.where(valid)
    out['fails_invalid_levels'] = ~valid
    out['fails_stop_width'] = valid & (risk > max_stop_pct)
    out['fails_reward_risk'] = valid & (rr < min_reward_risk)
    out['diagnostic_status'] = np.where(out[['fails_invalid_levels','fails_stop_width','fails_reward_risk']].any(axis=1), 'REJECT', 'PASS')
    out['diagnostic_reasons'] = [','.join(name for name, col in [('INVALID_LEVELS','fails_invalid_levels'),('STOP_TOO_WIDE','fails_stop_width'),('LOW_REWARD_RISK','fails_reward_risk')] if bool(row[col])) or 'PASS' for _, row in out.iterrows()]
    return out

def summarize(scored: pd.DataFrame) -> dict:
    n = len(scored)
    def stats(col):
        values = pd.to_numeric(scored[col], errors='coerce').dropna() if n else pd.Series(dtype=float)
        return {'median': float(values.median()) if len(values) else None, 'p10': float(values.quantile(.1)) if len(values) else None, 'p90': float(values.quantile(.9)) if len(values) else None}
    return {'plans': n, 'passed': int((scored['diagnostic_status'] == 'PASS').sum()) if n else 0,
            'rejected': int((scored['diagnostic_status'] == 'REJECT').sum()) if n else 0,
            'rejection_counts_overlapping': {name: int(scored[name].sum()) if n else 0 for name in ('fails_invalid_levels','fails_stop_width','fails_reward_risk')},
            'stop_distance_pct': stats('stop_distance_pct'), 'target_upside_pct': stats('target_upside_pct'), 'reward_risk_ratio': stats('reward_risk_ratio_diagnostic')}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/recommendations/historical_trade_audit.csv')
    parser.add_argument('--output', default='data/recommendations/diagnostics')
    args = parser.parse_args()
    src = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise SystemExit(f'Input not found: {src}. Run python backend/trade_validation.py first.')
    historical = diagnose(pd.read_csv(src))
    historical.to_csv(output/'historical_diagnostics.csv', index=False)
    report = {'historical': summarize(historical), 'warning': 'Historical trade plans may contain look-ahead leakage: recommendations.make_trade_plans uses return distributions without an as-of-date cutoff. This is diagnostic only, not valid out-of-sample performance.'}
    current_path = Path('data/recommendations/trade_quality_report.csv')
    if current_path.exists():
        current = diagnose(pd.read_csv(current_path))
        current.to_csv(output/'current_diagnostics.csv', index=False)
        report['current'] = summarize(current)
    (output/'quality_diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('VERTEXCAPITAL UPGRADE 5 - QUALITY DIAGNOSTICS')
    for name in ('historical','current'):
        if name in report:
            print(f'{name.upper()}: {report[name]["plans"]} plans, {report[name]["passed"]} pass, {report[name]["rejected"]} reject')
            for reason, count in report[name]['rejection_counts_overlapping'].items():
                print(f'  {reason}: {count}')
            for metric in ('stop_distance_pct','target_upside_pct','reward_risk_ratio'):
                print(f'  {metric} median: {report[name][metric]["median"]}')
    print('WARNING: Historical trade plan construction may have look-ahead leakage; do not treat results as validated returns.')
    print('Reports saved to', output)

if __name__ == '__main__':
    main()
