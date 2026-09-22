import pandas as pd

from factors import RAW_FACTOR_COLUMNS, build_factor_panel
from market_data import download_market_data


FACTOR_WEIGHTS = {
    "momentum_12_1": 0.20,
    "momentum_6m": 0.12,
    "momentum_3m": 0.08,
    "short_reversal": 0.05,
    "trend_50": 0.08,
    "trend_200": 0.10,
    "low_volatility": 0.10,
    "downside_risk": 0.08,
    "residual_momentum": 0.10,
    "distance_52w_high": 0.04,
    "rsi_signal": 0.02,
    "volume_momentum": 0.03,
}


def calculate_composite_score(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["quant_score"] = 0.0

    for factor, weight in FACTOR_WEIGHTS.items():
        result["quant_score"] += result[factor] * weight

    result["quant_score"] = (
        result["quant_score"]
        .groupby(level="date")
        .rank(pct=True)
        * 100
    )

    return result


if __name__ == "__main__":
    market = download_market_data(period="10y")

    panel = build_factor_panel(
        market["close"],
        market["volume"],
    )

    scored = calculate_composite_score(panel)

    valid = scored.dropna(subset=RAW_FACTOR_COLUMNS)
    latest_date = valid.index.get_level_values("date").max()

    columns = RAW_FACTOR_COLUMNS + ["quant_score"]

    print("\nVERTEXCAPITAL MULTI-FACTOR RANKINGS")
    print("=" * 65)
    print(f"Date: {latest_date.date()}\n")
    print(
        scored.loc[latest_date, columns]
        .sort_values("quant_score", ascending=False)
        .round(3)
    )