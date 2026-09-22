import numpy as np
import pandas as pd

from market_data import download_market_data


RAW_FACTOR_COLUMNS = [
    "momentum_12_1",
    "momentum_6m",
    "momentum_3m",
    "short_reversal",
    "trend_50",
    "trend_200",
    "low_volatility",
    "downside_risk",
    "residual_momentum",
    "distance_52w_high",
    "rsi_signal",
    "volume_momentum",
]


def calculate_rsi(prices: pd.DataFrame, window=14) -> pd.DataFrame:
    changes = prices.diff()

    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / window,
        min_periods=window,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / window,
        min_periods=window,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    return 100 - (100 / (1 + relative_strength))


def rolling_beta(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window=126,
) -> pd.DataFrame:
    market_variance = market_returns.rolling(window).var()
    betas = pd.DataFrame(index=returns.index, columns=returns.columns)

    for ticker in returns.columns:
        covariance = returns[ticker].rolling(window).cov(market_returns)
        betas[ticker] = covariance / market_variance

    return betas.astype(float)


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True) - 0.5


def build_factor_panel(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark="SPY",
    horizon=20,
) -> pd.DataFrame:
    close = close.sort_index().ffill(limit=3)
    volume = volume.reindex(
    index=close.index,
    columns=close.columns,
).astype("float64")

    returns = close.pct_change(fill_method=None)
    market_returns = returns[benchmark]

    beta = rolling_beta(returns, market_returns)

    residual_returns = returns.subtract(
        beta.mul(market_returns, axis=0),
        axis=0,
    )

    residual_momentum = (
        (1 + residual_returns)
        .rolling(126)
        .apply(np.prod, raw=True)
        - 1
    )

    volatility_63 = returns.rolling(63).std() * np.sqrt(252)

    downside_returns = returns.where(returns < 0, 0)
    downside_volatility = (
        downside_returns.rolling(63).std() * np.sqrt(252)
    )

    rsi = calculate_rsi(close)

    factors = {
        "momentum_12_1": close.shift(21) / close.shift(252) - 1,
        "momentum_6m": close / close.shift(126) - 1,
        "momentum_3m": close / close.shift(63) - 1,
        "short_reversal": -(close / close.shift(5) - 1),
        "trend_50": close / close.rolling(50).mean() - 1,
        "trend_200": close / close.rolling(200).mean() - 1,
        "low_volatility": -volatility_63,
        "downside_risk": -downside_volatility,
        "residual_momentum": residual_momentum,
        "distance_52w_high": close / close.rolling(252).max() - 1,
        "rsi_signal": -(rsi - 50).abs(),
        "volume_momentum": (
    volume.rolling(
        window=20,
        min_periods=10,
    ).mean()
    / volume.rolling(
        window=126,
        min_periods=60,
    ).mean()
    - 1
),
    }

    future_returns = close.shift(-horizon) / close - 1
    benchmark_future = future_returns[benchmark]

    excess_target = future_returns.subtract(
        benchmark_future,
        axis=0,
    )

    panel_parts = {}

    for factor_name, factor_values in factors.items():
        ranked_values = cross_sectional_rank(factor_values)
        panel_parts[factor_name] = ranked_values.stack(
            future_stack=True
        )

    panel = pd.concat(panel_parts, axis=1)
    panel["target_20d_excess"] = excess_target.stack(
        future_stack=True
    )

    panel.index.names = ["date", "ticker"]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.sort_index()

    return panel


if __name__ == "__main__":
    market = download_market_data(period="10y")

    factor_panel = build_factor_panel(
        close=market["close"],
        volume=market["volume"],
    )

    latest_date = factor_panel.dropna(
        subset=RAW_FACTOR_COLUMNS
    ).index.get_level_values("date").max()

    print("\nVERTEXCAPITAL FACTOR SNAPSHOT")
    print("=" * 55)
    print(f"Date: {latest_date.date()}\n")
    print(
        factor_panel.loc[latest_date, RAW_FACTOR_COLUMNS]
        .round(3)
        .sort_values("momentum_12_1", ascending=False)
    )