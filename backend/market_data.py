from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "AMD",
    "JPM",
    "RY.TO",
    "TD.TO",
    "XEQT.TO",
    "SPY",
]

DATA_DIR = Path("data")


def download_market_data(
    tickers=None,
    period="10y",
) -> dict[str, pd.DataFrame]:
    tickers = tickers or DEFAULT_TICKERS

    raw = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no market data.")

    market_data = {}

    for field in ["Open", "High", "Low", "Close", "Volume"]:
        if isinstance(raw.columns, pd.MultiIndex):
            if field not in raw.columns.get_level_values(0):
                continue
            frame = raw[field].copy()
        else:
            frame = raw[[field]].copy()
            frame.columns = [tickers[0]]

        frame = frame.sort_index()
        frame = frame.replace([float("inf"), float("-inf")], pd.NA)

        if field != "Volume":
            frame = frame.ffill(limit=3)

        market_data[field.lower()] = frame

    return market_data


def download_prices(tickers=None, period="10y") -> pd.DataFrame:
    return download_market_data(tickers, period)["close"]


def save_market_data(data: dict[str, pd.DataFrame]) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    for name, frame in data.items():
        frame.to_csv(DATA_DIR / f"{name}.csv")


if __name__ == "__main__":
    data = download_market_data()
    save_market_data(data)

    close = data["close"]

    print("\nVERTEXCAPITAL MARKET DATA")
    print("=" * 45)
    print(f"First date:       {close.index.min().date()}")
    print(f"Latest date:      {close.index.max().date()}")
    print(f"Trading days:     {len(close):,}")
    print(f"Assets:           {close.shape[1]}")
    print("\nLatest adjusted prices:")
    print(close.iloc[-1].dropna().round(2).sort_values(ascending=False))