from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'CSCO', 'IBM', 'INTC', 'QCOM', 'TXN', 'AMAT', 'MU', 'NOW', 'INTU', 'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'EA', 'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'BKNG', 'TJX', 'GM', 'F', 'WMT', 'COST', 'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KHC', 'MDLZ', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP', 'BLK', 'SCHW', 'RY.TO', 'TD.TO', 'BMO.TO', 'JNJ', 'UNH', 'ABBV', 'MRK', 'LLY', 'PFE', 'TMO', 'ABT', 'AMGN', 'GILD', 'ISRG', 'BMY', 'CAT', 'DE', 'GE', 'HON', 'UPS', 'UNP', 'RTX', 'LMT', 'BA', 'ETN', 'MMM', 'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'MPC', 'PSX', 'SU.TO', 'CNQ.TO', 'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'SRE', 'XEL', 'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'NUE', 'PLD', 'AMT', 'EQIX', 'O', 'WELL', 'SPG', 'PSA', 'SPY', 'XEQT.TO']

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
SECTOR_MAP = {'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'AVGO': 'Technology', 'ORCL': 'Technology', 'CRM': 'Technology', 'ADBE': 'Technology', 'CSCO': 'Technology', 'IBM': 'Technology', 'INTC': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology', 'AMAT': 'Technology', 'MU': 'Technology', 'NOW': 'Technology', 'INTU': 'Technology', 'GOOGL': 'Communication Services', 'META': 'Communication Services', 'NFLX': 'Communication Services', 'DIS': 'Communication Services', 'CMCSA': 'Communication Services', 'T': 'Communication Services', 'VZ': 'Communication Services', 'TMUS': 'Communication Services', 'EA': 'Communication Services', 'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary', 'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'PM': 'Consumer Staples', 'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'KHC': 'Consumer Staples', 'MDLZ': 'Consumer Staples', 'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials', 'V': 'Financials', 'MA': 'Financials', 'AXP': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials', 'RY.TO': 'Financials', 'TD.TO': 'Financials', 'BMO.TO': 'Financials', 'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'LLY': 'Healthcare', 'PFE': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'ISRG': 'Healthcare', 'BMY': 'Healthcare', 'CAT': 'Industrials', 'DE': 'Industrials', 'GE': 'Industrials', 'HON': 'Industrials', 'UPS': 'Industrials', 'UNP': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials', 'BA': 'Industrials', 'ETN': 'Industrials', 'MMM': 'Industrials', 'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy', 'OXY': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'SU.TO': 'Energy', 'CNQ.TO': 'Energy', 'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities', 'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'FCX': 'Materials', 'NEM': 'Materials', 'DOW': 'Materials', 'NUE': 'Materials', 'PLD': 'Real Estate', 'AMT': 'Real Estate', 'EQIX': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate', 'SPG': 'Real Estate', 'PSA': 'Real Estate'}
