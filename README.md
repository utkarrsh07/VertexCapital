# VertexCapital

VertexCapital is an institutional-style quantitative investing research platform built with Python.

It downloads market data, constructs quantitative factors, trains machine-learning models using walk-forward validation, optimizes portfolios under risk constraints, and evaluates performance after transaction costs.

> This project is for research and paper trading only. It does not provide financial advice or execute real trades.

## Current Features

- Downloads up to 10 years of adjusted OHLCV market data
- Builds cross-sectional quantitative factors
- Predicts 20-day excess returns relative to SPY
- Uses Ridge regression and gradient boosting
- Applies a 20-trading-day training embargo
- Performs walk-forward out-of-sample testing
- Rebalances portfolios every 20 trading days
- Uses Ledoit-Wolf covariance shrinkage
- Applies long-only portfolio optimization
- Limits individual positions to 20%
- Maintains a 5% cash reserve
- Penalizes portfolio turnover
- Includes 10-basis-point transaction costs
- Calculates institutional-style risk metrics

## Quantitative Factors

The current model includes:

- 12–1 momentum
- Six-month momentum
- Three-month momentum
- Five-day reversal
- 50-day trend
- 200-day trend
- Low volatility
- Downside risk
- Residual momentum
- Distance from the 52-week high
- RSI signal
- Volume momentum

## Machine-Learning Process

VertexCapital uses an ensemble consisting of:

- 40% Ridge regression
- 60% HistGradientBoostingRegressor

The model is evaluated using walk-forward testing. At each prediction date, it is trained only on information that would have been available historically.

A 20-trading-day embargo separates the training data from each prediction date to reduce target leakage.

## Portfolio Construction

Expected returns from the model are passed into a CVXPY portfolio optimizer.

Current constraints include:

- Long-only positions
- Maximum 20% allocation per asset
- 5% cash reserve
- 20% target-volatility constraint
- Risk-aversion penalty
- Turnover penalty

The backtest applies predictions on the following trading day and includes estimated transaction costs.

## Risk Metrics

The risk engine calculates:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Historical Value at Risk
- Expected shortfall
- Positive-day percentage
- Market beta
- Benchmark correlation
- Portfolio turnover

## Current Research Universe

- AAPL
- MSFT
- GOOGL
- AMZN
- META
- NVDA
- AMD
- JPM
- RY.TO
- TD.TO
- XEQT.TO
- SPY

This small universe is temporary and creates survivorship-bias limitations.

## Project Structure

```text
VertexCapital/
├── backend/
│   ├── backtester.py
│   ├── dataset_builder.py
│   ├── factors.py
│   ├── market_data.py
│   ├── optimizer.py
│   ├── risk_engine.py
│   └── walk_forward.py
├── data/
├── models/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Running the Research Pipeline

Create and activate a virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the walk-forward pipeline:

```powershell
python backend/walk_forward.py
```

## Current Limitations

- Small, manually selected stock universe
- Survivorship bias
- Yahoo Finance data is not point-in-time institutional data
- No sector-exposure constraints yet
- USD and CAD currency exposure is not yet modelled
- No point-in-time fundamental data
- No live paper-trading ledger yet
- Backtest results may be affected by modelling assumptions

## Roadmap

- Add automated tests
- Build a true current-date paper-trading signal
- Compare performance against SPY and XEQT
- Add sector and factor-exposure constraints
- Model USD/CAD currency exposure
- Expand to a larger historical universe
- Reduce survivorship bias
- Add point-in-time fundamental data
- Build a FastAPI backend
- Create a professional web dashboard
- Deploy the interface on Vercel

## Disclaimer

VertexCapital is an educational software project. All outputs are simulated research results and must not be presented as real investment performance.

Backtested results do not guarantee future performance. The system is not connected to a brokerage and should not be used to trade real money without extensive additional testing and independent professional review.