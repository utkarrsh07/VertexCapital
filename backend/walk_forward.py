from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backtester import build_portfolio_weights, run_backtest
from dataset_builder import build_dataset
from factors import RAW_FACTOR_COLUMNS


RESULTS_DIR = Path("data")


def make_models():
    linear_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )

    nonlinear_model = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=42,
    )

    return linear_model, nonlinear_model


def walk_forward_predictions(
    dataset: pd.DataFrame,
    train_years=3,
    prediction_horizon=20,
    rebalance_every=20,
) -> pd.DataFrame:
    data = dataset.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "ticker"])

    dates = np.array(sorted(data["date"].unique()))

    print(f"Usable trading dates: {len(dates)}")

    minimum_history = train_years * 252

    if len(dates) <= minimum_history + prediction_horizon:
        raise ValueError(
            "Not enough history for the requested walk-forward test."
        )

    prediction_dates = dates[
        minimum_history::rebalance_every
    ]

    all_predictions = []

    for number, prediction_date in enumerate(
        prediction_dates,
        start=1,
    ):
        prediction_position = np.searchsorted(
            dates,
            prediction_date,
        )

        training_end_position = (
            prediction_position - prediction_horizon
        )

        training_start_position = max(
            0,
            training_end_position - minimum_history,
        )

        if training_end_position <= training_start_position:
            continue

        training_dates = dates[
            training_start_position:training_end_position
        ]

        train = data[data["date"].isin(training_dates)].dropna(
            subset=RAW_FACTOR_COLUMNS + ["target_20d_excess"]
        )

        test = data[
            data["date"] == prediction_date
        ].dropna(subset=RAW_FACTOR_COLUMNS)

        if len(train) < 500 or len(test) < 3:
            continue

        x_train = train[RAW_FACTOR_COLUMNS]
        y_train = train["target_20d_excess"]
        x_test = test[RAW_FACTOR_COLUMNS]

        linear_model, nonlinear_model = make_models()

        linear_model.fit(x_train, y_train)
        nonlinear_model.fit(x_train, y_train)

        linear_prediction = linear_model.predict(x_test)
        nonlinear_prediction = nonlinear_model.predict(x_test)

        # Simple ensemble reduces dependence on one model family.
        ensemble_prediction = (
            0.40 * linear_prediction
            + 0.60 * nonlinear_prediction
        )

        prediction_frame = test[["date", "ticker"]].copy()
        prediction_frame["prediction"] = ensemble_prediction
        prediction_frame["linear_prediction"] = linear_prediction
        prediction_frame["nonlinear_prediction"] = nonlinear_prediction

        all_predictions.append(prediction_frame)

        if number % 10 == 0:
            print(
                f"Completed {number}/{len(prediction_dates)} "
                "walk-forward periods"
            )

    if not all_predictions:
        raise RuntimeError("No walk-forward predictions were generated.")

    return pd.concat(all_predictions, ignore_index=True)


def print_report(report: dict) -> None:
    percentage_fields = {
        "annual_return",
        "annual_volatility",
        "maximum_drawdown",
        "daily_var_95",
        "expected_shortfall_95",
        "positive_day_rate",
    }

    print("\nVERTEXCAPITAL OUT-OF-SAMPLE REPORT")
    print("=" * 52)

    for name, value in report.items():
        readable_name = name.replace("_", " ").title()

        if name in percentage_fields:
            print(f"{readable_name:<30} {value:>10.2%}")
        elif isinstance(value, float):
            print(f"{readable_name:<30} {value:>10.3f}")
        else:
            print(f"{readable_name:<30} {value}")


def main():
    print("\nBuilding point-in-time factor dataset...")
    dataset, prices = build_dataset(period="10y")

    print("Training walk-forward ensemble...")
    predictions = walk_forward_predictions(dataset)

    print("Optimizing historical portfolios...")
    weights = build_portfolio_weights(
        predictions=predictions,
        prices=prices,
    )

    print("Running cost-adjusted backtest...")
    results, report = run_backtest(
        prices=prices,
        weights=weights,
        transaction_cost_bps=10,
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    predictions.to_csv(
        RESULTS_DIR / "walk_forward_predictions.csv",
        index=False,
    )

    weights.to_csv(
        RESULTS_DIR / "portfolio_weights.csv",
        index=False,
    )

    results.to_csv(
        RESULTS_DIR / "backtest_results.csv"
    )

    print_report(report)

    latest_weights = (
        weights[weights["date"] == weights["date"].max()]
        .sort_values("weight", ascending=False)
    )

    print("\nLATEST PAPER PORTFOLIO")
    print("=" * 52)
    print(latest_weights.to_string(index=False))

    print(
        "\nResearch output only. Do not connect this directly "
        "to a brokerage account."
    )


if __name__ == "__main__":
    main()