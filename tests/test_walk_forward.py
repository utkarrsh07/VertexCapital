import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "backend"),
)

import walk_forward  # noqa: E402
from factors import RAW_FACTOR_COLUMNS  # noqa: E402
from walk_forward import (  # noqa: E402
    make_models,
    walk_forward_predictions,
)


def make_dataset(days=700, tickers=None):
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]

    dates = pd.date_range(
        "2020-01-01",
        periods=days,
        freq="B",
    )

    records = []

    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(tickers):
            record = {
                "date": date,
                "ticker": ticker,
                "target_20d_excess": (
                    date_number * 0.0001
                    + ticker_number * 0.001
                ),
            }

            for factor_number, factor in enumerate(
                RAW_FACTOR_COLUMNS
            ):
                record[factor] = (
                    date_number
                    + ticker_number * 0.01
                    + factor_number * 0.001
                )

            records.append(record)

    return pd.DataFrame(records)


class RecordingModel:
    def __init__(self, prediction_value, fitted_frames):
        self.prediction_value = prediction_value
        self.fitted_frames = fitted_frames

    def fit(self, x, y):
        self.fitted_frames.append(x.copy())
        return self

    def predict(self, x):
        return np.full(len(x), self.prediction_value)


def install_fake_models(monkeypatch):
    fitted_frames = []

    def fake_make_models():
        return (
            RecordingModel(1.0, fitted_frames),
            RecordingModel(3.0, fitted_frames),
        )

    monkeypatch.setattr(
        walk_forward,
        "make_models",
        fake_make_models,
    )

    return fitted_frames


def test_make_models_returns_expected_model_types():
    linear_model, nonlinear_model = make_models()

    assert isinstance(linear_model, Pipeline)
    assert isinstance(
        nonlinear_model,
        HistGradientBoostingRegressor,
    )


def test_not_enough_history_is_rejected():
    dataset = make_dataset(days=270)

    with pytest.raises(
        ValueError,
        match="Not enough history",
    ):
        walk_forward_predictions(
            dataset,
            train_years=1,
            prediction_horizon=20,
        )


def test_walk_forward_generates_expected_columns(monkeypatch):
    dataset = make_dataset()
    install_fake_models(monkeypatch)

    predictions = walk_forward_predictions(
        dataset,
        train_years=1,
        prediction_horizon=20,
        rebalance_every=1000,
    )

    expected_columns = {
        "date",
        "ticker",
        "prediction",
        "linear_prediction",
        "nonlinear_prediction",
    }

    assert expected_columns.issubset(predictions.columns)


def test_ensemble_uses_40_60_weighting(monkeypatch):
    dataset = make_dataset()
    install_fake_models(monkeypatch)

    predictions = walk_forward_predictions(
        dataset,
        train_years=1,
        prediction_horizon=20,
        rebalance_every=1000,
    )

    expected_prediction = 0.40 * 1.0 + 0.60 * 3.0

    assert np.allclose(
        predictions["prediction"],
        expected_prediction,
    )


def test_training_data_respects_twenty_day_embargo(monkeypatch):
    dataset = make_dataset()
    fitted_frames = install_fake_models(monkeypatch)

    predictions = walk_forward_predictions(
        dataset,
        train_years=1,
        prediction_horizon=20,
        rebalance_every=1000,
    )

    prediction_date = predictions["date"].min()

    ordered_dates = np.array(
        sorted(pd.to_datetime(dataset["date"].unique()))
    )

    prediction_position = np.searchsorted(
        ordered_dates,
        np.datetime64(prediction_date),
    )

    latest_allowed_training_position = (
        prediction_position - 20 - 1
    )

    latest_allowed_value = latest_allowed_training_position

    for fitted_frame in fitted_frames:
        assert (
            fitted_frame["momentum_12_1"].max()
            < latest_allowed_value + 1
        )


def test_prediction_target_is_not_required_at_test_time(
    monkeypatch,
):
    dataset = make_dataset()
    install_fake_models(monkeypatch)

    dates = sorted(dataset["date"].unique())
    first_prediction_date = dates[252]

    dataset.loc[
        dataset["date"] == first_prediction_date,
        "target_20d_excess",
    ] = np.nan

    predictions = walk_forward_predictions(
        dataset,
        train_years=1,
        prediction_horizon=20,
        rebalance_every=1000,
    )

    assert len(predictions) == 3
    assert predictions["prediction"].notna().all()


def test_rebalance_schedule_is_followed(monkeypatch):
    dataset = make_dataset(days=700)
    install_fake_models(monkeypatch)

    predictions = walk_forward_predictions(
        dataset,
        train_years=1,
        prediction_horizon=20,
        rebalance_every=100,
    )

    unique_prediction_dates = predictions["date"].nunique()

    expected_dates = len(
        np.array(sorted(dataset["date"].unique()))[252::100]
    )

    assert unique_prediction_dates == expected_dates


def test_at_least_three_test_assets_are_required(monkeypatch):
    dataset = make_dataset(
        days=700,
        tickers=["AAPL", "MSFT"],
    )

    install_fake_models(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="No walk-forward predictions",
    ):
        walk_forward_predictions(
            dataset,
            train_years=1,
            prediction_horizon=20,
            rebalance_every=1000,
        )


def test_input_dataset_is_not_modified(monkeypatch):
    dataset = make_dataset()
    original = dataset.copy(deep=True)

    install_fake_models(monkeypatch)

    walk_forward_predictions(
        dataset,
        train_years=1,
        prediction_horizon=20,
        rebalance_every=1000,
    )

    pd.testing.assert_frame_equal(dataset, original)