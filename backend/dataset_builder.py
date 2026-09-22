from pathlib import Path

import pandas as pd

from factors import RAW_FACTOR_COLUMNS, build_factor_panel
from market_data import download_market_data


DATASET_PATH = Path("data/model_dataset.csv")


def build_dataset(period="10y") -> tuple[pd.DataFrame, pd.DataFrame]:
    market = download_market_data(period=period)

    panel = build_factor_panel(
        close=market["close"],
        volume=market["volume"],
    )


    dataset = panel.dropna(
        subset=RAW_FACTOR_COLUMNS + ["target_20d_excess"]
    ).copy()

    dataset = dataset.reset_index()
    dataset = dataset.sort_values(["date", "ticker"])

    return dataset, market["close"]


def save_dataset(dataset: pd.DataFrame) -> None:
    DATASET_PATH.parent.mkdir(exist_ok=True)
    dataset.to_csv(DATASET_PATH, index=False)


if __name__ == "__main__":
    dataset, _ = build_dataset()
    save_dataset(dataset)

    print("\nVERTEXCAPITAL ML DATASET")
    print("=" * 45)
    print(f"Rows:       {len(dataset):,}")
    print(f"Features:   {len(RAW_FACTOR_COLUMNS)}")
    print(f"First date: {dataset['date'].min().date()}")
    print(f"Last date:  {dataset['date'].max().date()}")
    print("\nTarget distribution:")
    print(dataset["target_20d_excess"].describe().round(4))