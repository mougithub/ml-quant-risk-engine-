import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


# def compute_log_returns(df: pd.DataFrame) -> pd.Series:

#    close_prices = df["Close"]

#    return np.log(close_prices / close_prices.shift(1))


def compute_log_returns(df: pd.DataFrame) -> pd.Series:

    close_prices = df["Close"]

    # Handle MultiIndex case from yfinance
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    returns = np.log(close_prices / close_prices.shift(1))

    return returns


def combine_returns(data_dict):

    returns = {}

    for ticker, df in data_dict.items():
        returns[ticker] = compute_log_returns(df)

    combined = pd.DataFrame(returns)

    combined = combined.dropna()

    return combined


def save_processed_data(df: pd.DataFrame, filename: str):

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DIR / filename)


def load_processed_data(name: str):

    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", index_col=0, parse_dates=True)
