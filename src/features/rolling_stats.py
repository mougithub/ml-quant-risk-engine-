"""
Rolling statistical features.
"""

import pandas as pd


def rolling_mean(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:

    return returns.rolling(window=window).mean()


def rolling_skewness(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:

    return returns.rolling(window=window).skew()


def rolling_kurtosis(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:

    return returns.rolling(window=window).kurt()