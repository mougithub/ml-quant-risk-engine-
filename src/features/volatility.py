"""
Volatility feature calculations.
"""

from typing import Union
import numpy as np
import pandas as pd


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility.
    """

    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def calculate_ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate EWMA volatility using RiskMetrics methodology.
    """

    variance = np.zeros(len(returns))

    variance[0] = returns.var()

    for t in range(1, len(returns)):
        variance[t] = (
            lambda_ * variance[t - 1]
            + (1 - lambda_) * returns.iloc[t] ** 2
        )

    volatility = np.sqrt(variance)

    series = pd.Series(volatility, index=returns.index)

    if annualize:
        series = series * np.sqrt(252)

    return series


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Realized volatility estimate.
    """

    return np.sqrt(
        (returns ** 2).rolling(window=window).sum()
    ) * np.sqrt(252)