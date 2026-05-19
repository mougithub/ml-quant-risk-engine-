"""
Drawdown calculations.
"""

import pandas as pd


def calculate_drawdown(
    prices: pd.Series
) -> pd.Series:
    """
    Calculate drawdown series.
    """

    running_max = prices.cummax()

    drawdown = (
        prices - running_max
    ) / running_max

    return drawdown


def maximum_drawdown(
    prices: pd.Series
) -> float:
    """
    Maximum drawdown.
    """

    dd = calculate_drawdown(prices)

    return dd.min()


def drawdown_duration(
    drawdown: pd.Series
) -> int:
    """
    Longest drawdown duration.
    """

    duration = 0
    max_duration = 0

    for value in drawdown:

        if value < 0:
            duration += 1
            max_duration = max(
                max_duration,
                duration
            )
        else:
            duration = 0

    return max_duration