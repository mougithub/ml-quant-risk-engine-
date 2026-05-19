import numpy as np
import pandas as pd

from src.features.volatility import (
    calculate_rolling_volatility,
    calculate_ewma_volatility
)


def generate_returns():

    np.random.seed(42)

    return pd.Series(
        np.random.normal(0, 0.01, 252)
    )


def test_rolling_volatility():

    returns = generate_returns()

    vol = calculate_rolling_volatility(
        returns,
        window=20
    )

    assert len(vol) == len(returns)

    assert vol.isna().sum() == 19


def test_ewma_volatility():

    returns = generate_returns()

    ewma = calculate_ewma_volatility(
        returns
    )

    assert len(ewma) == len(returns)

    assert ewma.isna().sum() == 0

    assert (ewma > 0).all()