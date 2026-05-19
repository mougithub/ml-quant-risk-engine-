import pandas as pd
import numpy as np

from src.data.preprocessing import compute_log_returns
from src.data.validation import detect_outliers


def test_log_returns_basic():

    df = pd.DataFrame({"Close": [100, 110]})

    returns = compute_log_returns(df)

    assert len(returns.dropna()) == 1


def test_log_returns_values():

    df = pd.DataFrame({"Close": [100, 110]})

    returns = compute_log_returns(df)

    expected = np.log(110 / 100)

    assert abs(returns.iloc[1] - expected) < 1e-6


def test_outlier_detection_obvious():

    series = pd.Series([1, 1, 1, 1, 100])

    outliers = detect_outliers(series)

    assert len(outliers) > 0


def test_outlier_detection_normal():

    np.random.seed(42)

    series = pd.Series(np.random.normal(0, 1, 1000))

    outliers = detect_outliers(series)

    assert len(outliers) < 20
