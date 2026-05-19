"""
Generate portfolio features.
"""

import pandas as pd

from src.features.volatility import (
    calculate_rolling_volatility,
    calculate_ewma_volatility
)

from src.features.rolling_stats import (
    rolling_mean,
    rolling_skewness,
    rolling_kurtosis
)


def main():

    print("Loading returns data...")

    returns = pd.read_csv(
        "data/processed/portfolio_returns.csv",
        index_col=0,
        parse_dates=True
    )

    portfolio_returns = returns.mean(axis=1)

    print("Calculating volatility...")

    features = pd.DataFrame(index=returns.index)

    features["vol_20"] = (
        calculate_rolling_volatility(
            portfolio_returns,
            window=20
        )
    )

    features["vol_60"] = (
        calculate_rolling_volatility(
            portfolio_returns,
            window=60
        )
    )

    features["ewma_vol"] = (
        calculate_ewma_volatility(
            portfolio_returns
        )
    )

    print("Calculating rolling statistics...")

    features["rolling_mean"] = (
        rolling_mean(portfolio_returns)
    )

    features["rolling_skew"] = (
        rolling_skewness(portfolio_returns)
    )

    features["rolling_kurtosis"] = (
        rolling_kurtosis(portfolio_returns)
    )

    output_path = (
        "data/processed/portfolio_features.csv"
    )

    features.to_csv(output_path)

    print(f"Saved features to {output_path}")

    print(features.describe())


if __name__ == "__main__":
    main()