"""
historical_var.py

Historical Value at Risk (VaR) implementation.

Non-parametric VaR estimation using empirical distribution
of historical returns. No distribution assumptions required.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional


def historical_var(
    returns: Union[pd.Series, np.ndarray], confidence_level: float = 0.95
) -> float:
    """
    Calculate Historical Value at Risk.

    Uses empirical distribution of returns - no parametric assumptions.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% VaR)

    Returns
    -------
    float
        VaR estimate (positive value representing potential loss)

    Notes
    -----
    Historical VaR is the negative of the α-percentile of returns:
    VaR_α = -Percentile(returns, (1-α) × 100)

    For 95% VaR: VaR_0.95 = -Percentile(returns, 5)

    Interpretation: We are 95% confident that losses won't exceed VaR
    on any given day.

    Examples
    --------
    >>> returns = pd.Series([-0.02, -0.01, 0.01, 0.02, -0.03])
    >>> var_95 = historical_var(returns, confidence_level=0.95)
    >>> print(f"95% VaR: {var_95:.4f}")
    """
    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    # Remove NaN values
    returns_clean = (
        returns[~np.isnan(returns)]
        if isinstance(returns, np.ndarray)
        else returns.dropna()
    )

    if len(returns_clean) == 0:
        raise ValueError("No valid returns after removing NaN")

    # Calculate percentile
    # For 95% confidence, we want 5th percentile (lower tail)
    percentile = (1 - confidence_level) * 100
    # var = -np.percentile(returns_clean, percentile)
    var = -np.percentile(returns_clean, percentile, method="lower")
    return var


def rolling_historical_var(
    returns: pd.Series,
    window: int = 250,
    confidence_level: float = 0.95,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Calculate rolling Historical VaR over time.

    Creates a time series of VaR estimates using a rolling window.

    Parameters
    ----------
    returns : pd.Series
        Historical returns with DatetimeIndex
    window : int
        Rolling window size in days (default: 250 = ~1 year)
    confidence_level : float
        Confidence level (default: 0.95)
    min_periods : int, optional
        Minimum periods required. Defaults to window.

    Returns
    -------
    pd.Series
        Rolling VaR estimates, indexed by date

    Notes
    -----
    Window size choice:
    - Too small (< 100 days): Noisy, unstable estimates
    - Too large (> 500 days): Stale, slow to adapt to regime changes
    - 250 days (~1 year): Good balance

    Examples
    --------
    >>> rolling_var = rolling_historical_var(returns, window=250, confidence_level=0.95)
    >>> print(rolling_var.tail())
    """
    if min_periods is None:
        min_periods = window

    def calc_var(x):
        if len(x) < min_periods:
            return np.nan
        return historical_var(x, confidence_level)

    rolling_var = returns.rolling(window=window, min_periods=min_periods).apply(
        calc_var, raw=True
    )

    return rolling_var


def portfolio_historical_var(
    returns: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Historical VaR for a portfolio of assets.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns for each asset (columns = assets, rows = dates)
    weights : np.ndarray, optional
        Portfolio weights (must sum to 1). If None, equal weights.
    confidence_level : float
        Confidence level (default: 0.95)

    Returns
    -------
    float
        Portfolio VaR estimate

    Notes
    -----
    Portfolio VaR accounts for correlations between assets.
    Cannot simply sum individual VaRs due to diversification.

    Method: Calculate portfolio returns, then compute VaR on portfolio.

    Examples
    --------
    >>> portfolio_var = portfolio_historical_var(returns_df, weights=np.array([0.6, 0.4]))
    """
    if weights is None:
        # Equal weights
        weights = np.ones(returns.shape[1]) / returns.shape[1]

    # Validate weights
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1, got {weights.sum():.4f}")

    if len(weights) != returns.shape[1]:
        raise ValueError(
            f"Weights length ({len(weights)}) must match number of assets ({returns.shape[1]})"
        )

    # Calculate portfolio returns
    portfolio_returns = returns @ weights

    # Calculate VaR on portfolio returns
    var = historical_var(portfolio_returns, confidence_level)

    return var


def var_breach_analysis(
    returns: pd.Series, var_estimates: pd.Series, confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Analyze VaR breaches (exceedances).

    Identifies when actual losses exceeded VaR estimate.

    Parameters
    ----------
    returns : pd.Series
        Actual returns
    var_estimates : pd.Series
        VaR estimates (positive values)
    confidence_level : float
        Confidence level used for VaR

    Returns
    -------
    pd.DataFrame
        Breach analysis with columns:
        - 'return': Actual return
        - 'var': VaR estimate
        - 'breach': Boolean, True if breach occurred
        - 'excess': How much loss exceeded VaR (if breach)

    Notes
    -----
    A breach occurs when: actual_return < -var_estimate
    (i.e., loss is greater than VaR)

    Expected breach rate: (1 - confidence_level)
    For 95% VaR: expect ~5% breaches

    Examples
    --------
    >>> breach_df = var_breach_analysis(returns, var_series, confidence_level=0.95)
    >>> breach_rate = breach_df['breach'].mean()
    >>> print(f"Breach rate: {breach_rate:.2%} (expected: 5.00%)")
    """
    # Align indices
    common_index = returns.index.intersection(var_estimates.index)
    returns_aligned = returns.loc[common_index]
    var_aligned = var_estimates.loc[common_index]

    # Create analysis DataFrame
    df = pd.DataFrame({"return": returns_aligned, "var": var_aligned})

    # Breach occurs when loss > VaR (return < -VaR)
    df["breach"] = df["return"] < -df["var"]

    # Calculate excess loss (how much worse than VaR)
    df["excess"] = np.where(df["breach"], -df["return"] - df["var"], 0)

    return df


def calculate_var_confidence_intervals(
    returns: pd.Series, confidence_levels: list = [0.90, 0.95, 0.99]
) -> pd.DataFrame:
    """
    Calculate VaR at multiple confidence levels.

    Parameters
    ----------
    returns : pd.Series
        Historical returns
    confidence_levels : list
        List of confidence levels to calculate

    Returns
    -------
    pd.DataFrame
        VaR estimates at each confidence level

    Examples
    --------
    >>> var_df = calculate_var_confidence_intervals(returns, [0.90, 0.95, 0.99])
    >>> print(var_df)
    """
    var_estimates = {}

    for cl in confidence_levels:
        var_estimates[f"{int(cl*100)}%"] = historical_var(returns, cl)

    df = pd.DataFrame(var_estimates, index=["VaR"]).T
    df.index.name = "Confidence Level"

    return df


def get_breach_statistics(
    breach_df: pd.DataFrame, confidence_level: float = 0.95
) -> dict:
    """
    Calculate breach statistics from breach analysis.

    Parameters
    ----------
    breach_df : pd.DataFrame
        Output from var_breach_analysis()
    confidence_level : float
        Confidence level used

    Returns
    -------
    dict
        Statistics about VaR breaches

    Examples
    --------
    >>> stats = get_breach_statistics(breach_df, confidence_level=0.95)
    >>> print(f"Actual breach rate: {stats['breach_rate']:.2%}")
    """
    total_days = len(breach_df)
    breach_count = breach_df["breach"].sum()
    breach_rate = breach_count / total_days
    expected_breach_rate = 1 - confidence_level

    breach_dates = breach_df[breach_df["breach"]].index.tolist()

    if breach_count > 0:
        avg_excess = breach_df[breach_df["breach"]]["excess"].mean()
        max_excess = breach_df["excess"].max()
        worst_breach_date = breach_df["excess"].idxmax()
    else:
        avg_excess = 0
        max_excess = 0
        worst_breach_date = None

    stats = {
        "total_days": total_days,
        "breach_count": int(breach_count),
        "breach_rate": breach_rate,
        "expected_breach_rate": expected_breach_rate,
        "breach_rate_diff": breach_rate - expected_breach_rate,
        "avg_excess_loss": avg_excess,
        "max_excess_loss": max_excess,
        "worst_breach_date": worst_breach_date,
        "breach_dates": breach_dates,
    }

    return stats
