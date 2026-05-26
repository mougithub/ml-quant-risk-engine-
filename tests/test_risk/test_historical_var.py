"""
test_historical_var.py

Unit tests for Historical VaR calculations.
"""

import pytest
import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.risk.historical_var import (
    historical_var,
    rolling_historical_var,
    portfolio_historical_var,
    var_breach_analysis,
    calculate_var_confidence_intervals,
    get_breach_statistics,
)


class TestHistoricalVaR:
    """Test basic Historical VaR calculation."""

    def test_historical_var_normal_distribution(self):
        """Test VaR on normal distribution - should match theoretical value."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 10000))

        var_95 = historical_var(returns, confidence_level=0.95)

        # For normal distribution, 95% VaR ≈ 1.65 × σ
        theoretical_var = 1.65 * 0.01

        # Allow 10% tolerance due to sampling
        assert np.isclose(var_95, theoretical_var, rtol=0.1)

        # VaR should be positive
        assert var_95 > 0

    def test_historical_var_percentile_direction(self):
        """Test that VaR uses correct percentile direction."""
        returns = pd.Series([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05])

        var_95 = historical_var(returns, confidence_level=0.95)

        # 95% VaR should be at 5th percentile
        # 5th percentile of [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05] is -0.05
        # VaR = -(-0.05) = 0.05
        assert np.isclose(var_95, 0.05, atol=0.001)

    def test_historical_var_99_confidence(self):
        """Test 99% VaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 10000))

        var_99 = historical_var(returns, confidence_level=0.99)
        var_95 = historical_var(returns, confidence_level=0.95)

        # 99% VaR should be larger than 95% VaR (more conservative)
        assert var_99 > var_95

    def test_historical_var_with_nan(self):
        """Test VaR calculation with NaN values."""
        returns = pd.Series([0.01, np.nan, -0.02, 0.03, np.nan, -0.01])

        # Should not raise error, should ignore NaN
        var = historical_var(returns, confidence_level=0.95)

        assert not np.isnan(var)
        assert var > 0

    def test_historical_var_empty_raises_error(self):
        """Test that empty returns raises ValueError."""
        returns = pd.Series([])

        with pytest.raises(ValueError):
            historical_var(returns)

    def test_historical_var_all_nan_raises_error(self):
        """Test that all NaN raises ValueError."""
        returns = pd.Series([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError):
            historical_var(returns)


class TestRollingHistoricalVaR:
    """Test rolling Historical VaR."""

    def test_rolling_var_length(self):
        """Test that rolling VaR has correct length."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0, 0.01, 500),
            index=pd.date_range("2020-01-01", periods=500),
        )

        rolling_var = rolling_historical_var(returns, window=250)

        # Length should match returns
        assert len(rolling_var) == len(returns)

    def test_rolling_var_nan_at_start(self):
        """Test that rolling VaR has NaN at start (warmup period)."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0, 0.01, 300),
            index=pd.date_range("2020-01-01", periods=300),
        )

        window = 250
        rolling_var = rolling_historical_var(returns, window=window)

        # First window-1 values should be NaN
        assert pd.isna(rolling_var.iloc[: window - 1]).all()

        # After warmup, should have valid values
        assert not pd.isna(rolling_var.iloc[window:]).any()

    def test_rolling_var_time_varying(self):
        """Test that rolling VaR changes over time."""
        # Create returns with changing volatility
        np.random.seed(42)
        low_vol_returns = np.random.normal(0, 0.005, 200)
        high_vol_returns = np.random.normal(0, 0.02, 200)
        returns = pd.Series(
            np.concatenate([low_vol_returns, high_vol_returns]),
            index=pd.date_range("2020-01-01", periods=400),
        )

        rolling_var = rolling_historical_var(returns, window=100)

        # VaR in second half should be higher than first half
        var_first_half = rolling_var.iloc[200:250].mean()
        var_second_half = rolling_var.iloc[350:400].mean()

        assert var_second_half > var_first_half


class TestPortfolioHistoricalVaR:
    """Test portfolio VaR calculation."""

    def test_portfolio_var_equal_weights(self):
        """Test portfolio VaR with equal weights."""
        np.random.seed(42)
        returns = pd.DataFrame(
            {"A": np.random.normal(0, 0.01, 1000), "B": np.random.normal(0, 0.01, 1000)}
        )

        portfolio_var = portfolio_historical_var(returns)

        # Should be positive
        assert portfolio_var > 0

        # Should be less than sum of individual VaRs (diversification)
        var_a = historical_var(returns["A"])
        var_b = historical_var(returns["B"])

        # Equal weights: 0.5 × VaR_A + 0.5 × VaR_B
        # Portfolio VaR should be less due to diversification
        assert portfolio_var < 0.5 * var_a + 0.5 * var_b

    def test_portfolio_var_custom_weights(self):
        """Test portfolio VaR with custom weights."""
        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 1000),
                "B": np.random.normal(0, 0.02, 1000),  # Higher vol
            }
        )

        # 80% in A (lower vol), 20% in B (higher vol)
        weights = np.array([0.8, 0.2])
        portfolio_var = portfolio_historical_var(returns, weights=weights)

        assert portfolio_var > 0

    def test_portfolio_var_weights_sum_to_one(self):
        """Test that weights must sum to 1."""
        returns = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [-0.01, 0.02, -0.03]})

        weights = np.array([0.6, 0.3])  # Sum = 0.9, not 1

        with pytest.raises(ValueError):
            portfolio_historical_var(returns, weights=weights)

    def test_portfolio_var_perfect_correlation(self):
        """Test portfolio VaR with perfectly correlated assets."""
        np.random.seed(42)
        returns_a = pd.Series(np.random.normal(0, 0.01, 1000))
        returns_b = returns_a.copy()  # Perfect correlation

        returns = pd.DataFrame({"A": returns_a, "B": returns_b})
        weights = np.array([0.5, 0.5])

        portfolio_var = portfolio_historical_var(returns, weights=weights)
        individual_var = historical_var(returns_a)

        # With perfect correlation, portfolio VaR = weighted sum of individual VaRs
        assert np.isclose(portfolio_var, individual_var, rtol=0.01)


class TestVaRBreachAnalysis:
    """Test VaR breach detection and analysis."""

    def test_breach_detection(self):
        """Test that breaches are correctly identified."""
        returns = pd.Series(
            [-0.03, -0.01, 0.01, -0.05, 0.02],
            index=pd.date_range("2020-01-01", periods=5),
        )
        var_estimates = pd.Series(
            [0.02, 0.02, 0.02, 0.02, 0.02], index=pd.date_range("2020-01-01", periods=5)
        )

        breach_df = var_breach_analysis(returns, var_estimates)

        # Breaches occur when return < -VaR
        # Day 1: -0.03 < -0.02 → breach
        # Day 2: -0.01 >= -0.02 → no breach
        # Day 4: -0.05 < -0.02 → breach

        assert breach_df["breach"].iloc[0] == True
        assert breach_df["breach"].iloc[1] == False
        assert breach_df["breach"].iloc[3] == True

    def test_breach_excess_calculation(self):
        """Test excess loss calculation."""
        returns = pd.Series([-0.05], index=pd.date_range("2020-01-01", periods=1))
        var_estimates = pd.Series([0.02], index=pd.date_range("2020-01-01", periods=1))

        breach_df = var_breach_analysis(returns, var_estimates)

        # Excess = |-0.05| - 0.02 = 0.03
        assert np.isclose(breach_df["excess"].iloc[0], 0.03)

    def test_no_breach_zero_excess(self):
        """Test that non-breaches have zero excess."""
        returns = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1))
        var_estimates = pd.Series([0.02], index=pd.date_range("2020-01-01", periods=1))

        breach_df = var_breach_analysis(returns, var_estimates)

        assert breach_df["breach"].iloc[0] == False
        assert breach_df["excess"].iloc[0] == 0


class TestBreachStatistics:
    """Test breach statistics calculation."""

    def test_breach_statistics_normal_case(self):
        """Test breach statistics with normal breach pattern."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000),
        )
        var_estimates = pd.Series(0.0165, index=returns.index)  # Theoretical 95% VaR

        breach_df = var_breach_analysis(returns, var_estimates, confidence_level=0.95)
        stats = get_breach_statistics(breach_df, confidence_level=0.95)

        # Should have approximately 5% breaches
        assert 0.03 < stats["breach_rate"] < 0.07  # Allow some sampling variation
        assert stats["breach_count"] > 0
        # assert stats['expected_breach_rate'] == 0.05
        assert np.isclose(stats["expected_breach_rate"], 0.05)

    def test_breach_statistics_no_breaches(self):
        """Test statistics when there are no breaches."""
        returns = pd.Series(
            [0.01, 0.02, 0.03], index=pd.date_range("2020-01-01", periods=3)
        )
        var_estimates = pd.Series(
            [0.10, 0.10, 0.10], index=pd.date_range("2020-01-01", periods=3)
        )

        breach_df = var_breach_analysis(returns, var_estimates)
        stats = get_breach_statistics(breach_df)

        assert stats["breach_count"] == 0
        assert stats["breach_rate"] == 0
        assert stats["avg_excess_loss"] == 0
        assert stats["worst_breach_date"] is None


class TestConfidenceIntervals:
    """Test VaR at multiple confidence levels."""

    def test_multiple_confidence_levels(self):
        """Test that higher confidence → higher VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 10000))

        var_df = calculate_var_confidence_intervals(returns, [0.90, 0.95, 0.99])

        # VaR should increase with confidence level
        assert var_df.loc["90%", "VaR"] < var_df.loc["95%", "VaR"]
        assert var_df.loc["95%", "VaR"] < var_df.loc["99%", "VaR"]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
