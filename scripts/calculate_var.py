"""
calculate_var.py

Calculate Historical VaR for all tickers and save estimates.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import load_processed_data
from src.risk.historical_var import (
    rolling_historical_var,
    historical_var,
    calculate_var_confidence_intervals,
)


def main():
    """Calculate VaR for all tickers."""

    print("=" * 70)
    print("HISTORICAL VaR CALCULATION")
    print("=" * 70)

    # Configuration
    WINDOW = 250  # ~1 year of trading days
    CONFIDENCE_LEVELS = [0.95, 0.99]

    print(f"\nConfiguration:")
    print(f"  Window size: {WINDOW} days (~1 year)")
    print(f"  Confidence levels: {[f'{int(cl*100)}%' for cl in CONFIDENCE_LEVELS]}")

    # Load data
    print("\n" + "-" * 70)
    print("Loading data...")
    print("-" * 70)

    returns = load_processed_data("portfolio_returns")
    tickers = returns.columns.tolist()

    print(f"Loaded {len(returns)} days of data")
    print(f"Tickers: {', '.join(tickers)}")

    # Calculate rolling VaR for each ticker and confidence level
    print("\n" + "-" * 70)
    print("Calculating rolling VaR...")
    print("-" * 70)

    var_results = {}

    for cl in CONFIDENCE_LEVELS:
        cl_str = f"{int(cl*100)}%"
        print(f"\n{cl_str} VaR:")

        for ticker in tickers:
            var_series = rolling_historical_var(
                returns[ticker], window=WINDOW, confidence_level=cl
            )

            var_results[f"{ticker}_{cl_str}"] = var_series

            # Print current VaR (most recent value)
            current_var = var_series.iloc[-1]
            print(f"  {ticker:6s}: {current_var:7.4f} ({current_var*100:5.2f}%)")

    # Create DataFrame with all VaR estimates
    var_df = pd.DataFrame(var_results)
    var_df.index.name = "Date"

    # Save to file
    output_path = "data/processed/var_estimates.csv"
    var_df.to_csv(output_path)
    print(f"\n✓ VaR estimates saved to {output_path}")

    # Calculate point estimates (latest window)
    print("\n" + "-" * 70)
    print("Point VaR Estimates (Latest Window):")
    print("-" * 70)

    point_estimates = {}

    for ticker in tickers:
        # Use last WINDOW days
        recent_returns = returns[ticker].iloc[-WINDOW:]

        var_dict = {}
        for cl in CONFIDENCE_LEVELS:
            var_dict[f"{int(cl*100)}%"] = historical_var(recent_returns, cl)

        point_estimates[ticker] = var_dict

    point_df = pd.DataFrame(point_estimates).T
    point_df.index.name = "Ticker"

    print(point_df.to_string())

    # Save point estimates
    point_output_path = "data/processed/var_point_estimates.csv"
    point_df.to_csv(point_output_path)
    print(f"\n✓ Point estimates saved to {point_output_path}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for cl in CONFIDENCE_LEVELS:
        cl_str = f"{int(cl*100)}%"
        print(f"\n{cl_str} VaR Summary:")

        var_cols = [col for col in var_df.columns if cl_str in col]
        summary = var_df[var_cols].describe()

        # Rename columns for clarity
        summary.columns = [col.split("_")[0] for col in summary.columns]

        print(summary.to_string())

    # Create summary report
    report_path = "data/var_summary_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("HISTORICAL VaR SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Date Range: {returns.index.min().date()} to {returns.index.max().date()}\n"
        )
        f.write(f"Total Days: {len(returns)}\n")
        f.write(f"Window Size: {WINDOW} days\n")
        f.write(f"Tickers: {', '.join(tickers)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("CURRENT VaR ESTIMATES (Most Recent)\n")
        f.write("-" * 70 + "\n\n")

        f.write(point_df.to_string())
        f.write("\n\n")

        f.write("-" * 70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n\n")

        f.write("95% VaR Interpretation:\n")
        f.write("- We are 95% confident that daily losses will not exceed VaR\n")
        f.write("- Expected to breach ~5% of the time (1 in 20 days)\n")
        f.write("- Used for risk monitoring and position sizing\n\n")

        f.write("99% VaR Interpretation:\n")
        f.write("- We are 99% confident that daily losses will not exceed VaR\n")
        f.write("- Expected to breach ~1% of the time (1 in 100 days)\n")
        f.write("- Used for tail risk assessment and stress testing\n\n")

        f.write("-" * 70 + "\n")
        f.write("KEY OBSERVATIONS\n")
        f.write("-" * 70 + "\n\n")

        # Rank by risk
        var_95_current = {
            ticker: var_df[f"{ticker}_95%"].iloc[-1] for ticker in tickers
        }
        ranked = sorted(var_95_current.items(), key=lambda x: x[1], reverse=True)

        f.write("Tickers ranked by risk (95% VaR, highest to lowest):\n")
        for i, (ticker, var) in enumerate(ranked, 1):
            f.write(f"  {i}. {ticker}: {var:.4f} ({var*100:.2f}%)\n")

        f.write("\nNote: Higher VaR = higher risk\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ Summary report saved to {report_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("CALCULATION COMPLETE")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  {output_path}")
    print(f"  {point_output_path}")
    print(f"  {report_path}")
    print("\nNext steps:")
    print("  1. Review data/var_summary_report.txt")
    print("  2. Run notebooks/03_historical_var.ipynb for visualizations")
    print("  3. Proceed to Day 5A: Parametric VaR")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
