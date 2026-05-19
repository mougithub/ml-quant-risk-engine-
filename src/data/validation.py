import pandas as pd
import numpy as np


def check_missing_values(df: pd.DataFrame):

    return df.isnull().sum()


def detect_outliers(series: pd.Series, threshold=3):

    median = series.median()

    mad = np.median(np.abs(series - median))

    # avoid division by zero
    if mad == 0:
        return series[series != median]

    modified_z = 0.6745 * (series - median) / mad

    return series[np.abs(modified_z) > threshold]


def generate_quality_report(df: pd.DataFrame):

    report = []

    report.append("DATA QUALITY REPORT")
    report.append("=" * 40)

    report.append(f"Shape: {df.shape}")

    report.append("\nMissing Values:")
    report.append(str(check_missing_values(df)))

    report.append("\nSummary Statistics:")
    report.append(str(df.describe()))

    return "\n".join(report)
