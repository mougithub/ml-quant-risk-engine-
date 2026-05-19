"""
Correlation and PCA analysis.
"""

import numpy as np
import pandas as pd


def rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 60
):
    """
    Generate rolling correlation matrices.
    """

    correlations = {}

    for i in range(window, len(returns)):
        date = returns.index[i]

        corr_matrix = (
            returns.iloc[i - window:i]
            .corr()
        )

        correlations[date] = corr_matrix

    return correlations


def ewma_covariance(
    returns: pd.DataFrame,
    lambda_: float = 0.94
) -> pd.DataFrame:
    """
    EWMA covariance matrix.
    """

    cov = returns.cov().values

    for i in range(1, len(returns)):
        r = returns.iloc[i].values.reshape(-1, 1)

        cov = (
            lambda_ * cov
            + (1 - lambda_) * (r @ r.T)
        )

    return pd.DataFrame(
        cov,
        index=returns.columns,
        columns=returns.columns
    )


def eigenvalue_analysis(
    correlation_matrix: pd.DataFrame
):
    """
    PCA-style eigenvalue decomposition.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(
        correlation_matrix
    )

    idx = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance = (
        eigenvalues / eigenvalues.sum()
    )

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance": explained_variance
    }