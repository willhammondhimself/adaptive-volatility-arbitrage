"""
Alternative Realized Volatility Estimators.

Implements more efficient RV estimators using OHLC data:
- Parkinson (1980): High-low range estimator
- Garman-Klass (1980): OHLC estimator with drift adjustment
- Yang-Zhang (2000): Handles overnight gaps

References:
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data.
- Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open, and close prices.
"""

import numpy as np
import pandas as pd
from typing import Literal, List, Optional

RVMethod = Literal["close_to_close", "parkinson", "garman_klass", "yang_zhang"]


def close_to_close_rv(
    prices: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Standard close-to-close realized volatility.

    This is the naive estimator using log returns of closing prices.

    Args:
        prices: DataFrame with 'close' column
        window: Rolling window in days
        annualize: If True, annualize to yearly volatility

    Returns:
        Series of RV estimates
    """
    if 'close' not in prices.columns:
        raise ValueError("prices must have 'close' column")

    returns = np.log(prices['close'] / prices['close'].shift(1))
    rv = returns.rolling(window).std()

    if annualize:
        rv = rv * np.sqrt(252)

    return rv


def parkinson_rv(
    prices: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Parkinson (1980) high-low range estimator.

    sigma^2 = (1 / 4*ln(2)) * E[ln(H/L)^2]

    This estimator is approximately 5x more efficient than close-to-close
    for continuous price processes (no jumps).

    Args:
        prices: DataFrame with 'high' and 'low' columns
        window: Rolling window in days
        annualize: If True, annualize to yearly volatility

    Returns:
        Series of RV estimates
    """
    required_cols = ['high', 'low']
    for col in required_cols:
        if col not in prices.columns:
            raise ValueError(f"prices must have '{col}' column")

    # Parkinson constant
    k = 1.0 / (4 * np.log(2))

    log_hl = np.log(prices['high'] / prices['low'])
    variance = k * (log_hl ** 2).rolling(window).mean()

    # Ensure non-negative
    variance = variance.clip(lower=0)
    rv = np.sqrt(variance)

    if annualize:
        rv = rv * np.sqrt(252)

    return rv


def garman_klass_rv(
    prices: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Garman-Klass (1980) OHLC estimator.

    sigma^2 = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

    This estimator accounts for opening jumps and is more efficient
    than Parkinson when there is drift.

    Args:
        prices: DataFrame with 'open', 'high', 'low', 'close' columns
        window: Rolling window in days
        annualize: If True, annualize to yearly volatility

    Returns:
        Series of RV estimates
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in prices.columns:
            raise ValueError(f"prices must have '{col}' column")

    log_hl = np.log(prices['high'] / prices['low'])
    log_co = np.log(prices['close'] / prices['open'])

    # Garman-Klass formula
    variance = (0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2).rolling(window).mean()

    # Ensure non-negative
    variance = variance.clip(lower=0)
    rv = np.sqrt(variance)

    if annualize:
        rv = rv * np.sqrt(252)

    return rv


def yang_zhang_rv(
    prices: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Yang-Zhang (2000) estimator for overnight gaps.

    Combines overnight, open-to-close, and Rogers-Satchell estimators.
    This is the most efficient estimator for assets with significant
    overnight gaps (like most equities).

    The estimator is drift-independent and handles both opening jumps
    and intraday volatility.

    Args:
        prices: DataFrame with 'open', 'high', 'low', 'close' columns
        window: Rolling window in days
        annualize: If True, annualize to yearly volatility

    Returns:
        Series of RV estimates
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in prices.columns:
            raise ValueError(f"prices must have '{col}' column")

    # Overnight return (close-to-open)
    log_overnight = np.log(prices['open'] / prices['close'].shift(1))

    # Open-to-close return
    log_oc = np.log(prices['close'] / prices['open'])

    # Rogers-Satchell variance (intraday component)
    log_ho = np.log(prices['high'] / prices['open'])
    log_lo = np.log(prices['low'] / prices['open'])
    log_hc = np.log(prices['high'] / prices['close'])
    log_lc = np.log(prices['low'] / prices['close'])
    rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()

    # Component variances
    overnight_var = log_overnight.rolling(window).var()
    open_close_var = log_oc.rolling(window).var()

    # Yang-Zhang combination (k=0.34 is optimal for GBM)
    k = 0.34
    variance = overnight_var + k * open_close_var + (1 - k) * rs_var

    # Ensure non-negative
    variance = variance.clip(lower=0)
    rv = np.sqrt(variance)

    if annualize:
        rv = rv * np.sqrt(252)

    return rv


def calculate_rv(
    prices: pd.DataFrame,
    method: RVMethod = "close_to_close",
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate realized volatility using specified method.

    Args:
        prices: DataFrame with OHLC columns (open, high, low, close)
        method: RV estimation method
        window: Rolling window in days
        annualize: If True, annualize to yearly volatility

    Returns:
        Series of annualized RV estimates
    """
    methods = {
        "close_to_close": close_to_close_rv,
        "parkinson": parkinson_rv,
        "garman_klass": garman_klass_rv,
        "yang_zhang": yang_zhang_rv,
    }

    if method not in methods:
        raise ValueError(f"Unknown RV method: {method}. Choose from {list(methods.keys())}")

    return methods[method](prices, window, annualize)


def ensemble_rv(
    prices: pd.DataFrame,
    methods: Optional[List[RVMethod]] = None,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Average multiple RV estimators for robustness.

    Combining estimators can reduce estimation error when the true
    data generating process is unknown.

    Args:
        prices: DataFrame with OHLC columns
        methods: List of methods to combine (default: all except close_to_close)
        window: Rolling window in days
        annualize: If True, annualize to yearly volatility

    Returns:
        Series of averaged RV estimates
    """
    if methods is None:
        methods = ["parkinson", "garman_klass", "yang_zhang"]

    estimates = []
    for method in methods:
        try:
            est = calculate_rv(prices, method, window, annualize)
            estimates.append(est)
        except ValueError:
            # Skip methods that can't be computed (missing columns)
            continue

    if not estimates:
        raise ValueError("No valid RV methods could be computed")

    return pd.concat(estimates, axis=1).mean(axis=1)
