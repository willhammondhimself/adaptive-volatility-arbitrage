"""
Market Regime Detection Implementation.

Implements machine learning approaches to detect market regimes based on
returns, volatility, and volume patterns. Supports both Gaussian Mixture Models
and Hidden Markov Models.

Typical regimes:
    - Low Volatility: Stable markets with low returns variance
    - High Volatility: Elevated volatility with larger price swings
    - Crisis: Extreme volatility with significant downside risk
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm  # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeStatistics:
    """
    Statistics for a detected market regime.

    Attributes:
        regime_id: Regime identifier (0, 1, 2, ...)
        mean_return: Average return in this regime
        volatility: Return volatility in this regime
        duration_days: Average duration in days
        frequency: Percentage of time in this regime
    """

    regime_id: int
    mean_return: float
    volatility: float
    duration_days: float
    frequency: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "regime_id": self.regime_id,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "duration_days": self.duration_days,
            "frequency": self.frequency,
        }


class RegimeDetector(ABC):
    """
    Abstract base class for regime detection models.

    Regime detectors identify distinct market states based on historical
    price patterns, volatility, and volume.
    """

    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of regimes to detect (typically 2-3)
        """
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> None:
        """
        Fit regime detector to historical data.

        Args:
            returns: Return time series
            volatility: Realized volatility time series (optional)
            volume: Volume time series (optional)
        """
        pass

    @abstractmethod
    def predict(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Predict regime labels for given data.

        Args:
            returns: Return time series
            volatility: Realized volatility time series (optional)
            volume: Volume time series (optional)

        Returns:
            Series of regime labels (0, 1, 2, ...)
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Predict regime probabilities for given data.

        Args:
            returns: Return time series
            volatility: Realized volatility time series (optional)
            volume: Volume time series (optional)

        Returns:
            DataFrame with probability for each regime
        """
        pass

    def _prepare_features(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Prepare feature matrix for regime detection.

        Args:
            returns: Return time series
            volatility: Realized volatility time series (optional)
            volume: Volume time series (optional)

        Returns:
            Feature matrix [n_samples x n_features]
        """
        features = [returns.values]

        if volatility is not None:
            features.append(volatility.values)

        if volume is not None:
            features.append(volume.values)

        return np.column_stack(features)

    def get_regime_statistics(
        self,
        returns: pd.Series,
        regime_labels: pd.Series,
    ) -> list[RegimeStatistics]:
        """
        Calculate statistics for each detected regime.

        Args:
            returns: Return time series
            regime_labels: Regime labels from prediction

        Returns:
            List of RegimeStatistics for each regime
        """
        statistics = []

        for regime_id in range(self.n_regimes):
            mask = regime_labels == regime_id
            regime_returns = returns[mask]

            if len(regime_returns) == 0:
                continue

            # Calculate duration (average consecutive days in regime)
            durations = []
            current_duration = 0

            for label in regime_labels:
                if label == regime_id:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0

            if current_duration > 0:
                durations.append(current_duration)

            avg_duration = np.mean(durations) if durations else 0

            stats = RegimeStatistics(
                regime_id=regime_id,
                mean_return=float(regime_returns.mean()),
                volatility=float(regime_returns.std()),
                duration_days=float(avg_duration),
                frequency=float(len(regime_returns) / len(returns) * 100),
            )

            statistics.append(stats)

        return statistics


class GaussianMixtureRegimeDetector(RegimeDetector):
    """
    Regime detection using Gaussian Mixture Models.

    Uses sklearn's GaussianMixture to cluster market states based on
    statistical features. Assumes each regime follows a Gaussian distribution.

    Best for: Identifying stable regime structures
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        """
        Initialize GMM regime detector.

        Args:
            n_regimes: Number of regimes to detect
            covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
            random_state: Random seed for reproducibility
        """
        super().__init__(n_regimes)
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: Optional[GaussianMixture] = None

    def fit(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> None:
        """Fit GMM to historical data."""
        logger.info(f"Fitting GMM with {self.n_regimes} regimes")

        # Prepare features
        X = self._prepare_features(returns, volatility, volume)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit Gaussian Mixture Model
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=1000,
        )

        self.model.fit(X_scaled)
        self.is_fitted = True

        logger.info(f"GMM fitted successfully (BIC: {self.model.bic(X_scaled):.2f})")

    def predict(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Predict regime labels."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = self._prepare_features(returns, volatility, volume)
        X_scaled = self.scaler.transform(X)

        labels = self.model.predict(X_scaled)

        return pd.Series(labels, index=returns.index, name="regime")

    def predict_proba(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Predict regime probabilities."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = self._prepare_features(returns, volatility, volume)
        X_scaled = self.scaler.transform(X)

        probas = self.model.predict_proba(X_scaled)

        return pd.DataFrame(
            probas,
            index=returns.index,
            columns=[f"regime_{i}_prob" for i in range(self.n_regimes)],
        )


class HiddenMarkovRegimeDetector(RegimeDetector):
    """
    Regime detection using Hidden Markov Models.

    Uses hmmlearn's GaussianHMM to model regime sequences with transition
    probabilities. Captures temporal dependencies between regimes.

    Best for: Modeling regime transitions and persistence
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            covariance_type: Type of covariance parameters
            n_iter: Number of EM iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(n_regimes)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.transition_matrix: Optional[np.ndarray] = None

    def fit(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> None:
        """Fit HMM to historical data."""
        logger.info(f"Fitting HMM with {self.n_regimes} regimes")

        # Prepare features
        X = self._prepare_features(returns, volatility, volume)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit Hidden Markov Model
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        self.model.fit(X_scaled)
        self.transition_matrix = self.model.transmat_
        self.is_fitted = True

        logger.info(f"HMM fitted successfully (score: {self.model.score(X_scaled):.2f})")

    def predict(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Predict regime labels using Viterbi algorithm."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = self._prepare_features(returns, volatility, volume)
        X_scaled = self.scaler.transform(X)

        labels = self.model.predict(X_scaled)

        return pd.Series(labels, index=returns.index, name="regime")

    def predict_proba(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Predict regime probabilities (forward-backward algorithm)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = self._prepare_features(returns, volatility, volume)
        X_scaled = self.scaler.transform(X)

        probas = self.model.predict_proba(X_scaled)

        return pd.DataFrame(
            probas,
            index=returns.index,
            columns=[f"regime_{i}_prob" for i in range(self.n_regimes)],
        )

    def get_transition_probabilities(self) -> Optional[pd.DataFrame]:
        """
        Get regime transition probability matrix.

        Returns:
            DataFrame with transition probabilities [from_regime x to_regime]
        """
        if self.transition_matrix is None:
            return None

        return pd.DataFrame(
            self.transition_matrix,
            index=[f"regime_{i}" for i in range(self.n_regimes)],
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )


def plot_regime_transitions(
    timestamps: pd.Series,
    regime_labels: pd.Series,
    returns: Optional[pd.Series] = None,
) -> None:
    """
    Plot regime timeline with transitions.

    Args:
        timestamps: Time series timestamps
        regime_labels: Regime labels
        returns: Optional return series to overlay
    """
    # This will be imported when matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot returns if provided
        if returns is not None:
            ax2 = ax.twinx()
            ax2.plot(timestamps, returns, alpha=0.3, color="gray", label="Returns")
            ax2.set_ylabel("Returns")
            ax2.legend(loc="upper right")

        # Color map for regimes
        colors = ["green", "yellow", "red"]
        regime_colors = [colors[int(r)] for r in regime_labels]

        # Plot regime as background color
        for i in range(len(timestamps) - 1):
            ax.axvspan(
                timestamps.iloc[i],
                timestamps.iloc[i + 1],
                alpha=0.3,
                color=regime_colors[i],
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Regime")
        ax.set_title("Market Regime Timeline")

        plt.tight_layout()
        plt.show()

    except ImportError:
        logger.warning("matplotlib not available for plotting")


def regime_conditional_metrics(
    returns: pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    Calculate regime-conditional performance metrics.

    Args:
        returns: Return time series
        regime_labels: Regime labels

    Returns:
        DataFrame with metrics by regime
    """
    metrics = []

    for regime_id in regime_labels.unique():
        mask = regime_labels == regime_id
        regime_returns = returns[mask]

        if len(regime_returns) == 0:
            continue

        # Calculate Sharpe ratio (assuming 252 trading days)
        sharpe = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0

        metrics.append({
            "regime": int(regime_id),
            "observations": len(regime_returns),
            "mean_return": regime_returns.mean(),
            "volatility": regime_returns.std(),
            "sharpe_ratio": sharpe,
            "min_return": regime_returns.min(),
            "max_return": regime_returns.max(),
        })

    return pd.DataFrame(metrics)
