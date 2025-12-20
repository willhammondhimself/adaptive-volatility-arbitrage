#!/usr/bin/env python3
"""
Demo: Bayesian LSTM Volatility Forecasting with Uncertainty-Adjusted Sizing.

Demonstrates:
1. Synthetic price data generation with volatility clustering
2. BayesianVolNet training
3. Monte Carlo Dropout uncertainty estimation
4. Uncertainty-adjusted position sizing
5. Visualization with confidence intervals

Output: docs/bayesian_uncertainty_plot.png
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volatility_arbitrage.models.bayesian_lstm import BayesianVolNet, BayesianVolNetConfig
from volatility_arbitrage.risk.uncertainty_sizing import size_position_with_uncertainty


def generate_synthetic_data(
    n_days: int = 500,
    base_vol: float = 0.15,
    vol_of_vol: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic price series with time-varying volatility.

    Uses a simple GARCH-like process for volatility clustering.

    Returns:
        (returns, realized_vol) arrays
    """
    np.random.seed(seed)

    # Generate time-varying volatility (GARCH-like)
    vol = np.zeros(n_days)
    vol[0] = base_vol

    omega = base_vol**2 * 0.05  # Long-run variance weight
    alpha = 0.10  # Shock coefficient
    beta = 0.85  # Persistence

    returns = np.zeros(n_days)

    for t in range(1, n_days):
        # GARCH(1,1) variance update
        var_t = omega + alpha * returns[t - 1] ** 2 + beta * vol[t - 1] ** 2
        vol[t] = np.sqrt(var_t)

        # Generate return with current volatility
        returns[t] = vol[t] * np.random.randn() / np.sqrt(252)

    # Compute realized vol (rolling 20-day)
    realized_vol = np.zeros(n_days)
    for t in range(20, n_days):
        realized_vol[t] = np.std(returns[t - 20 : t]) * np.sqrt(252)

    return returns, realized_vol


def prepare_sequences(
    returns: np.ndarray,
    realized_vol: np.ndarray,
    seq_len: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare training sequences for LSTM.

    Input: past `seq_len` returns
    Target: realized vol at the next timestep
    """
    X, y = [], []

    for t in range(seq_len, len(returns) - 1):
        # Input: past returns
        X.append(returns[t - seq_len : t])
        # Target: next realized vol
        y.append(realized_vol[t + 1])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM: (batch, seq_len, features)
    X = X.reshape(-1, seq_len, 1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)


def train_model(
    model: BayesianVolNet,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 5,
    lr: float = 0.01,
) -> list[float]:
    """Train the model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.6f}")

    return losses


def main() -> None:
    print("=" * 60)
    print("Bayesian LSTM Volatility Forecasting Demo")
    print("=" * 60)

    # Generate synthetic data
    print("\n[1] Generating synthetic price data...")
    returns, realized_vol = generate_synthetic_data(n_days=500)
    print(f"    Generated {len(returns)} days of data")
    print(f"    Avg realized vol: {realized_vol[20:].mean() * 100:.1f}%")

    # Prepare sequences
    print("\n[2] Preparing training sequences...")
    seq_len = 20
    X, y = prepare_sequences(returns, realized_vol, seq_len=seq_len)

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Initialize model
    print("\n[3] Initializing BayesianVolNet...")
    config = BayesianVolNetConfig(
        input_size=1,
        hidden_size=64,
        dropout_p=0.2,
        n_mc_samples=50,
    )
    model = BayesianVolNet(config)
    print(f"    Architecture: LSTM(64) -> Dropout(0.2) -> Linear(1)")

    # Train
    print("\n[4] Training for 5 epochs...")
    losses = train_model(model, X_train, y_train, epochs=5)

    # Generate predictions with uncertainty on test set
    print("\n[5] Running Monte Carlo Dropout inference...")
    predictions = []
    uncertainties = []

    for i in range(len(X_test)):
        result = model.predict_with_uncertainty(X_test[i : i + 1], n_samples=50)
        predictions.append(result["mean_vol"])
        uncertainties.append(result["epistemic_uncertainty"])

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    actuals = y_test.numpy().squeeze()

    # Example position sizing
    print("\n[6] Computing uncertainty-adjusted position sizes...")
    capital = 1_000_000  # $1M

    # Pick a sample point for display
    sample_idx = len(predictions) // 2
    sample_pred = predictions[sample_idx]
    sample_unc = uncertainties[sample_idx]

    # Signal strength based on vol deviation from mean
    mean_vol = predictions.mean()
    signal = (sample_pred - mean_vol) / mean_vol  # Normalized deviation

    position_size = size_position_with_uncertainty(
        signal_strength=abs(signal),
        uncertainty=sample_unc,
        capital=capital,
        kelly_fraction=0.25,
        uncertainty_penalty=2.0,
    )

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Predicted Vol: {sample_pred * 100:.1f}%")
    print(f"Uncertainty:   {sample_unc * 100:.1f}%")
    print(f"Rec. Position: ${position_size:,.0f}")
    print("=" * 60)

    # Generate plot
    print("\n[7] Generating uncertainty plot...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Time axis for test period
    t = np.arange(len(predictions))

    # Top plot: Volatility forecast with confidence interval
    ax1 = axes[0]
    ax1.plot(t, actuals * 100, "b-", label="Realized Vol", alpha=0.7, linewidth=1.5)
    ax1.plot(t, predictions * 100, "r-", label="Predicted Vol", linewidth=1.5)
    ax1.fill_between(
        t,
        (predictions - 2 * uncertainties) * 100,
        (predictions + 2 * uncertainties) * 100,
        alpha=0.3,
        color="red",
        label="95% CI (±2σ)",
    )
    ax1.set_ylabel("Volatility (%)")
    ax1.set_title("Bayesian LSTM Volatility Forecast with Epistemic Uncertainty")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Uncertainty over time
    ax2 = axes[1]
    ax2.fill_between(t, 0, uncertainties * 100, alpha=0.5, color="orange")
    ax2.plot(t, uncertainties * 100, "orange", linewidth=1.5)
    ax2.set_xlabel("Days (Test Period)")
    ax2.set_ylabel("Epistemic Uncertainty (%)")
    ax2.set_title("Model Uncertainty (Monte Carlo Dropout Std Dev)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "bayesian_uncertainty_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {output_path}")

    plt.close()
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
