"""
Visualization utilities for backtest results.

Provides minimal, professional charts for presenting strategy performance.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_equity_curve(
    equity_df: pd.DataFrame,
    save_path: Path,
    title: str = "Volatility Arbitrage Strategy Performance",
    benchmark_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Plot equity curve with drawdown overlay.

    Args:
        equity_df: DataFrame with columns: timestamp, total_equity
        save_path: Path to save the plot
        title: Chart title
        benchmark_df: Optional benchmark data for comparison

    Creates a two-panel chart:
    - Top: Cumulative return line
    - Bottom: Drawdown area chart
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # Prepare data
    equity = equity_df["total_equity"]
    timestamps = equity_df["timestamp"]

    # Calculate cumulative returns
    initial_equity = equity.iloc[0]
    cumulative_returns = (equity / initial_equity - 1) * 100  # Percentage

    # Top panel: Equity curve
    ax1.plot(
        timestamps,
        cumulative_returns,
        color="steelblue",
        linewidth=2.5,
        label="Strategy",
    )

    # Add benchmark if provided
    if benchmark_df is not None and "total_equity" in benchmark_df.columns:
        benchmark_equity = benchmark_df["total_equity"]
        benchmark_returns = (benchmark_equity / benchmark_equity.iloc[0] - 1) * 100
        ax1.plot(
            benchmark_df["timestamp"],
            benchmark_returns,
            color="gray",
            linewidth=2,
            linestyle="--",
            label="Benchmark",
            alpha=0.7,
        )

    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
    ax1.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Add performance stats box
    final_return = cumulative_returns.iloc[-1]
    max_return = cumulative_returns.max()
    stats_text = f"Final Return: {final_return:.1f}%\nPeak Return: {max_return:.1f}%"
    ax1.text(
        0.98,
        0.02,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Bottom panel: Drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max

    ax2.fill_between(
        timestamps,
        drawdown,
        0,
        color="crimson",
        alpha=0.4,
        label="Drawdown",
    )
    ax2.plot(
        timestamps,
        drawdown,
        color="darkred",
        linewidth=1.5,
    )

    ax2.set_ylabel("Drawdown (%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Add max drawdown annotation
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()
    ax2.annotate(
        f"Max DD: {max_dd_value:.1f}%",
        xy=(timestamps.iloc[max_dd_idx], max_dd_value),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved equity curve to {save_path}")


def plot_volatility_comparison(
    vol_data: pd.DataFrame,
    save_path: Path,
    title: str = "Implied vs Realized Volatility",
) -> None:
    """
    Plot implied volatility vs forecasted realized volatility.

    Args:
        vol_data: DataFrame with columns: timestamp, implied_vol, forecasted_vol, spread
        save_path: Path to save the plot
        title: Chart title

    Shows:
    - Two lines: IV and RV forecast
    - Shaded regions: Entry zones (when spread is large)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    timestamps = vol_data["timestamp"]
    iv = vol_data["implied_vol"] * 100  # Convert to percentage
    rv = vol_data["forecasted_vol"] * 100

    # Top panel: Volatility levels
    ax1.plot(
        timestamps,
        iv,
        color="darkorange",
        linewidth=2,
        label="Implied Volatility (Market)",
        alpha=0.8,
    )
    ax1.plot(
        timestamps,
        rv,
        color="darkgreen",
        linewidth=2,
        label="Forecasted Realized Vol (GARCH)",
        alpha=0.8,
    )

    ax1.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylabel("Volatility (%)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Bottom panel: Spread (IV - RV)
    spread = vol_data["spread"] * 100  # Percentage points

    ax2.plot(timestamps, spread, color="steelblue", linewidth=2)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

    # Highlight entry zones (spread > 5% or < -5%)
    entry_threshold = 5.0
    ax2.axhline(
        y=entry_threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Entry threshold: ±{entry_threshold}%",
    )
    ax2.axhline(y=-entry_threshold, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Fill entry zones
    ax2.fill_between(
        timestamps,
        spread,
        entry_threshold,
        where=(spread > entry_threshold),
        color="red",
        alpha=0.2,
        label="Short vol zone",
    )
    ax2.fill_between(
        timestamps,
        spread,
        -entry_threshold,
        where=(spread < -entry_threshold),
        color="green",
        alpha=0.2,
        label="Long vol zone",
    )

    ax2.set_ylabel("Volatility Spread (IV - RV, %)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved volatility comparison to {save_path}")


def plot_greeks_evolution(
    greeks_df: pd.DataFrame,
    save_path: Path,
    title: str = "Portfolio Greeks Evolution",
) -> None:
    """
    Plot portfolio Greeks over time.

    Args:
        greeks_df: DataFrame with columns: timestamp, portfolio_delta, portfolio_vega, etc.
        save_path: Path to save the plot
        title: Chart title

    Shows delta and vega exposure over the backtest period.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    timestamps = greeks_df["timestamp"]

    # Top panel: Delta (directional risk)
    if "portfolio_delta" in greeks_df.columns:
        delta = greeks_df["portfolio_delta"]
        ax1.plot(timestamps, delta, color="steelblue", linewidth=2)
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax1.fill_between(timestamps, delta, 0, alpha=0.3, color="steelblue")

        ax1.set_title(f"{title} - Delta", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Delta", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, linestyle="--")

    # Bottom panel: Vega (volatility risk)
    if "portfolio_vega" in greeks_df.columns:
        vega = greeks_df["portfolio_vega"]
        ax2.plot(timestamps, vega, color="darkorange", linewidth=2)
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax2.fill_between(timestamps, vega, 0, alpha=0.3, color="darkorange")

        ax2.set_title(f"{title} - Vega", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Portfolio Vega", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved Greeks evolution to {save_path}")


def create_summary_table(
    metrics: dict,
    save_path: Path,
) -> None:
    """
    Create a summary table image with key metrics.

    Args:
        metrics: Dictionary of performance metrics
        save_path: Path to save the table image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    # Prepare data
    table_data = [
        ["Metric", "Value"],
        ["Total Return", f"{metrics.get('total_return_pct', 0):.2f}%"],
        ["Annualized Return", f"{metrics.get('annualized_return', 0):.2f}%"],
        ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ["Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"],
        ["Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%"],
        ["Win Rate", f"{metrics.get('win_rate', 0):.1f}%"],
        ["Number of Trades", f"{metrics.get('num_trades', 0)}"],
        ["Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"],
    ]

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.6, 0.4],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor("#4CAF50")
        cell.set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#f0f0f0")

    plt.title("Strategy Performance Summary", fontsize=16, fontweight="bold", pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved summary table to {save_path}")
