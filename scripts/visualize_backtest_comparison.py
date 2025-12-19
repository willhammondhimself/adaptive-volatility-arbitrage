#!/usr/bin/env python3
"""
Generate comparison chart: Strategy vs Buy-and-Hold vs No Trading.

Shows training and testing periods with clear labels.
"""

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from volatility_arbitrage.core.config import load_strategy_config

# Import backtest runner
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load_json_options_data, run_qv_backtest


def load_spy_ohlc(path: str = "data/spy_ohlc.csv") -> pd.DataFrame:
    """Load SPY OHLC data for buy-and-hold benchmark."""
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df


def calculate_buy_and_hold(ohlc_df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Calculate buy-and-hold returns aligned to backtest dates."""
    # Align to backtest dates
    aligned = ohlc_df.reindex(dates, method="ffill")
    initial_price = aligned["close"].iloc[0]
    returns = (aligned["close"] / initial_price - 1) * 100
    return returns


def plot_comparison(
    strategy_equity: pd.DataFrame,
    ohlc_df: pd.DataFrame,
    train_end: str,
    output_path: str,
    initial_capital: float = 100000,
):
    """
    Create comparison chart with training/testing periods labeled.

    Args:
        strategy_equity: DataFrame with 'equity' column indexed by date
        ohlc_df: SPY OHLC data for buy-and-hold
        train_end: End date of training period (YYYY-MM-DD)
        output_path: Path to save the chart
        initial_capital: Starting capital
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    # Prepare data
    dates = pd.to_datetime(strategy_equity.index)
    train_end_date = pd.to_datetime(train_end)

    # Strategy returns
    strategy_returns = (strategy_equity["equity"] / initial_capital - 1) * 100

    # Buy-and-hold returns
    bh_returns = calculate_buy_and_hold(ohlc_df, dates)

    # No trading (flat line)
    no_trading = pd.Series(0, index=dates)

    # ===== TOP PANEL: Cumulative Returns =====

    # Add period shading FIRST (behind lines)
    train_mask = dates <= train_end_date
    test_mask = dates > train_end_date

    if train_mask.any():
        ax1.axvspan(
            dates[train_mask].min(),
            train_end_date,
            alpha=0.15,
            color="green",
            label="_nolegend_",
        )
    if test_mask.any():
        ax1.axvspan(
            train_end_date,
            dates[test_mask].max(),
            alpha=0.15,
            color="blue",
            label="_nolegend_",
        )

    # Vertical line at train/test split
    ax1.axvline(
        train_end_date,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="_nolegend_",
    )

    # Plot lines
    ax1.plot(dates, strategy_returns, color="steelblue", linewidth=2.5, label="QV Strategy")
    ax1.plot(dates, bh_returns, color="gray", linewidth=2, linestyle="--", label="Buy & Hold SPY")
    ax1.plot(dates, no_trading, color="black", linewidth=1.5, linestyle=":", alpha=0.6, label="No Trading")

    # Zero line
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)

    # Period labels
    ax1.text(
        dates[train_mask].min() + (train_end_date - dates[train_mask].min()) / 2,
        ax1.get_ylim()[1] * 0.95,
        "TRAINING",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="darkgreen",
        alpha=0.7,
    )
    if test_mask.any():
        ax1.text(
            train_end_date + (dates[test_mask].max() - train_end_date) / 2,
            ax1.get_ylim()[1] * 0.95,
            "TESTING",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="darkblue",
            alpha=0.7,
        )

    ax1.set_title(
        "Volatility Arbitrage Strategy: Training vs Testing Performance",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax1.set_ylabel("Cumulative Return (%)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Stats box
    train_strategy = strategy_returns[train_mask].iloc[-1] if train_mask.any() else 0
    test_strategy = strategy_returns[test_mask].iloc[-1] - train_strategy if test_mask.any() else 0
    final_strategy = strategy_returns.iloc[-1]
    final_bh = bh_returns.iloc[-1]

    # Calculate Sharpe ratios
    strategy_daily = strategy_equity["equity"].pct_change().dropna()
    train_sharpe = (
        strategy_daily[train_mask[1:]].mean() / strategy_daily[train_mask[1:]].std() * np.sqrt(252)
        if train_mask.any() and strategy_daily[train_mask[1:]].std() > 0
        else 0
    )
    test_sharpe = (
        strategy_daily[test_mask[1:]].mean() / strategy_daily[test_mask[1:]].std() * np.sqrt(252)
        if test_mask.any() and len(strategy_daily[test_mask[1:]]) > 20 and strategy_daily[test_mask[1:]].std() > 0
        else 0
    )

    stats_text = (
        f"Strategy Final: {final_strategy:.1f}%\n"
        f"Buy & Hold Final: {final_bh:.1f}%\n"
        f"Train Sharpe: {train_sharpe:.2f}\n"
        f"Test Sharpe: {test_sharpe:.2f}"
    )
    ax1.text(
        0.02,
        0.02,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ===== BOTTOM PANEL: Drawdown =====

    # Strategy drawdown - calculate on EQUITY (not returns) to match backtest
    equity_values = strategy_equity["equity"]
    rolling_max_equity = equity_values.expanding().max()
    drawdown = (equity_values - rolling_max_equity) / rolling_max_equity * 100

    ax2.fill_between(dates, drawdown, 0, color="crimson", alpha=0.4, label="Strategy Drawdown")
    ax2.plot(dates, drawdown, color="darkred", linewidth=1.5)

    # Period shading
    if train_mask.any():
        ax2.axvspan(dates[train_mask].min(), train_end_date, alpha=0.1, color="green")
    if test_mask.any():
        ax2.axvspan(train_end_date, dates[test_mask].max(), alpha=0.1, color="blue")

    ax2.axvline(train_end_date, color="black", linestyle="--", linewidth=2, alpha=0.7)

    ax2.set_ylabel("Drawdown (%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Max drawdown annotation
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax2.annotate(
        f"Max DD: {max_dd_val:.1f}%",
        xy=(max_dd_idx, max_dd_val),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nChart saved to: {output_path}")
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Strategy Total Return:  {final_strategy:.1f}%")
    print(f"Buy & Hold Return:      {final_bh:.1f}%")
    print(f"Excess Return:          {final_strategy - final_bh:.1f}%")
    print(f"\nTrain Period Sharpe:    {train_sharpe:.2f}")
    print(f"Test Period Sharpe:     {test_sharpe:.2f}")
    print(f"Max Drawdown:           {max_dd_val:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate backtest comparison chart")
    parser.add_argument(
        "--config",
        type=str,
        default="config/volatility_arb.yaml",
        help="Strategy config file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="src/volatility_arbitrage/data/SPY_Options_2019_24",
        help="Options data directory",
    )
    parser.add_argument(
        "--ohlc",
        type=str,
        default="data/spy_ohlc.csv",
        help="SPY OHLC data for buy-and-hold",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2022-12-31",
        help="End date of training period",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/backtest_comparison.png",
        help="Output chart path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BACKTEST COMPARISON: Strategy vs Buy-and-Hold")
    print("=" * 60)

    # Load config
    print(f"\nLoading config from {args.config}...")
    config = load_strategy_config(Path(args.config))

    # Load options data
    print(f"Loading options data from {args.data}...")
    options_df = load_json_options_data(args.data)
    print(f"  Records: {len(options_df):,}")
    print(f"  Date range: {options_df['date'].min()} to {options_df['date'].max()}")

    # Load OHLC data
    print(f"Loading SPY OHLC from {args.ohlc}...")
    ohlc_df = load_spy_ohlc(args.ohlc)
    print(f"  Records: {len(ohlc_df):,}")

    # Run backtest
    print("\nRunning backtest...")
    results = run_qv_backtest(options_df, config, args.capital, config_path=args.config)

    # Generate comparison chart
    print("\nGenerating comparison chart...")
    plot_comparison(
        strategy_equity=results["equity_curve"],
        ohlc_df=ohlc_df,
        train_end=args.train_end,
        output_path=args.output,
        initial_capital=args.capital,
    )


if __name__ == "__main__":
    main()
