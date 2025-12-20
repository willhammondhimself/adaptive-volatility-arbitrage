#!/usr/bin/env python3
"""
Run backtest using MultiAssetBacktestEngine with VolatilityArbitrageStrategy.

Demonstrates Phase 2 features:
- BayesianLSTMForecaster for volatility forecasting
- SquareRootImpactModel for transaction costs
- UncertaintySizer for position sizing
"""

import argparse
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from volatility_arbitrage.backtest.multi_asset_engine import MultiAssetBacktestEngine
from volatility_arbitrage.core.config import BacktestConfig
from volatility_arbitrage.core.types import OptionChain, OptionContract, OptionType
from volatility_arbitrage.strategy.volatility_arbitrage import (
    VolatilityArbitrageStrategy,
    VolatilityArbitrageConfig,
)


def load_json_options_data(data_dir: str) -> pd.DataFrame:
    """Load JSON options data from directory."""
    data_path = Path(data_dir)
    all_records = []

    json_files = sorted(data_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        print(f"Loading {json_file.name}...")
        with open(json_file, "r") as f:
            data = json.load(f)

        # Handle nested list structure
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                for day_records in data:
                    all_records.extend(day_records)
            else:
                all_records.extend(data)

    print(f"Total records loaded: {len(all_records):,}")

    df = pd.DataFrame(all_records)

    # Convert types
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    for col in [
        "strike",
        "last",
        "bid",
        "ask",
        "mark",
        "volume",
        "open_interest",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_option_chain(day_df: pd.DataFrame, date: datetime, underlying_price: float) -> OptionChain:
    """
    Build an OptionChain object from daily options data.

    Args:
        day_df: DataFrame with options data for a single day
        date: The trading date
        underlying_price: Estimated underlying price

    Returns:
        OptionChain object with all contracts
    """
    # Find the ~30-day expiry (25-35 DTE)
    day_df = day_df.copy()
    day_df["dte"] = (day_df["expiration"] - date).dt.days
    target_df = day_df[(day_df["dte"] >= 25) & (day_df["dte"] <= 35)]

    if target_df.empty:
        # Fallback to closest expiry
        target_df = day_df[(day_df["dte"] >= 14) & (day_df["dte"] <= 60)]

    if target_df.empty:
        target_df = day_df

    # Get the most common expiry in range
    if not target_df.empty:
        expiry = target_df["expiration"].mode().iloc[0] if len(target_df["expiration"].mode()) > 0 else target_df["expiration"].iloc[0]
        target_df = target_df[target_df["expiration"] == expiry]

    # Build call contracts
    calls = []
    call_df = target_df[target_df["type"] == "call"]
    for _, row in call_df.iterrows():
        try:
            contract = OptionContract(
                symbol=row.get("symbol", "SPY"),
                option_type=OptionType.CALL,
                strike=Decimal(str(row["strike"])),
                expiry=row["expiration"],
                price=Decimal(str(max(0.01, row.get("mark", row.get("last", 0.01))))),
                bid=Decimal(str(max(0, row.get("bid", 0)))) if pd.notna(row.get("bid")) else None,
                ask=Decimal(str(max(0, row.get("ask", 0)))) if pd.notna(row.get("ask")) else None,
                volume=int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                open_interest=int(row.get("open_interest", 0)) if pd.notna(row.get("open_interest")) else 0,
                implied_volatility=Decimal(str(row["implied_volatility"])) if pd.notna(row.get("implied_volatility")) else None,
            )
            calls.append(contract)
        except Exception:
            continue

    # Build put contracts
    puts = []
    put_df = target_df[target_df["type"] == "put"]
    for _, row in put_df.iterrows():
        try:
            contract = OptionContract(
                symbol=row.get("symbol", "SPY"),
                option_type=OptionType.PUT,
                strike=Decimal(str(row["strike"])),
                expiry=row["expiration"],
                price=Decimal(str(max(0.01, row.get("mark", row.get("last", 0.01))))),
                bid=Decimal(str(max(0, row.get("bid", 0)))) if pd.notna(row.get("bid")) else None,
                ask=Decimal(str(max(0, row.get("ask", 0)))) if pd.notna(row.get("ask")) else None,
                volume=int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                open_interest=int(row.get("open_interest", 0)) if pd.notna(row.get("open_interest")) else 0,
                implied_volatility=Decimal(str(row["implied_volatility"])) if pd.notna(row.get("implied_volatility")) else None,
            )
            puts.append(contract)
        except Exception:
            continue

    # Get expiry for the chain
    chain_expiry = expiry if not target_df.empty else date + pd.Timedelta(days=30)

    return OptionChain(
        symbol="SPY",
        timestamp=date,
        expiry=chain_expiry,
        underlying_price=Decimal(str(underlying_price)),
        calls=calls,
        puts=puts,
        risk_free_rate=Decimal("0.05"),
    )


def prepare_market_data(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare market data in the format expected by VolatilityArbitrageStrategy.

    The strategy expects:
    - symbol column
    - close column (underlying price)
    - option_chain column (OptionChain object)
    """
    # Get unique dates
    dates = sorted(options_df["date"].unique())

    records = []
    for date in dates:
        day_df = options_df[options_df["date"] == date]

        # Get underlying symbol
        symbol = day_df["symbol"].iloc[0] if "symbol" in day_df.columns else "SPY"

        # Estimate underlying price from ATM strikes
        calls = day_df[day_df["type"] == "call"]
        if len(calls) > 0:
            # Find ATM strike (closest to 0.5 delta)
            atm_calls = calls[calls["delta"].between(0.4, 0.6)]
            if len(atm_calls) > 0:
                underlying_price = float(atm_calls["strike"].median())
            else:
                underlying_price = float(calls["strike"].median())
        else:
            underlying_price = 400.0

        # Build option chain
        option_chain = build_option_chain(day_df, date, underlying_price)

        # Total volume
        total_volume = int(day_df["volume"].sum())

        records.append(
            {
                "date": date,
                "symbol": symbol,
                "open": underlying_price,
                "high": underlying_price * 1.005,
                "low": underlying_price * 0.995,
                "close": underlying_price,
                "volume": total_volume,
                "option_chain": option_chain,
            }
        )

    market_df = pd.DataFrame(records)
    market_df = market_df.sort_values("date").reset_index(drop=True)

    return market_df


def run_integrated_backtest(
    data_dir: str,
    initial_capital: float = 100000.0,
    use_phase2_features: bool = True,
    max_days: int = None,
) -> dict:
    """
    Run backtest using MultiAssetBacktestEngine with VolatilityArbitrageStrategy.

    Args:
        data_dir: Directory containing JSON options data
        initial_capital: Starting capital
        use_phase2_features: Enable Phase 2 features (Bayesian LSTM, impact model, etc.)
        max_days: Maximum number of days to run (None for all data)

    Returns:
        Backtest results dictionary
    """
    print("\n" + "=" * 60)
    print("INTEGRATED BACKTEST: MultiAssetBacktestEngine + VolatilityArbitrageStrategy")
    print("=" * 60)

    # Load options data
    print("\n[1/5] Loading options data...")
    options_df = load_json_options_data(data_dir)

    # Prepare market data with option chains
    print("\n[2/5] Preparing market data with option chains...")
    market_df = prepare_market_data(options_df)

    if max_days:
        market_df = market_df.head(max_days)

    print(f"Market data: {len(market_df)} days from {market_df['date'].min()} to {market_df['date'].max()}")

    # Sample option chain stats
    sample_chain = market_df.iloc[0]["option_chain"]
    print(f"Sample option chain: {len(sample_chain.calls)} calls, {len(sample_chain.puts)} puts")

    # Create BacktestConfig with Phase 2 settings
    print("\n[3/5] Creating backtest configuration...")
    backtest_config = BacktestConfig(
        initial_capital=Decimal(str(initial_capital)),
        commission_rate=Decimal("0.001"),
        slippage=Decimal("0.001"),
        option_spread=Decimal("0.05"),
        option_commission_per_contract=Decimal("0.65"),
        daily_hedge_cost=Decimal("0.0002"),
        margin_rate=Decimal("0.05"),
        position_size_pct=Decimal("0.05"),
        max_positions=5,
        risk_free_rate=Decimal("0.05"),
        # Phase 2: Square-Root Impact Model
        use_impact_model=use_phase2_features,
        impact_half_spread_bps=Decimal("5.0"),
        impact_coefficient=Decimal("0.1"),
    )

    print(f"  - Initial capital: ${float(backtest_config.initial_capital):,.2f}")
    print(f"  - Use impact model: {backtest_config.use_impact_model}")
    if use_phase2_features:
        print(f"  - Impact half-spread: {float(backtest_config.impact_half_spread_bps):.1f} bps")
        print(f"  - Impact coefficient: {float(backtest_config.impact_coefficient):.2f}")

    # Create VolatilityArbitrageConfig with Phase 2 settings
    print("\n[4/5] Creating strategy configuration...")
    strategy_config = VolatilityArbitrageConfig(
        # Entry/exit thresholds
        entry_threshold_pct=Decimal("5.0"),
        exit_threshold_pct=Decimal("2.0"),
        # Time constraints
        min_days_to_expiry=14,
        max_days_to_expiry=60,
        # Position sizing
        # NOTE: For SPY at ~$285, each ATM contract has ~$12,800 delta exposure
        # Need at least 13% to get 1 contract with $100k capital
        position_size_pct=Decimal("15.0"),
        max_vega_exposure=Decimal("1000"),
        max_positions=5,
        # Enable QV strategy
        use_qv_strategy=True,
        consensus_threshold=Decimal("0.15"),
        # Base long bias = 0 for pure vol arb
        base_long_bias=Decimal("0.0"),
        # Disable tiered sizing for simpler position calculation
        use_tiered_sizing=False,
        # Phase 2: Volatility forecasting method
        vol_forecast_method="bayesian_lstm" if use_phase2_features else "garch",
        bayesian_lstm_hidden_size=64,
        bayesian_lstm_dropout_p=0.2,
        bayesian_lstm_sequence_length=20,
        bayesian_lstm_n_mc_samples=50,
        # Phase 2: Uncertainty-adjusted position sizing
        # TEMPORARILY DISABLED for debugging - uncertainty too high with untrained model
        use_uncertainty_sizing=False,  # was: use_phase2_features
        uncertainty_penalty=2.0,
        uncertainty_min_position_pct=0.01,
        uncertainty_max_position_pct=0.15,
        # Leverage (Phase 2)
        use_leverage=use_phase2_features,
        short_vol_leverage=Decimal("1.15"),
        long_vol_leverage=Decimal("1.5"),
    )

    print(f"  - QV Strategy: {strategy_config.use_qv_strategy}")
    print(f"  - Vol forecast method: {strategy_config.vol_forecast_method}")
    print(f"  - Uncertainty sizing: {strategy_config.use_uncertainty_sizing}")
    print(f"  - Leverage enabled: {strategy_config.use_leverage}")

    # Create strategy
    strategy = VolatilityArbitrageStrategy(config=strategy_config)
    print(f"  - Strategy created: {type(strategy).__name__}")

    # Create backtest engine
    print("\n[5/5] Running backtest...")
    engine = MultiAssetBacktestEngine(
        config=backtest_config,
        strategy=strategy,
        option_commission_per_contract=Decimal("0.65"),
        option_slippage_pct=Decimal("0.025"),
        margin_requirement_pct=Decimal("0.25"),
    )

    # Get date range
    start_date = market_df["date"].min()
    end_date = market_df["date"].max()

    print(f"  - Start date: {start_date}")
    print(f"  - End date: {end_date}")

    # Run backtest day by day
    strategy.on_backtest_start(start_date, end_date)

    for idx, row in market_df.iterrows():
        timestamp = row["date"]
        # Pass single-row DataFrame with option_chain
        day_data = market_df[market_df["date"] == timestamp].copy()

        # Process the day
        engine._process_day(timestamp, day_data)

        # Progress update every 100 days
        if (idx + 1) % 100 == 0:
            equity = engine.history[-1]["equity"] if engine.history else initial_capital
            print(f"  - Day {idx + 1} / {len(market_df)}: Equity ${equity:,.2f}")

    strategy.on_backtest_end()

    # Get results
    results = engine.get_results()

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return: {results['total_return'] * 100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
    print(f"Total Trades: {results['total_trades']}")

    if results["equity_curve"]:
        final_equity = results["equity_curve"][-1]["equity"]
        print(f"Final Equity: ${final_equity:,.2f}")

    # Print Phase 2 feature status
    print("\n" + "-" * 40)
    print("Phase 2 Features Status:")
    print("-" * 40)
    print(f"  - SquareRootImpactModel: {'ACTIVE' if engine.cost_model else 'INACTIVE'}")
    print(f"  - BayesianLSTMForecaster: {'ACTIVE' if strategy_config.vol_forecast_method == 'bayesian_lstm' else 'INACTIVE'}")
    print(f"  - UncertaintySizer: {'ACTIVE' if strategy.uncertainty_sizer else 'INACTIVE'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run integrated backtest")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="src/volatility_arbitrage/data/SPY_Options_2019_24",
        help="Directory containing JSON options data",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--no-phase2",
        action="store_true",
        help="Disable Phase 2 features (for comparison)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Maximum number of days to run (for quick testing)",
    )

    args = parser.parse_args()

    results = run_integrated_backtest(
        data_dir=args.data_dir,
        initial_capital=args.capital,
        use_phase2_features=not args.no_phase2,
        max_days=args.max_days,
    )


if __name__ == "__main__":
    main()
