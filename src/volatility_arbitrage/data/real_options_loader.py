"""
Real Options Data Loader

Loads historical SPY options data from JSON files and converts to OptionChain format
for use in backtesting with the QV strategy.

Data source: SPY_Options_2019_24/ directory with daily options snapshots
Format: List of days, each day is a list of option contracts with:
    - strike, expiration, type (call/put)
    - bid, ask, mark, volume, open_interest
    - implied_volatility, delta, gamma, theta, vega
"""

import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

from volatility_arbitrage.core.types import OptionChain, OptionContract, OptionType

logger = logging.getLogger(__name__)


@dataclass
class DailyOptionsSnapshot:
    """Aggregated daily options data for QV strategy signals."""
    date: datetime
    underlying_price: Decimal

    # Put/Call metrics
    put_volume: int
    call_volume: int
    put_oi: int
    call_oi: int
    pc_ratio: Decimal  # Put/Call volume ratio

    # IV metrics (ATM options)
    atm_put_iv: Decimal
    atm_call_iv: Decimal
    iv_skew: Decimal  # OTM Put IV - OTM Call IV

    # Term structure (front month vs back month ATM IV)
    front_month_iv: Decimal
    back_month_iv: Decimal
    term_slope: Decimal

    # Option chains by expiry
    chains: Dict[datetime, OptionChain]


def load_spy_options_year(year: int, data_dir: str) -> List[DailyOptionsSnapshot]:
    """
    Load SPY options data for a specific year.

    Args:
        year: Year to load (2019-2024)
        data_dir: Path to SPY_Options_2019_24 directory

    Returns:
        List of daily options snapshots
    """
    filename = f"spy_options_data_{str(year)[-2:]}.json"
    filepath = Path(data_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Options data file not found: {filepath}")

    logger.info(f"Loading options data from {filepath}")

    with open(filepath, 'r') as f:
        raw_data = json.load(f)

    snapshots = []
    for day_options in raw_data:
        if not day_options:
            continue

        snapshot = _process_day_options(day_options)
        if snapshot:
            snapshots.append(snapshot)

    logger.info(f"Loaded {len(snapshots)} daily snapshots for {year}")
    return snapshots


def load_spy_options_range(
    start_year: int,
    end_year: int,
    data_dir: str
) -> List[DailyOptionsSnapshot]:
    """
    Load SPY options data for a range of years.

    Args:
        start_year: Starting year (inclusive)
        end_year: Ending year (inclusive)
        data_dir: Path to SPY_Options_2019_24 directory

    Returns:
        List of daily options snapshots, sorted by date
    """
    all_snapshots = []

    for year in range(start_year, end_year + 1):
        try:
            year_snapshots = load_spy_options_year(year, data_dir)
            all_snapshots.extend(year_snapshots)
        except FileNotFoundError as e:
            logger.warning(f"Skipping year {year}: {e}")

    # Sort by date
    all_snapshots.sort(key=lambda x: x.date)

    logger.info(f"Total snapshots loaded: {len(all_snapshots)} ({start_year}-{end_year})")
    return all_snapshots


def _process_day_options(day_options: List[dict]) -> Optional[DailyOptionsSnapshot]:
    """
    Process a day's options data into aggregated metrics.

    Args:
        day_options: List of option contracts for a single day

    Returns:
        DailyOptionsSnapshot or None if insufficient data
    """
    if not day_options:
        return None

    # Get date and estimate underlying price from ATM options
    date_str = day_options[0]['date']
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Organize by expiration and type
    by_expiry: Dict[str, Dict[str, List[dict]]] = {}

    put_volume = 0
    call_volume = 0
    put_oi = 0
    call_oi = 0

    for opt in day_options:
        expiry = opt['expiration']
        opt_type = opt['type'].lower()  # Normalize to lowercase

        if expiry not in by_expiry:
            by_expiry[expiry] = {'call': [], 'put': []}

        by_expiry[expiry][opt_type].append(opt)

        # Aggregate volume/OI
        vol = int(opt.get('volume', 0) or 0)
        oi = int(opt.get('open_interest', 0) or 0)

        if opt_type == 'put':
            put_volume += vol
            call_oi += oi
        else:
            call_volume += vol
            put_oi += oi

    # Estimate underlying from highest-volume ATM strike
    underlying_price = _estimate_underlying_price(day_options)
    if underlying_price is None:
        return None

    # Calculate P/C ratio
    pc_ratio = Decimal(str(put_volume / max(call_volume, 1)))

    # Find front and back month expirations (targeting 30 and 60 DTE)
    sorted_expiries = sorted(by_expiry.keys())
    front_expiry, back_expiry = _find_target_expiries(date, sorted_expiries, target_dte=[30, 60])

    if not front_expiry:
        return None

    # Calculate ATM IVs
    atm_put_iv, atm_call_iv = _get_atm_ivs(
        by_expiry.get(front_expiry, {'call': [], 'put': []}),
        underlying_price
    )

    # Calculate IV skew (25-delta put IV - 25-delta call IV)
    iv_skew = _calculate_iv_skew(
        by_expiry.get(front_expiry, {'call': [], 'put': []}),
        underlying_price
    )

    # Calculate term structure
    front_month_iv = atm_call_iv  # Use call IV as reference
    back_month_iv = Decimal("0.20")  # Default

    if back_expiry and back_expiry in by_expiry:
        _, back_call_iv = _get_atm_ivs(by_expiry[back_expiry], underlying_price)
        if back_call_iv > 0:
            back_month_iv = back_call_iv

    term_slope = back_month_iv - front_month_iv if front_month_iv > 0 else Decimal("0")

    # Build option chains for each expiry
    chains = _build_option_chains(date, by_expiry, underlying_price)

    return DailyOptionsSnapshot(
        date=date,
        underlying_price=underlying_price,
        put_volume=put_volume,
        call_volume=call_volume,
        put_oi=put_oi,
        call_oi=call_oi,
        pc_ratio=pc_ratio,
        atm_put_iv=atm_put_iv,
        atm_call_iv=atm_call_iv,
        iv_skew=iv_skew,
        front_month_iv=front_month_iv,
        back_month_iv=back_month_iv,
        term_slope=term_slope,
        chains=chains,
    )


def _estimate_underlying_price(day_options: List[dict]) -> Optional[Decimal]:
    """Estimate underlying price from ATM option strikes."""
    # Find strikes where put and call have similar prices (ATM)
    by_strike: Dict[float, dict] = {}

    for opt in day_options:
        strike = float(opt['strike'])
        if strike not in by_strike:
            by_strike[strike] = {'call': None, 'put': None}
        by_strike[strike][opt['type']] = opt

    # Find strike where |call_price - put_price| is minimized (put-call parity)
    best_strike = None
    min_diff = float('inf')

    for strike, opts in by_strike.items():
        if opts['call'] and opts['put']:
            call_mark = float(opts['call'].get('mark', 0) or 0)
            put_mark = float(opts['put'].get('mark', 0) or 0)

            if call_mark > 0 and put_mark > 0:
                # For ATM, call ≈ put, so look for smallest difference
                diff = abs(call_mark - put_mark)
                if diff < min_diff:
                    min_diff = diff
                    best_strike = strike

    if best_strike:
        return Decimal(str(best_strike))

    # Fallback: use median strike
    strikes = sorted(by_strike.keys())
    if strikes:
        return Decimal(str(strikes[len(strikes) // 2]))

    return None


def _find_target_expiries(
    date: datetime,
    expiries: List[str],
    target_dte: List[int] = [30, 60]
) -> Tuple[Optional[str], Optional[str]]:
    """Find expiries closest to target DTEs."""
    result = [None, None]

    for i, target in enumerate(target_dte):
        best_expiry = None
        best_diff = float('inf')

        for exp_str in expiries:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
            dte = (exp_date - date).days

            if dte > 0:  # Must be in the future
                diff = abs(dte - target)
                if diff < best_diff:
                    best_diff = diff
                    best_expiry = exp_str

        result[i] = best_expiry

    return tuple(result)


def _get_atm_ivs(
    expiry_options: Dict[str, List[dict]],
    underlying: Decimal
) -> Tuple[Decimal, Decimal]:
    """Get ATM put and call implied volatilities."""
    underlying_float = float(underlying)

    def find_atm_iv(options: List[dict]) -> Decimal:
        if not options:
            return Decimal("0.20")  # Default

        # Find option closest to ATM
        best_opt = None
        min_dist = float('inf')

        for opt in options:
            strike = float(opt['strike'])
            dist = abs(strike - underlying_float)
            iv = float(opt.get('implied_volatility', 0) or 0)

            if dist < min_dist and iv > 0:
                min_dist = dist
                best_opt = opt

        if best_opt:
            return Decimal(str(best_opt['implied_volatility']))
        return Decimal("0.20")

    put_iv = find_atm_iv(expiry_options.get('put', []))
    call_iv = find_atm_iv(expiry_options.get('call', []))

    return put_iv, call_iv


def _calculate_iv_skew(
    expiry_options: Dict[str, List[dict]],
    underlying: Decimal
) -> Decimal:
    """
    Calculate IV skew as OTM put IV (25-delta) minus OTM call IV (25-delta).

    Positive skew = puts more expensive (bearish sentiment)
    Negative skew = calls more expensive (bullish sentiment)
    """
    underlying_float = float(underlying)

    def find_delta_iv(options: List[dict], target_delta: float) -> Optional[float]:
        """Find IV for option closest to target delta."""
        best_opt = None
        min_diff = float('inf')

        for opt in options:
            delta = abs(float(opt.get('delta', 0) or 0))
            iv = float(opt.get('implied_volatility', 0) or 0)

            if iv > 0:
                diff = abs(delta - target_delta)
                if diff < min_diff:
                    min_diff = diff
                    best_opt = opt

        if best_opt and min_diff < 0.15:  # Within 15 delta points
            return float(best_opt['implied_volatility'])
        return None

    # 25-delta put (OTM)
    put_iv = find_delta_iv(expiry_options.get('put', []), 0.25)

    # 25-delta call (OTM)
    call_iv = find_delta_iv(expiry_options.get('call', []), 0.25)

    if put_iv and call_iv:
        return Decimal(str(put_iv - call_iv))

    return Decimal("0")


def _build_option_chains(
    date: datetime,
    by_expiry: Dict[str, Dict[str, List[dict]]],
    underlying: Decimal
) -> Dict[datetime, OptionChain]:
    """Build OptionChain objects for each expiration."""
    chains = {}

    for exp_str, opts_by_type in by_expiry.items():
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')

        # Skip expired options
        if exp_date <= date:
            continue

        # Build option contracts
        calls = []
        puts = []

        for opt in opts_by_type.get('call', []):
            contract = _build_option_contract(opt, 'call', underlying, exp_date)
            if contract:
                calls.append(contract)

        for opt in opts_by_type.get('put', []):
            contract = _build_option_contract(opt, 'put', underlying, exp_date)
            if contract:
                puts.append(contract)

        if calls or puts:
            chain = OptionChain(
                symbol='SPY',
                timestamp=date,
                expiry=exp_date,
                underlying_price=underlying,
                calls=calls,
                puts=puts,
            )
            chains[exp_date] = chain

    return chains


def _build_option_contract(
    opt: dict,
    opt_type: str,
    underlying: Decimal,
    expiry: datetime
) -> Optional[OptionContract]:
    """Build an OptionContract from raw data."""
    try:
        strike = Decimal(str(opt['strike']))
        iv = Decimal(str(opt.get('implied_volatility', 0) or 0))

        if iv <= 0:
            return None

        # Get price (prefer mark, fallback to mid)
        mark = float(opt.get('mark', 0) or 0)
        bid = float(opt.get('bid', 0) or 0)
        ask = float(opt.get('ask', 0) or 0)

        if mark > 0:
            price = Decimal(str(mark))
        elif bid > 0 and ask > 0:
            price = Decimal(str((bid + ask) / 2))
        else:
            return None

        # Map option type string to OptionType enum
        option_type = OptionType.CALL if opt_type == 'call' else OptionType.PUT

        return OptionContract(
            symbol='SPY',
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            price=price,
            implied_volatility=iv,
            volume=int(opt.get('volume', 0) or 0),
            open_interest=int(opt.get('open_interest', 0) or 0),
            bid=Decimal(str(bid)) if bid > 0 else None,
            ask=Decimal(str(ask)) if ask > 0 else None,
        )
    except (ValueError, KeyError) as e:
        return None


def snapshots_to_dataframe(snapshots: List[DailyOptionsSnapshot]) -> pd.DataFrame:
    """
    Convert snapshots to a DataFrame suitable for walk-forward optimization.

    Returns DataFrame with columns matching the expected format:
    - date (index)
    - close (underlying price)
    - vix (derived from ATM IV)
    - Plus option chain metadata
    """
    records = []

    for snap in snapshots:
        # Convert ATM IV to annualized "VIX-like" number
        # VIX is typically ATM 30-day IV * 100
        vix_approx = float(snap.front_month_iv) * 100

        records.append({
            'date': snap.date,
            'close': float(snap.underlying_price),
            'vix': vix_approx,
            'pc_ratio': float(snap.pc_ratio),
            'put_volume': snap.put_volume,
            'call_volume': snap.call_volume,
            'iv_skew': float(snap.iv_skew),
            'term_slope': float(snap.term_slope),
            'atm_put_iv': float(snap.atm_put_iv),
            'atm_call_iv': float(snap.atm_call_iv),
        })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df


def get_option_chain_for_date(
    snapshots: List[DailyOptionsSnapshot],
    target_date: datetime,
    target_dte: int = 52
) -> Optional[OptionChain]:
    """
    Get the option chain closest to target DTE for a specific date.

    Args:
        snapshots: List of daily snapshots
        target_date: Date to find
        target_dte: Target days to expiry (default 52 for tiered sizing)

    Returns:
        OptionChain or None
    """
    # Find snapshot for target date
    for snap in snapshots:
        if snap.date.date() == target_date.date():
            # Find expiry closest to target DTE
            best_chain = None
            best_diff = float('inf')

            for exp_date, chain in snap.chains.items():
                dte = (exp_date - snap.date).days
                diff = abs(dte - target_dte)

                if diff < best_diff and dte > 0:
                    best_diff = diff
                    best_chain = chain

            return best_chain

    return None


if __name__ == "__main__":
    # Test loader
    import sys

    data_dir = "/Users/willhammond/Adaptive Volatility Arbitrage Backtesting Engine/data/SPY_Options_2019_24"

    print("Loading 2019 data...")
    snapshots = load_spy_options_year(2019, data_dir)

    print(f"\nFirst snapshot: {snapshots[0].date}")
    print(f"  Underlying: ${snapshots[0].underlying_price}")
    print(f"  P/C Ratio: {snapshots[0].pc_ratio:.2f}")
    print(f"  ATM Put IV: {snapshots[0].atm_put_iv:.2%}")
    print(f"  ATM Call IV: {snapshots[0].atm_call_iv:.2%}")
    print(f"  IV Skew: {snapshots[0].iv_skew:.4f}")
    print(f"  Term Slope: {snapshots[0].term_slope:.4f}")
    print(f"  Chains available: {len(snapshots[0].chains)} expirations")

    # Convert to DataFrame
    df = snapshots_to_dataframe(snapshots)
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())
