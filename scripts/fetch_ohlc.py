"""
Fetch SPY OHLC data from yfinance for alternative RV estimators.

Usage:
    python scripts/fetch_ohlc.py
    python scripts/fetch_ohlc.py --start 2019-01-01 --end 2024-12-31
"""

import argparse
from pathlib import Path
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance required: pip install yfinance")


def fetch_spy_ohlc(
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    output_path: str = None
) -> pd.DataFrame:
    """
    Fetch SPY OHLC data from Yahoo Finance.

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        output_path: Where to save CSV (default: data/spy_ohlc.csv)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    print(f"Fetching SPY OHLC from {start} to {end}...")

    spy = yf.Ticker("SPY")
    df = spy.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError("No data returned from yfinance")

    # Reset index to make Date a column
    df = df.reset_index()

    # Rename columns for consistency
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Convert timezone-aware datetime to date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Keep only needed columns
    df = df[["date", "open", "high", "low", "close", "volume"]]

    print(f"Retrieved {len(df)} days of data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Save if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")

    return df


def load_ohlc(path: str = "data/spy_ohlc.csv") -> pd.DataFrame:
    """
    Load OHLC data from CSV.

    Returns:
        DataFrame with date index and OHLC columns
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch SPY OHLC data")
    parser.add_argument("--start", default="2019-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--output", default="data/spy_ohlc.csv", help="Output CSV path")
    args = parser.parse_args()

    fetch_spy_ohlc(args.start, args.end, args.output)
