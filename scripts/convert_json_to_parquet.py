"""
Convert SPY options JSON files to Parquet format for faster loading.

Converts the large JSON files (~900MB each) to compressed Parquet format,
which typically achieves 80%+ compression and much faster read times.

Usage:
    python scripts/convert_json_to_parquet.py

Output:
    - Parquet files in the same directory as source JSON
    - Quality report in data/conversion_quality_report.txt
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Required columns for options data
REQUIRED_COLUMNS = [
    "date", "symbol", "strike", "expiration", "type",
    "bid", "ask", "implied_volatility"
]

# All columns to keep (including Greeks)
KEEP_COLUMNS = [
    "contractID", "symbol", "expiration", "strike", "type",
    "last", "mark", "bid", "bid_size", "ask", "ask_size",
    "volume", "open_interest", "date",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho"
]


def convert_json_to_parquet(json_path: Path, output_dir: Path) -> dict:
    """
    Convert a single JSON file to Parquet format.

    Args:
        json_path: Path to source JSON file
        output_dir: Directory for output Parquet file

    Returns:
        dict with conversion statistics
    """
    stats = {
        "source_file": json_path.name,
        "source_size_mb": json_path.stat().st_size / (1024 * 1024),
        "success": False,
        "total_records": 0,
        "valid_iv_records": 0,
        "missing_iv_pct": 0.0,
        "output_size_mb": 0.0,
        "compression_ratio": 0.0,
    }

    logger.info(f"Processing {json_path.name} ({stats['source_size_mb']:.1f} MB)")

    try:
        # Load JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        # Flatten: data is list of days, each day is list of options
        all_records = []
        for day_options in data:
            if day_options:
                all_records.extend(day_options)

        stats["total_records"] = len(all_records)
        logger.info(f"  Total records: {stats['total_records']:,}")

        if not all_records:
            logger.warning(f"  No records found in {json_path.name}")
            return stats

        # Convert to DataFrame
        df = pd.DataFrame(all_records)

        # Keep only relevant columns (if they exist)
        available_cols = [c for c in KEEP_COLUMNS if c in df.columns]
        df = df[available_cols]

        # Validate required columns
        missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            logger.error(f"  Missing required columns: {missing_required}")
            return stats

        # Check IV quality (convert to numeric first since JSON may have strings)
        if "implied_volatility" in df.columns:
            iv_numeric = pd.to_numeric(df["implied_volatility"], errors="coerce")
            valid_iv = iv_numeric.notna() & (iv_numeric > 0)
            stats["valid_iv_records"] = int(valid_iv.sum())
            stats["missing_iv_pct"] = 100.0 * (1 - stats["valid_iv_records"] / len(df))
            logger.info(f"  Valid IV records: {stats['valid_iv_records']:,} ({100 - stats['missing_iv_pct']:.1f}%)")

        # Convert data types for efficient storage
        df = _optimize_dtypes(df)

        # Write to Parquet with snappy compression
        output_path = output_dir / json_path.name.replace(".json", ".parquet")
        df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

        stats["output_size_mb"] = output_path.stat().st_size / (1024 * 1024)
        stats["compression_ratio"] = 1 - (stats["output_size_mb"] / stats["source_size_mb"])
        stats["success"] = True

        logger.info(f"  Output: {output_path.name} ({stats['output_size_mb']:.1f} MB)")
        logger.info(f"  Compression: {stats['compression_ratio']*100:.1f}%")

    except Exception as e:
        logger.error(f"  Error processing {json_path.name}: {e}")
        stats["error"] = str(e)

    return stats


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame dtypes for Parquet storage."""
    # Date columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"])

    # Numeric columns - convert to appropriate types
    float_cols = ["strike", "last", "mark", "bid", "ask",
                  "implied_volatility", "delta", "gamma", "theta", "vega", "rho"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    int_cols = ["bid_size", "ask_size", "volume", "open_interest"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")

    # String columns - convert to category for compression
    cat_cols = ["symbol", "type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def write_quality_report(all_stats: list, output_path: Path):
    """Write conversion quality report."""
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SPY Options Data Conversion Quality Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        total_source = sum(s["source_size_mb"] for s in all_stats)
        total_output = sum(s["output_size_mb"] for s in all_stats if s["success"])
        total_records = sum(s["total_records"] for s in all_stats)
        total_valid_iv = sum(s["valid_iv_records"] for s in all_stats)

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Files processed: {len(all_stats)}\n")
        f.write(f"Successful: {sum(1 for s in all_stats if s['success'])}\n")
        f.write(f"Total source size: {total_source:.1f} MB\n")
        f.write(f"Total output size: {total_output:.1f} MB\n")
        f.write(f"Overall compression: {100*(1-total_output/total_source):.1f}%\n")
        f.write(f"Total records: {total_records:,}\n")
        f.write(f"Records with valid IV: {total_valid_iv:,} ({100*total_valid_iv/total_records:.1f}%)\n")
        f.write("\n")

        f.write("PER-FILE DETAILS\n")
        f.write("-" * 40 + "\n")
        for s in all_stats:
            status = "✅" if s["success"] else "❌"
            f.write(f"\n{status} {s['source_file']}\n")
            f.write(f"   Source: {s['source_size_mb']:.1f} MB\n")
            f.write(f"   Output: {s['output_size_mb']:.1f} MB\n")
            f.write(f"   Records: {s['total_records']:,}\n")
            f.write(f"   Valid IV: {100 - s['missing_iv_pct']:.1f}%\n")
            if s["missing_iv_pct"] > 10:
                f.write(f"   ⚠️ LOW IV QUALITY - {s['missing_iv_pct']:.1f}% missing\n")
            if "error" in s:
                f.write(f"   Error: {s['error']}\n")

    logger.info(f"Quality report written to {output_path}")


def main():
    # Find JSON files
    data_dir = Path("src/volatility_arbitrage/data/SPY_Options_2019_24")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(json_files)} JSON files to convert")

    # Convert each file
    all_stats = []
    for json_path in json_files:
        stats = convert_json_to_parquet(json_path, data_dir)
        all_stats.append(stats)

    # Write quality report
    report_dir = Path("data")
    report_dir.mkdir(exist_ok=True)
    write_quality_report(all_stats, report_dir / "conversion_quality_report.txt")

    # Summary
    successful = sum(1 for s in all_stats if s["success"])
    logger.info(f"\n{'='*40}")
    logger.info(f"Conversion complete: {successful}/{len(all_stats)} files successful")

    if successful < len(all_stats):
        sys.exit(1)


if __name__ == "__main__":
    main()
