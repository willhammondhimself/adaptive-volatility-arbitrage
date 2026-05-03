# Data Audit Report: Real Options Data Integration

## Executive Summary

**Issue Identified**: The backtest engine was using hardcoded placeholder values for implied volatility (25%) and time-to-expiry (~29 days) instead of real historical options data.

**Root Cause**: Real options data (3.5GB) existed in the repository, and a proper data loader existed, but the backtest engine was never connected to use it.

**Resolution**: Connected the backtest engine to the existing real options data loader with proper IV lookup, TTE calculation, and data quality tracking.

## Audit Findings

### Data Assets (VERIFIED)

| Item | Status | Location |
|------|--------|----------|
| Options data files | Exists (3.5GB) | `src/volatility_arbitrage/data/SPY_Options_2019_24/` |
| IV + Greeks in data | Valid | `implied_volatility, delta, gamma, theta, vega` |
| Proper data loader | Exists | `src/volatility_arbitrage/data/real_options_loader.py` |
| Backtest uses loader | **Fixed** | `src/volatility_arbitrage/backtest/multi_asset_engine.py` |

### Data Coverage

| Year | File Size (Parquet) | Trading Days | Options/Day | IV Quality |
|------|---------------------|--------------|-------------|------------|
| 2019 | 67 MB | 252 | ~7,000 | 100% |
| 2020 | 70 MB | 252 | ~7,000 | 100% |
| 2021 | 75 MB | 252 | ~7,000 | 100% |
| 2024 | 64 MB | 252 | ~7,000 | 100% |

**Total**: ~276 MB (Parquet) / 3.5 GB (JSON)
**Compression ratio**: 92%

## Bug Location (FIXED)

**File**: `src/volatility_arbitrage/backtest/multi_asset_engine.py`
**Lines**: 419-424 (original)

### Before (Broken)
```python
# For now, use a simplified pricing model
# In production, would use actual option chain data
time_to_expiry = Decimal("0.08")  # ~1 month placeholder
implied_vol = Decimal("0.25")  # Placeholder
```

### After (Fixed)
```python
# Parse expiry date from symbol (format: YYYYMMDD)
expiry_date = datetime.strptime(expiry_str, "%Y%m%d")

# Calculate real time to expiry
time_to_expiry = Decimal(str((expiry_date - timestamp).days / 365.0))

# Get real IV from historical options data
if self.strategy_config.use_real_options_data:
    snapshot = get_snapshot_for_date(timestamp.date(), data_dir)
    # ... lookup logic with ATM IV fallback
```

## Validation: March 2020 Stress Test

The COVID crash (March 2020) provides excellent validation because:
- VIX spiked to 80+ (extreme volatility)
- Placeholder IV (25%) would be completely wrong
- Real data shows IV >80% on worst days

### Key Crisis Dates

| Date | SPY | ATM Put IV | ATM Call IV | Placeholder Would Show |
|------|-----|------------|-------------|----------------------|
| 2020-03-09 | $269 | 54.3% | 47.4% | 25% (wrong) |
| 2020-03-12 | $243 | 76.1% | 70.9% | 25% (wrong) |
| 2020-03-16 | $239 | 73.4% | 81.4% | 25% (wrong) |
| 2020-03-23 | $219 | 60.7% | 53.8% | 25% (wrong) |

### Stress Test Results

```
[PASS] Real data exists
[PASS] March 2020 data available (22 trading days)
[PASS] IV elevated during crisis (max 81.4%)
[PASS] IV different from placeholder (25%)
[PASS] 100% of options have valid IV
[PASS] Cache lookup fast (<1ms after initial load)
```

## Changes Made

### Phase 1: Data Optimization
1. **Created**: `scripts/convert_json_to_parquet.py`
   - Converts 3.5GB JSON to 276MB Parquet (92% compression)
   - Validates data quality during conversion

2. **Modified**: `src/volatility_arbitrage/data/real_options_loader.py`
   - Added Parquet file support (prefers Parquet, falls back to JSON)
   - Added in-memory caching for fast date lookups
   - Added `get_snapshot_for_date()` function
   - Added `clear_cache()` function

### Phase 2: Integration Fix
3. **Modified**: `src/volatility_arbitrage/core/config.py`
   - Added `use_real_options_data: bool = True`
   - Added `options_data_dir: str = "src/volatility_arbitrage/data/SPY_Options_2019_24"`

4. **Modified**: `src/volatility_arbitrage/backtest/multi_asset_engine.py`
   - Import `get_snapshot_for_date` from real_options_loader
   - Added `strategy_config` parameter to `__init__`
   - Added tracking: `trades_with_real_data`, `trades_with_placeholder_data`, `trades_skipped`
   - Replaced placeholder logic with real data lookup
   - Added `log_data_usage_report()` method
   - Added data quality metrics to results

### Phase 3: Validation
5. **Created**: `scripts/run_covid_stress_test.py`
   - Validates March 2020 data quality
   - Tests cache performance
   - Confirms IV elevation during crisis

## Data Quality Tracking

The backtest engine now reports data usage at completion:

```
==================================================
OPTIONS DATA USAGE REPORT
==================================================
  Trades with REAL IV data:       XXX
  Trades with PLACEHOLDER IV:       X
  Trades skipped (invalid):         X
  Real data coverage:           XX.X%
==================================================
```

**Acceptance Criteria**:
- >95% trades should use real data
- <5% placeholder usage acceptable (for edge cases)
- Data coverage included in backtest results

## Configuration

### Enable Real Data (Default)
```yaml
strategy:
  use_real_options_data: true
  options_data_dir: "src/volatility_arbitrage/data/SPY_Options_2019_24"
```

### Disable (Not Recommended)
```yaml
strategy:
  use_real_options_data: false  # Will log warning
```

## Performance

| Operation | Time |
|-----------|------|
| First year load (Parquet) | ~25s |
| Cache hit (date lookup) | <1ms |
| Full 4-year load | ~2min |

Memory usage: ~200MB per year cached (~800MB for all 4 years)

## Conclusion

The P0 bug has been fixed. The backtest engine now:
1. Uses real historical IV from options data
2. Calculates actual time-to-expiry from option symbols
3. Falls back to ATM IV when specific strike not found
4. Tracks and reports data quality metrics
5. Validates IV range (1-200%) before using

**Impact**: Backtest results now reflect realistic option pricing during actual market conditions, including stress periods like the COVID crash.
