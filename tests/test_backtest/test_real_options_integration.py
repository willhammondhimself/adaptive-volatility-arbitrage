"""
Integration tests for real options data in backtest engine.

Tests that the backtest engine correctly loads and uses historical
options data instead of hardcoded placeholders.
"""

import pytest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from volatility_arbitrage.core.config import VolatilityArbitrageConfig
from volatility_arbitrage.data.real_options_loader import (
    get_snapshot_for_date,
    load_spy_options_year,
    clear_cache,
    DailyOptionsSnapshot,
)


# Skip all tests if data files don't exist
DATA_DIR = "src/volatility_arbitrage/data/SPY_Options_2019_24"
DATA_EXISTS = Path(DATA_DIR).exists() and any(
    Path(DATA_DIR).glob("*.parquet")
) or any(Path(DATA_DIR).glob("*.json"))


@pytest.fixture(autouse=True)
def clear_data_cache():
    """Clear cache before each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.mark.skipif(not DATA_EXISTS, reason="Options data files not found")
class TestRealOptionsLoader:
    """Tests for the real options data loader."""

    def test_load_year_returns_snapshots(self):
        """Test that loading a year returns daily snapshots."""
        snapshots = load_spy_options_year(2020, DATA_DIR)

        assert len(snapshots) > 200  # Should have ~252 trading days
        assert all(isinstance(s, DailyOptionsSnapshot) for s in snapshots)

    def test_snapshot_has_required_fields(self):
        """Test that snapshots have all required fields."""
        snapshots = load_spy_options_year(2020, DATA_DIR)
        snap = snapshots[0]

        # Check required fields
        assert snap.date is not None
        assert snap.underlying_price > 0
        assert snap.atm_put_iv > 0
        assert snap.atm_call_iv > 0
        assert len(snap.chains) > 0

    def test_get_snapshot_for_date(self):
        """Test fast lookup by date."""
        # First call loads the year
        snap = get_snapshot_for_date(date(2020, 3, 16), DATA_DIR)

        assert snap is not None
        assert snap.date.date() == date(2020, 3, 16)

        # Verify it's the COVID crash data (IV should be elevated)
        assert float(snap.atm_call_iv) > 0.50  # >50% IV

    def test_cache_works(self):
        """Test that caching prevents redundant loads."""
        # First call - loads data
        snap1 = get_snapshot_for_date(date(2020, 3, 16), DATA_DIR)

        # Second call - should use cache
        snap2 = get_snapshot_for_date(date(2020, 3, 16), DATA_DIR)

        assert snap1 is snap2  # Same object from cache

    def test_missing_date_returns_none(self):
        """Test that missing dates return None gracefully."""
        # Weekend date (no trading)
        snap = get_snapshot_for_date(date(2020, 3, 15), DATA_DIR)  # Sunday
        assert snap is None

    def test_option_chains_have_valid_iv(self):
        """Test that option chains contain valid IV data."""
        snap = get_snapshot_for_date(date(2020, 3, 16), DATA_DIR)
        assert snap is not None

        for chain in snap.chains.values():
            # Check some calls have valid IV
            valid_calls = [c for c in chain.calls if c.implied_volatility and c.implied_volatility > 0]
            assert len(valid_calls) > 0, "No calls with valid IV"

            # Check some puts have valid IV
            valid_puts = [p for p in chain.puts if p.implied_volatility and p.implied_volatility > 0]
            assert len(valid_puts) > 0, "No puts with valid IV"


@pytest.mark.skipif(not DATA_EXISTS, reason="Options data files not found")
class TestMarch2020CrisisData:
    """Tests specifically for March 2020 COVID crash data quality."""

    def test_iv_elevated_during_crisis(self):
        """Test that IV is significantly elevated during COVID crash."""
        crisis_dates = [
            date(2020, 3, 9),   # Black Monday
            date(2020, 3, 12),  # Thursday crash
            date(2020, 3, 16),  # Worst single day
        ]

        for d in crisis_dates:
            snap = get_snapshot_for_date(d, DATA_DIR)
            if snap:
                # IV should be way above the old placeholder of 25%
                assert float(snap.atm_put_iv) > 0.40, f"Put IV too low on {d}"
                assert float(snap.atm_call_iv) > 0.40, f"Call IV too low on {d}"

    def test_iv_differs_from_placeholder(self):
        """Test that real IV differs from the old 25% placeholder."""
        snap = get_snapshot_for_date(date(2020, 3, 16), DATA_DIR)
        assert snap is not None

        # Old placeholder was 0.25 (25%)
        placeholder_iv = Decimal("0.25")

        # Real IV should be significantly different
        assert abs(snap.atm_call_iv - placeholder_iv) > Decimal("0.20"), \
            "IV too close to placeholder value"

    def test_market_bottom_data(self):
        """Test data quality at market bottom (March 23, 2020)."""
        snap = get_snapshot_for_date(date(2020, 3, 23), DATA_DIR)
        assert snap is not None

        # SPY was around $220 at the bottom
        assert 200 < float(snap.underlying_price) < 250

        # IV should still be elevated but coming down
        assert float(snap.atm_call_iv) > 0.30


@pytest.mark.skipif(not DATA_EXISTS, reason="Options data files not found")
class TestConfigIntegration:
    """Tests for config integration with real options data."""

    def test_default_config_enables_real_data(self):
        """Test that default config enables real options data."""
        config = VolatilityArbitrageConfig()

        assert config.use_real_options_data is True
        assert config.options_data_dir == "src/volatility_arbitrage/data/SPY_Options_2019_24"

    def test_config_can_disable_real_data(self):
        """Test that real data can be disabled via config."""
        config = VolatilityArbitrageConfig(use_real_options_data=False)

        assert config.use_real_options_data is False
