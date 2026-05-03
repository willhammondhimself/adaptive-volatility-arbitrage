"""Regression tests for the real_options_loader module-level cache.

Covers F8: cache previously ignored data_dir, so a second call with a
different directory short-circuited and returned data from the first dir.
"""

from datetime import datetime, date
from decimal import Decimal

import pytest

from volatility_arbitrage.data import real_options_loader as rol
from volatility_arbitrage.data.real_options_loader import (
    DailyOptionsSnapshot,
    clear_cache,
    get_snapshot_for_date,
)


def _make_snapshot(d: date, underlying: float) -> DailyOptionsSnapshot:
    return DailyOptionsSnapshot(
        date=datetime(d.year, d.month, d.day),
        underlying_price=Decimal(str(underlying)),
        put_volume=0,
        call_volume=0,
        put_oi=0,
        call_oi=0,
        pc_ratio=Decimal("0"),
        atm_put_iv=Decimal("0.20"),
        atm_call_iv=Decimal("0.20"),
        iv_skew=Decimal("0"),
        front_month_iv=Decimal("0.20"),
        back_month_iv=Decimal("0.20"),
        term_slope=Decimal("0"),
        chains={},
    )


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


def test_cache_keys_on_data_dir(monkeypatch, tmp_path):
    """Two calls to get_snapshot_for_date with different data_dirs must
    return data from each directory, not cross-contaminate."""
    target = date(2023, 6, 1)
    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    dir_a.mkdir()
    dir_b.mkdir()

    snap_a = _make_snapshot(target, underlying=400.0)
    snap_b = _make_snapshot(target, underlying=500.0)

    def fake_load(year, data_dir):
        # Mirror real loader's caching behavior, including (dir, year) keying.
        key_dir = rol._normalize_dir(data_dir)
        if (key_dir, year) in rol._LOADED_YEARS:
            return [s for (kd, dt), s in rol._SNAPSHOT_CACHE.items()
                    if kd == key_dir and dt.year == year]
        snap = snap_a if key_dir == rol._normalize_dir(str(dir_a)) else snap_b
        rol._SNAPSHOT_CACHE[(key_dir, target)] = snap
        rol._LOADED_YEARS.add((key_dir, year))
        return [snap]

    monkeypatch.setattr(rol, "load_spy_options_year", fake_load)

    got_a = get_snapshot_for_date(target, data_dir=str(dir_a))
    got_b = get_snapshot_for_date(target, data_dir=str(dir_b))

    assert got_a is not None and got_b is not None
    assert got_a.underlying_price == Decimal("400.0")
    assert got_b.underlying_price == Decimal("500.0"), (
        "data_dir B returned dir A's snapshot — cache is ignoring data_dir"
    )


def test_cache_normalizes_relative_and_absolute(monkeypatch, tmp_path):
    """./foo and /abs/foo pointing at the same dir should hit the same cache entry."""
    target = date(2023, 6, 1)
    real_dir = tmp_path / "data"
    real_dir.mkdir()

    snap = _make_snapshot(target, underlying=425.0)
    call_count = {"n": 0}

    def fake_load(year, data_dir):
        call_count["n"] += 1
        key_dir = rol._normalize_dir(data_dir)
        rol._SNAPSHOT_CACHE[(key_dir, target)] = snap
        rol._LOADED_YEARS.add((key_dir, year))
        return [snap]

    monkeypatch.setattr(rol, "load_spy_options_year", fake_load)

    # Absolute path
    got1 = get_snapshot_for_date(target, data_dir=str(real_dir))
    # Same dir via a path with a trailing redundant component
    got2 = get_snapshot_for_date(target, data_dir=str(real_dir) + "/./")

    assert got1 is got2
    assert call_count["n"] == 1, "second call should hit cache, not reload"
