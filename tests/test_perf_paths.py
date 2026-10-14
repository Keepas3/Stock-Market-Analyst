"""Pins the behaviour of the performance shortcuts to the straightforward
(slow) versions they replace, so they can never silently drift apart."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from stock_predictor import config
from stock_predictor.dashboard import components
from stock_predictor.model import overrides_store
from stock_predictor.prediction.service import predict_symbol
from stock_predictor.storage import repository as repo
from stock_predictor.storage.models import Base


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def add_bars(session, symbol_id, n, start=dt.date(2020, 1, 1), step_days=1, base=100.0):
    for i in range(n):
        price = base + i * 0.1 + (i % 7)
        repo.upsert_price_bar(
            session, symbol_id, start + dt.timedelta(days=i * step_days), price, price + 1, price - 1, price, 1000 + i
        )
    session.commit()


def test_recent_bars_equal_the_tail_of_full_history_for_every_symbol(session):
    long_sym = repo.get_or_create_symbol(session, "LONG", "Long Co")
    short_sym = repo.get_or_create_symbol(session, "SHORT", "Short Co")
    repo.get_or_create_symbol(session, "EMPTY", "No bars Co")
    session.flush()
    add_bars(session, long_sym.id, 700)
    add_bars(session, short_sym.id, 30)

    got = repo.recent_price_bars_by_symbol(session)

    assert set(got) == {long_sym.id, short_sym.id}  # a symbol with no bars is simply absent
    for symbol_id in got:
        expected = repo.price_bars_for_symbol(session, symbol_id).tail(repo.RECENT_BARS).reset_index(drop=True)
        pd.testing.assert_frame_equal(got[symbol_id], expected, check_exact=True)
    assert len(got[long_sym.id]) == repo.RECENT_BARS
    assert len(got[short_sym.id]) == 30
    assert isinstance(got[long_sym.id]["date"].iloc[0], dt.date)


def test_recent_bars_with_nothing_in_the_database(session):
    assert repo.recent_price_bars_by_symbol(session) == {}


def test_a_symbol_whose_data_stopped_months_ago_still_gets_its_own_recent_bars(session):
    stale = repo.get_or_create_symbol(session, "STALE", "Stale Co")
    fresh = repo.get_or_create_symbol(session, "FRESH", "Fresh Co")
    session.flush()
    add_bars(session, stale.id, 500, start=dt.date(2018, 1, 1))  # ends mid-2019
    add_bars(session, fresh.id, 500, start=dt.date(2024, 1, 1))
    got = repo.recent_price_bars_by_symbol(session)
    expected = repo.price_bars_for_symbol(session, stale.id).tail(repo.RECENT_BARS).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[stale.id], expected, check_exact=True)


def test_prediction_is_identical_with_preloaded_recent_bars(session, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    symbol = repo.get_or_create_symbol(session, "AAPL", "Apple Inc.")
    session.flush()
    add_bars(session, symbol.id, 900)
    repo.save_return_model_params(session, symbol.id, mu=0.001, sigma=0.02, dof=5.0, xi=0.008, n_bars=800)
    repo.replace_fundamentals_snapshot(session, symbol.id, dt.date.today(), 12.0, 11.0, name="Apple Inc.")
    session.commit()

    loaded = predict_symbol(session, symbol.id)
    preloaded = predict_symbol(session, symbol.id, None, repo.recent_price_bars_by_symbol(session)[symbol.id])

    assert preloaded.composite == loaded.composite
    assert preloaded.expected_return == loaded.expected_return
    assert preloaded.p_up == loaded.p_up and preloaded.return_range_80pct == loaded.return_range_80pct


def test_table_built_from_recent_bars_equals_table_built_from_full_history(session, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    ids = []
    for ticker, bars in (("AAA", 900), ("BBB", 250), ("CCC", 40), ("DDD", 0)):
        symbol = repo.get_or_create_symbol(session, ticker, f"{ticker} Co", "Technology")
        session.flush()
        ids.append(symbol.id)
        add_bars(session, symbol.id, bars, base=50 + len(ticker) * 10)
        repo.save_return_model_params(session, symbol.id, mu=0.001, sigma=0.02, dof=5.0, xi=0.008, n_bars=200)
    session.commit()

    fast = components.watchlist_dataframe(session)

    monkeypatch.setattr(
        components, "recent_price_bars_by_symbol", lambda sess: {i: repo.price_bars_for_symbol(sess, i) for i in ids}
    )
    slow = components.watchlist_dataframe(session)
    pd.testing.assert_frame_equal(fast, slow, check_exact=True)
    assert len(fast) == 4


# ---- the cached table ------------------------------------------------------------------------------


def test_data_version_changes_when_the_database_or_overrides_change(tmp_path, monkeypatch):
    db = tmp_path / "stocks.db"
    monkeypatch.setattr(config, "DB_PATH", db)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    db.write_bytes(b"one")
    first = components._data_version()
    assert components._data_version() == first  # stable while nothing changes

    db.write_bytes(b"two-two")
    second = components._data_version()
    assert second != first

    overrides_store.apply_changes("AAPL", {"news_weight": 0.5}, "x", "ai")
    third = components._data_version()
    assert third != second


def test_data_version_tolerates_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    assert components._data_version()  # no exception, still a usable key


def test_symbols_frame_hands_out_independent_copies(monkeypatch):
    shared = pd.DataFrame({"Ticker": ["AAA", "BBB"]})
    seen = {}

    def fake_cached(version):
        seen["version"] = version
        return shared

    monkeypatch.setattr(components, "_cached_symbols_frame", fake_cached)
    first = components.symbols_frame()
    first.loc[0, "Ticker"] = "CHANGED"
    assert shared.loc[0, "Ticker"] == "AAA"  # the cached original is untouched
    assert components.symbols_frame().loc[0, "Ticker"] == "AAA"
    assert seen["version"] == components._data_version()  # the version really is the cache key
