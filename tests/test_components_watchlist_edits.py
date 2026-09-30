from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from stock_predictor.dashboard.components import apply_watchlist_edits
from stock_predictor.storage.models import Base
from stock_predictor.storage.repository import get_or_create_symbol, set_watchlisted, watchlisted_symbols


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def _df(rows):
    return pd.DataFrame(rows)


def test_apply_watchlist_edits_persists_a_newly_checked_row():
    with _session() as session:
        aapl = get_or_create_symbol(session, "AAPL", "Apple Inc.")
        session.commit()

        original = _df([{"symbol_id": aapl.id, "Ticker": "AAPL", "Watchlist": False}])
        edited = _df([{"symbol_id": aapl.id, "Ticker": "AAPL", "Watchlist": True}])

        changed = apply_watchlist_edits(session, original, edited)
        session.commit()

        assert changed is True
        assert [s.ticker for s in watchlisted_symbols(session)] == ["AAPL"]


def test_apply_watchlist_edits_persists_an_unchecked_row():
    with _session() as session:
        aapl = get_or_create_symbol(session, "AAPL", "Apple Inc.")
        set_watchlisted(session, aapl.id, True)
        session.commit()

        original = _df([{"symbol_id": aapl.id, "Ticker": "AAPL", "Watchlist": True}])
        edited = _df([{"symbol_id": aapl.id, "Ticker": "AAPL", "Watchlist": False}])

        changed = apply_watchlist_edits(session, original, edited)
        session.commit()

        assert changed is True
        assert watchlisted_symbols(session) == []


def test_apply_watchlist_edits_returns_false_when_nothing_changed():
    with _session() as session:
        aapl = get_or_create_symbol(session, "AAPL", "Apple Inc.")
        session.commit()

        df = _df([{"symbol_id": aapl.id, "Ticker": "AAPL", "Watchlist": False}])

        changed = apply_watchlist_edits(session, df, df.copy())

        assert changed is False


def test_apply_watchlist_edits_only_touches_changed_rows():
    with _session() as session:
        aapl = get_or_create_symbol(session, "AAPL", "Apple Inc.")
        msft = get_or_create_symbol(session, "MSFT", "Microsoft Corporation")
        session.commit()

        original = _df(
            [
                {"symbol_id": aapl.id, "Ticker": "AAPL", "Watchlist": False},
                {"symbol_id": msft.id, "Ticker": "MSFT", "Watchlist": False},
            ]
        )
        edited = _df(
            [
                {"symbol_id": aapl.id, "Ticker": "AAPL", "Watchlist": True},
                {"symbol_id": msft.id, "Ticker": "MSFT", "Watchlist": False},
            ]
        )

        changed = apply_watchlist_edits(session, original, edited)
        session.commit()

        assert changed is True
        assert [s.ticker for s in watchlisted_symbols(session)] == ["AAPL"]
