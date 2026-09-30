"""Main page: every tracked symbol (config/watchlist.yaml, refreshed by
scripts/refresh_live_data.py), browsable, with a checkbox to add/remove it
from your personal Watchlist -- see storage/models.py::Symbol.is_watchlisted
and dashboard/views/watchlist.py, which shows only the ones checked here.
"""

from __future__ import annotations

import streamlit as st

from stock_predictor.dashboard.components import (
    apply_watchlist_edits,
    render_editable_symbol_table,
    watchlist_dataframe,
)
from stock_predictor.storage.db import session_scope


def render() -> None:
    st.title("🏠 Main")

    with session_scope() as session:
        df = watchlist_dataframe(session)

    if df.empty:
        st.warning("No symbols loaded yet. Run `uv run python scripts/fetch_historical_data.py` first.")
        st.stop()

    search_query = st.text_input("Search symbol", placeholder="e.g. AAPL", key="main_search")
    if search_query:
        filtered = df[
            df["Ticker"].str.contains(search_query, case=False, na=False)
            | df["Name"].str.contains(search_query, case=False, na=False)
        ]
    else:
        filtered = df

    if search_query and filtered.empty:
        st.info(f'No symbol matches "{search_query}".')
        return

    edited = render_editable_symbol_table(filtered, key="main_table")

    st.caption(
        "Check Watchlist to add a company to your personal Watchlist (see the Watchlist page in the "
        'sidebar); uncheck to remove it. Click a row\'s "View ->" link for the full breakdown. MA '
        "Signal/P/E/News Sentiment/Social Sentiment/Recommendation are blank until "
        "`uv run python scripts/run_training.py` has fit that symbol's return model at least once "
        "and `uv run python scripts/refresh_live_data.py` has pulled its fundamentals/sentiment readings."
    )

    if edited is not None:
        with session_scope() as session:
            changed = apply_watchlist_edits(session, filtered, edited)
        if changed:
            st.rerun()
