"""Main page: every tracked symbol (config/watchlist.yaml, refreshed by
scripts/refresh_live_data.py), browsable, with an Add/Remove Watchlist
button for whichever row you select -- see storage/models.py::Symbol.
is_watchlisted and dashboard/views/watchlist.py, which shows only the
ones added here.
"""

from __future__ import annotations

import streamlit as st

from stock_predictor.dashboard import navigation
from stock_predictor.dashboard.auth import require_owner
from stock_predictor.dashboard.components import render_symbol_table, watchlist_dataframe
from stock_predictor.storage.db import session_scope
from stock_predictor.storage.repository import set_watchlisted


def render() -> None:
    st.title("Main")

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

    selected = render_symbol_table(filtered, key="main_table")

    # Right below the table (not after the caption) -- with the table's
    # own height now capped (see components.MAX_TABLE_HEIGHT), these are
    # reachable with at most one short scroll after selecting a row,
    # instead of always sitting below the page's full row count.
    if selected is not None:
        symbol_id = int(selected["symbol_id"])
        is_watchlisted = bool(selected["Watchlist"])
        col1, col2 = st.columns(2)
        with col1:
            label = "Remove from Watchlist" if is_watchlisted else "Add to Watchlist"
            if st.button(label, key=f"toggle_{symbol_id}"):

                def _do_toggle(symbol_id=symbol_id, is_watchlisted=is_watchlisted) -> None:
                    with session_scope() as session:
                        set_watchlisted(session, symbol_id, not is_watchlisted)

                if require_owner(_do_toggle):
                    st.rerun()
        with col2:
            if st.button(f"View {selected['Ticker']} full detail →", key=f"view_{symbol_id}"):
                st.switch_page(navigation.symbol_detail_page(), query_params={"symbol": str(symbol_id)})

    st.caption(
        "Click a row to reveal buttons for adding it to your personal Watchlist (see the Watchlist "
        "page in the sidebar) or viewing its full breakdown -- adding/removing asks for the owner "
        "password if one is configured (see OWNER_PASSWORD in README.md), viewing is open to anyone. "
        "MA Signal/P/E/News Sentiment/Social Sentiment/Recommendation are blank until "
        "`uv run python scripts/run_training.py` has fit that symbol's return model at least once "
        "and `uv run python scripts/refresh_live_data.py` has pulled its fundamentals/sentiment readings."
    )
