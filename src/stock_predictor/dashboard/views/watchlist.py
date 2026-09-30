"""Watchlist page: only the symbols you've personally added on the Main
page (Symbol.is_watchlisted) -- everything else tracked lives on the Main
page's full browse table. Select a row here to remove it. Direct analog
of soccer-predictor's dashboard/views/home.py (minus the league picker/
standings-zone concepts, which have no stock analog).
"""

from __future__ import annotations

import streamlit as st

from stock_predictor.dashboard import navigation
from stock_predictor.dashboard.components import render_symbol_table, watchlist_dataframe
from stock_predictor.storage.db import session_scope
from stock_predictor.storage.repository import set_watchlisted, watchlisted_symbols


def render() -> None:
    st.title("📈 Watchlist")

    with session_scope() as session:
        df = watchlist_dataframe(session, watchlisted_symbols(session))

    if df.empty:
        st.info(
            'Your Watchlist is empty. Select a company on the Main page and click "Add to '
            'Watchlist" to add it here.'
        )
        return

    search_query = st.text_input("Search symbol", placeholder="e.g. AAPL")
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

    selected = render_symbol_table(filtered, key="watchlist_table")

    st.caption(
        "Click a row, then use the buttons below to remove it from your Watchlist or view its full "
        "breakdown. MA Signal/P/E/News Sentiment/Social Sentiment/Recommendation are blank until "
        "`uv run python scripts/run_training.py` has fit that symbol's return model at least once "
        "and `uv run python scripts/refresh_live_data.py` has pulled its fundamentals/sentiment readings."
    )

    if selected is not None:
        symbol_id = int(selected["symbol_id"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Remove from Watchlist", key=f"toggle_{symbol_id}"):
                with session_scope() as session:
                    set_watchlisted(session, symbol_id, False)
                st.rerun()
        with col2:
            if st.button(f"View {selected['Ticker']} full detail →", key=f"view_{symbol_id}"):
                st.switch_page(navigation.symbol_detail_page(), query_params={"symbol": str(symbol_id)})
