"""Watchlist page: only the symbols you've personally added on the Main
page (Symbol.is_watchlisted) -- everything else tracked lives on the Main
page's full browse table. Select a row here to remove it. Direct analog
of soccer-predictor's dashboard/views/home.py (minus the league picker/
standings-zone concepts, which have no stock analog).
"""

from __future__ import annotations

import streamlit as st

from stock_predictor.dashboard import navigation
from stock_predictor.dashboard.auth import require_owner
from stock_predictor.dashboard.components import render_symbol_table, watchlist_dataframe
from stock_predictor.storage.db import session_scope
from stock_predictor.storage.repository import (
    get_alert_threshold,
    set_alert_threshold,
    set_watchlisted,
    watchlisted_symbols,
)


def render() -> None:
    st.title("Watchlist")

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

    # Right below the table (not after the caption) -- with the table's
    # own height now capped (see components.MAX_TABLE_HEIGHT), these are
    # reachable with at most one short scroll after selecting a row.
    if selected is not None:
        symbol_id = int(selected["symbol_id"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove from Watchlist", key=f"toggle_{symbol_id}"):

                def _do_remove(symbol_id=symbol_id) -> None:
                    with session_scope() as session:
                        set_watchlisted(session, symbol_id, False)

                if require_owner(_do_remove):
                    st.rerun()
        with col2:
            if st.button(f"View {selected['Ticker']} full detail →", key=f"view_{symbol_id}"):
                st.switch_page(navigation.symbol_detail_page(), query_params={"symbol": str(symbol_id)})

        st.subheader(f"Price alert -- {selected['Ticker']}")
        with session_scope() as session:
            existing_alert = get_alert_threshold(session, symbol_id)
            # Extract while the session is open -- see symbol_detail.py for
            # the same DetachedInstanceError reasoning.
            existing_upper = existing_alert.upper_price if existing_alert else None
            existing_lower = existing_alert.lower_price if existing_alert else None

        alert_col1, alert_col2 = st.columns(2)
        with alert_col1:
            upper_input = st.number_input(
                "Alert when price rises above ($)",
                value=existing_upper,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                placeholder="Not set",
                key=f"upper_{symbol_id}",
            )
        with alert_col2:
            lower_input = st.number_input(
                "Alert when price falls below ($)",
                value=existing_lower,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                placeholder="Not set",
                key=f"lower_{symbol_id}",
            )
        if st.button("Save price alert", key=f"save_alert_{symbol_id}"):

            def _do_save_alert(symbol_id=symbol_id, upper_input=upper_input, lower_input=lower_input) -> None:
                with session_scope() as session:
                    set_alert_threshold(session, symbol_id, upper_input, lower_input)

            if require_owner(_do_save_alert):
                st.success("Price alert saved.")

    st.caption(
        "Click a row to reveal buttons for removing it from your Watchlist, viewing its full "
        "breakdown, or setting a price alert (checked every ~15 minutes during market hours). "
        "Removing/setting an alert may ask for a password; viewing is open to everyone. "
        "MA Signal/P/E/News Sentiment/Social Sentiment/Recommendation show as blank for a symbol "
        "until enough data has been collected for it."
    )
