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
from stock_predictor.dashboard.components import watchlist_dataframe
from stock_predictor.dashboard.symbol_browser import render_symbol_browser
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
            'Your Watchlist is empty. Select a company on the Companies page and click "Add to '
            'Watchlist" to add it here.'
        )
        return

    def _actions(selected: dict | None) -> None:
        if selected is None:
            st.caption("Select a company (click a table row, bar or point) to remove it, open its full breakdown, or set a price alert.")
            return
        symbol_id = int(selected["symbol_id"])
        name_col, remove_col, view_col = st.columns([3, 2, 3], vertical_alignment="center")
        name_col.markdown(f"**{selected['Ticker']}** · {selected['Name']}")
        with remove_col:
            if st.button("Remove from Watchlist", key=f"toggle_{symbol_id}", use_container_width=True):

                def _do_remove() -> None:
                    with session_scope() as session:
                        set_watchlisted(session, symbol_id, False)

                if require_owner(_do_remove):
                    st.rerun()
        with view_col:
            if st.button(f"View {selected['Ticker']} full detail →", key=f"view_{symbol_id}", use_container_width=True):
                st.switch_page(navigation.symbol_detail_page(), query_params={"symbol": str(symbol_id)})

    selected = render_symbol_browser(df, "watchlist", _actions, page_label="Watchlist")

    if selected is not None:
        symbol_id = int(selected["symbol_id"])
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
        "Switch between the Table and Chart views and filter by search, sector or recommendation. Select "
        "a company and the bar above lets you remove it from your Watchlist or view its full breakdown; "
        "the price-alert form appears below (alerts are checked every ~15 minutes during market hours). "
        "Removing/setting an alert may ask for a password; viewing is open to everyone. "
        "MA Signal/P/E/News Sentiment/Social Sentiment/Recommendation show as blank for a symbol "
        "until enough data has been collected for it."
    )
