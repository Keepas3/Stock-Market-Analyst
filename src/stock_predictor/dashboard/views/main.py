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
from stock_predictor.dashboard.components import watchlist_dataframe
from stock_predictor.dashboard.symbol_browser import render_symbol_browser
from stock_predictor.storage.db import session_scope
from stock_predictor.storage.repository import set_watchlisted


def render() -> None:
    st.title("Companies")

    with session_scope() as session:
        df = watchlist_dataframe(session)

    if df.empty:
        st.warning("No symbols loaded yet. Please check back soon.")
        st.stop()

    def _actions(selected: dict | None) -> None:
        if selected is None:
            st.caption("Select a company (click a table row, bar or point) to add it to your Watchlist or open its full breakdown.")
            return
        symbol_id = int(selected["symbol_id"])
        is_watchlisted = bool(selected["Watchlist"])
        name_col, toggle_col, view_col = st.columns([3, 2, 3], vertical_alignment="center")
        name_col.markdown(f"**{selected['Ticker']}** · {selected['Name']}")
        with toggle_col:
            label = "Remove from Watchlist" if is_watchlisted else "Add to Watchlist"
            if st.button(label, key=f"toggle_{symbol_id}", use_container_width=True):

                def _do_toggle() -> None:
                    with session_scope() as session:
                        set_watchlisted(session, symbol_id, not is_watchlisted)

                if require_owner(_do_toggle):
                    st.rerun()
        with view_col:
            if st.button(f"View {selected['Ticker']} full detail →", key=f"view_{symbol_id}", use_container_width=True):
                st.switch_page(navigation.symbol_detail_page(), query_params={"symbol": str(symbol_id)})

    render_symbol_browser(df, "main", _actions, page_label="Companies")

    st.caption(
        "Switch between the Table and Chart views, and filter by search, sector or recommendation. Select "
        "a company and the bar above lets you add it to your personal Watchlist (see the Watchlist page "
        "in the sidebar) or open its full breakdown. Adding/removing may ask for a "
        "password; viewing is open to everyone. MA Signal/P/E/News Sentiment/Social Sentiment/"
        "Recommendation show as blank for a symbol until enough data has been collected for it."
    )
