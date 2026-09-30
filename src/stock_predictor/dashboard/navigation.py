"""Constructs every st.Page used by the app -- direct analog of
soccer-predictor's dashboard/navigation.py.
"""

from __future__ import annotations

import streamlit as st

from stock_predictor.dashboard.views import main, symbol_detail, watchlist


def main_page() -> st.Page:
    return st.Page(main.render, title="Main", icon=":material/home:", url_path="", default=True)


def watchlist_page() -> st.Page:
    return st.Page(watchlist.render, title="Watchlist", icon=":material/star:", url_path="watchlist")


def symbol_detail_page() -> st.Page:
    # Hidden from the sidebar, not removed -- only ever reached by clicking
    # a symbol's row on the Watchlist page, same "hidden, not gone" pattern
    # as soccer-predictor's team_detail_page().
    return st.Page(
        symbol_detail.render,
        title="Symbol Detail",
        icon=":material/search:",
        url_path="symbol",
        visibility="hidden",
    )
