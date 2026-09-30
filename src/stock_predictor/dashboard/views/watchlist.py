"""Watchlist page: every tracked symbol with its current price and
prediction summary. Clicking a row drills into Symbol Detail. Direct
analog of soccer-predictor's dashboard/views/home.py (minus the league
picker/standings-zone concepts, which have no stock analog).
"""

from __future__ import annotations

import streamlit as st

from stock_predictor.dashboard import navigation
from stock_predictor.dashboard.components import WATCHLIST_DISPLAY_COLUMNS, color_by_sign, watchlist_dataframe
from stock_predictor.storage.db import session_scope

_CLEAR_SELECTION_FLAG = "_clear_watchlist_selection"


def render() -> None:
    st.title("📈 Watchlist")

    if st.session_state.pop(_CLEAR_SELECTION_FLAG, False):
        st.session_state["watchlist_table"] = {"selection": {"rows": []}}

    with session_scope() as session:
        df = watchlist_dataframe(session)

    if df.empty:
        st.warning(
            "No symbols loaded yet. Run `uv run python scripts/fetch_historical_data.py` first."
        )
        st.stop()

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

    styled = filtered.style.map(color_by_sign, subset=["Change", "% Change"])

    height = 35 * (len(filtered) + 1) + 3
    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_order=WATCHLIST_DISPLAY_COLUMNS,
        column_config={
            "Price": st.column_config.NumberColumn(format="dollar"),
            "Change": st.column_config.NumberColumn(format="dollar"),
            "% Change": st.column_config.NumberColumn(format="percent"),
            "Volume": st.column_config.NumberColumn(format="compact"),
            "52W Low": st.column_config.NumberColumn(format="dollar"),
            "52W High": st.column_config.NumberColumn(format="dollar"),
            # Unformatted, this rendered raw floats to 6 decimal places
            # (e.g. "39.080000") -- same %.2fx style as the Competitors
            # table's own P/E column (dashboard/components.py).
            "P/E": st.column_config.NumberColumn(format="%.2fx"),
            "News Sentiment": st.column_config.NumberColumn(format="%+.3f"),
            "Social Sentiment": st.column_config.NumberColumn(format="%+.3f"),
        },
        on_select="rerun",
        selection_mode="single-row",
        key="watchlist_table",
    )

    st.caption(
        "MA Signal/P/E/News Sentiment/Social Sentiment/Recommendation are blank until "
        "`uv run python scripts/run_training.py` has fit that symbol's return model at least once "
        "and `uv run python scripts/refresh_live_data.py` has pulled its fundamentals/sentiment "
        "readings. Click a row for the full breakdown."
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        symbol_id = int(filtered.iloc[selected_rows[0]]["symbol_id"])
        st.session_state[_CLEAR_SELECTION_FLAG] = True
        st.switch_page(navigation.symbol_detail_page(), query_params={"symbol": str(symbol_id)})
