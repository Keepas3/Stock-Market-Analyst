"""The shared browse UI for the Companies and Watchlist pages: filters
(search, sector, recommendation), a Table/Chart toggle, and click-to-select in
either view feeding the same sticky action bar (components.selection_bar).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from stock_predictor.dashboard.components import render_symbol_table, selection_bar
from stock_predictor.dashboard.symbol_charts import (
    CHART_KINDS,
    COLOR_BY_OPTIONS,
    METRICS,
    RECOMMENDATIONS,
    available_sectors,
    filter_symbols,
    missing_count,
    screen_context,
    symbols_chart,
)

VIEWS = ("Table", "Chart")

# Plain (non-widget) session_state key holding the latest screen_context. Widget
# state is dropped when you navigate away from the page that rendered the
# widget, so the Assistant page reads this copy instead of the widgets.
BROWSE_CONTEXT_KEY = "browse_context"


def _selected_from_chart(event, filtered: pd.DataFrame) -> dict | None:
    """The company for the clicked point (its symbol_id rides along as
    customdata), or None if nothing is selected or it isn't in `filtered`."""
    points = event.selection.points if event and event.selection else []
    for point in points:
        custom = point.get("customdata")
        if not custom:
            continue
        match = filtered[filtered["symbol_id"] == int(custom[0])]
        if not match.empty:
            return match.iloc[0].to_dict()
    return None


def _render_chart(filtered: pd.DataFrame, key: str, cols: list) -> tuple[object, dict | None, dict] | None:
    """Chart controls go into `cols` (the same row as the View toggle, so the
    chart itself stays near the top of the page). Returns (action_bar,
    selected_row, chart_settings), or None if there's nothing to chart."""
    kind = cols[0].segmented_control("Chart type", CHART_KINDS, default="Bar", key=f"{key}_chart_kind") or "Bar"
    metric_names = list(METRICS)
    if kind == "Bar":
        metric = cols[1].selectbox("Metric", metric_names, index=0, key=f"{key}_bar_metric")
        x = y = None
        columns = [metric]
        # Sign-colored metrics (green up / red down) don't use a color key.
        diverging = METRICS[metric].diverging
        color_by = (
            cols[2].segmented_control(
                "Color by", COLOR_BY_OPTIONS, default="Sector", key=f"{key}_bar_color", disabled=diverging
            )
            or "Sector"
        )
    else:
        x = cols[1].selectbox("X axis", metric_names, index=metric_names.index("P/E"), key=f"{key}_x")
        y = cols[2].selectbox("Y axis", metric_names, index=0, key=f"{key}_y")
        color_by = (
            cols[3].segmented_control("Color by", COLOR_BY_OPTIONS, default="Sector", key=f"{key}_scatter_color")
            or "Sector"
        )
        metric = None
        columns = [x, y]

    hidden = missing_count(filtered, columns)
    shown = len(filtered) - hidden
    if shown == 0:
        st.info("None of the selected companies has data for that chart yet.")
        return None

    bar = selection_bar()  # reserved above the chart, filled by the caller after we return
    fig = symbols_chart(filtered, kind, metric=metric or "% Change", x=x or "P/E", y=y or "% Change", color_by=color_by)
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=f"{key}_chart_{kind}",
    )
    caption = f"Showing {shown} compan{'y' if shown == 1 else 'ies'}."
    if hidden:
        caption += f" {hidden} not shown (no data for this chart yet)."
    st.caption(caption + " Click a bar or point to select a company.")
    settings = (
        {"kind": "Bar", "metric": metric, "color_by": None if METRICS[metric].diverging else color_by}
        if kind == "Bar"
        else {"kind": "Scatter", "x": x, "y": y, "color_by": color_by}
    )
    return bar, _selected_from_chart(event, filtered), settings


def render_symbol_browser(
    df: pd.DataFrame,
    key: str,
    actions: Callable[[dict | None], None],
    page_label: str | None = None,
) -> dict | None:
    """Filters + Table/Chart view of `df` (a watchlist_dataframe frame).
    `actions(selected)` fills the sticky bar above the table/chart for the
    selected company (or None). Returns the selected row (or None) so the
    caller can render anything else that depends on it. Also records what
    is on screen (see BROWSE_CONTEXT_KEY) for the Assistant.
    """
    page = page_label or key.title()
    search_col, sector_col, rec_col = st.columns([2, 3, 2])
    search = search_col.text_input("Search symbol", placeholder="e.g. AAPL", key=f"{key}_search")
    sectors = sector_col.multiselect(
        "Sector", available_sectors(df), placeholder="All sectors", key=f"{key}_sectors"
    )
    recommendations = rec_col.multiselect(
        "Recommendation", RECOMMENDATIONS, placeholder="Any", key=f"{key}_recommendations"
    )
    filtered = filter_symbols(df, search, sectors, recommendations)

    control_cols = st.columns([2, 2, 3, 3, 3])
    view = control_cols[0].segmented_control("View", VIEWS, default="Table", key=f"{key}_view") or "Table"
    if len(filtered) != len(df):
        st.caption(f"{len(filtered)} of {len(df)} companies match.")

    def remember(selected: dict | None, chart: dict | None = None) -> None:
        st.session_state[BROWSE_CONTEXT_KEY] = screen_context(
            page, view, search, sectors, recommendations, filtered, len(df), chart, selected
        )

    if filtered.empty:
        st.info("No companies match those filters.")
        remember(None)
        return None

    chart_settings = None
    if view == "Table":
        bar = selection_bar()
        selected = render_symbol_table(filtered, key=f"{key}_table")
    else:
        result = _render_chart(filtered, key, control_cols[1:])
        if result is None:
            remember(None)
            return None
        bar, selected, chart_settings = result

    remember(selected, chart_settings)
    with bar:
        actions(selected)
    return selected
