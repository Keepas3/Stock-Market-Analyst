"""Filtering and plotly figures for the Companies/Watchlist "Chart" view.
Pure pandas/plotly -- no Streamlit runtime needed to build a figure, so it's
independently testable (same split as components.py::price_chart).

Everything operates on the DataFrame components.watchlist_dataframe builds
(one row per company: symbol_id, Ticker, Name, Sector, Price, Change,
% Change, Volume, 52W Low/High, P/E, News/Social Sentiment, Recommendation).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

UP_COLOR = "#16a34a"
DOWN_COLOR = "#dc2626"
NEUTRAL_COLOR = "#d97706"
RECOMMENDATION_COLORS = {"Buy": UP_COLOR, "Hold": NEUTRAL_COLOR, "Sell": DOWN_COLOR}
NO_DATA = "No data"
RECOMMENDATIONS = ("Buy", "Hold", "Sell")

# Position of the price inside its 52-week range: 0% = at the low, 100% = at
# the high. Derived here rather than stored (same "derive, don't duplicate"
# idea as components.py::market_snapshot).
POSITION_COLUMN = "52W Position"


@dataclass(frozen=True)
class Metric:
    label: str
    value_format: str  # d3 format for hover text and axis ticks
    prefix: str = ""
    suffix: str = ""
    diverging: bool = False  # meaningful around zero -> bars colored green/red by sign


METRICS: dict[str, Metric] = {
    m.label: m
    for m in (
        Metric("% Change", ".2%", diverging=True),
        Metric("Change", ",.2f", prefix="$", diverging=True),
        Metric("Price", ",.2f", prefix="$"),
        Metric("Volume", ",.0f"),
        Metric("P/E", ".1f", suffix="x"),
        Metric("News Sentiment", "+.3f", diverging=True),
        Metric("Social Sentiment", "+.3f", diverging=True),
        Metric(POSITION_COLUMN, ".0%"),
    )
}
CHART_KINDS = ("Bar", "Scatter")
COLOR_BY_OPTIONS = ("Sector", "Recommendation")


def with_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    span = out["52W High"] - out["52W Low"]
    position = (out["Price"] - out["52W Low"]) / span.where(span > 0)
    out[POSITION_COLUMN] = position.clip(0, 1)
    return out


def available_sectors(df: pd.DataFrame) -> list[str]:
    return sorted(df["Sector"].dropna().unique())


def filter_symbols(
    df: pd.DataFrame,
    search: str = "",
    sectors: Iterable[str] = (),
    recommendations: Iterable[str] = (),
) -> pd.DataFrame:
    """Empty `sectors`/`recommendations` mean "no filter". `search` is a
    plain substring match on ticker or name (not a regex -- a stray "(" must
    not raise)."""
    mask = pd.Series(True, index=df.index)
    if search:
        mask &= df["Ticker"].str.contains(search, case=False, na=False, regex=False) | df["Name"].str.contains(
            search, case=False, na=False, regex=False
        )
    sectors = list(sectors)
    if sectors:
        mask &= df["Sector"].isin(sectors)
    recommendations = list(recommendations)
    if recommendations:
        mask &= df["Recommendation"].isin(recommendations)
    return df[mask]


def screen_context(
    page: str,
    view: str,
    search: str,
    sectors: Iterable[str],
    recommendations: Iterable[str],
    filtered: pd.DataFrame,
    total: int,
    chart: dict | None,
    selected: dict | None,
) -> dict:
    """A JSON-friendly snapshot of what the owner has on screen on the
    Companies/Watchlist pages, for the Assistant's get_screen_context tool.
    Pure so it's testable; the page stores it in st.session_state because
    widget state is dropped once you leave the page that rendered it."""
    return {
        "page": page,
        "view": view,
        "filters": {
            "search": search or None,
            "sectors": list(sectors),
            "recommendations": list(recommendations),
        },
        "chart": chart if view == "Chart" else None,
        "companies_visible": len(filtered),
        "companies_total": total,
        "visible_tickers": filtered["Ticker"].tolist()[:60],
        "selected_company": (
            {"ticker": selected["Ticker"], "name": selected["Name"]} if selected is not None else None
        ),
    }


def missing_count(df: pd.DataFrame, columns: Iterable[str]) -> int:
    """How many companies a chart over `columns` can't show for lack of data."""
    return int(with_derived_metrics(df)[list(columns)].isna().any(axis=1).sum())


def _group_colors(groups: list[str], color_by: str) -> dict[str, str]:
    if color_by == "Recommendation":
        return {g: RECOMMENDATION_COLORS.get(g, "#6b7280") for g in groups}
    palette = qualitative.T10
    return {g: palette[i % len(palette)] for i, g in enumerate(sorted(groups))}


def _hover_value(metric: Metric, axis: str) -> str:
    return f"{metric.prefix}%{{{axis}:{metric.value_format}}}{metric.suffix}"


def _customdata(data: pd.DataFrame) -> list[list]:
    """[symbol_id, name, sector, recommendation] per point. symbol_id is what a
    click selection hands back so the page can look the company up."""
    return [
        [
            int(r.symbol_id),
            r.Name,
            r.Sector if isinstance(r.Sector, str) else "n/a",
            r.Recommendation if isinstance(r.Recommendation, str) else NO_DATA,
        ]
        for r in data.itertuples()
    ]


def _bar_figure(df: pd.DataFrame, metric_label: str, color_by: str) -> go.Figure:
    metric = METRICS[metric_label]
    data = df.dropna(subset=[metric_label]).sort_values(metric_label, ascending=False)
    order = data["Ticker"].tolist()
    fig = go.Figure()
    hover = (
        "<b>%{x}</b> %{customdata[1]}<br>%{customdata[2]}<br>"
        f"{metric_label}: {_hover_value(metric, 'y')}<extra></extra>"
    )
    if metric.diverging:
        fig.add_bar(
            x=data["Ticker"],
            y=data[metric_label],
            marker_color=[UP_COLOR if v >= 0 else DOWN_COLOR for v in data[metric_label]],
            customdata=_customdata(data),
            hovertemplate=hover,
            showlegend=False,
        )
    else:
        group_col = data["Recommendation"].fillna(NO_DATA) if color_by == "Recommendation" else data["Sector"].fillna("n/a")
        colors = _group_colors(list(group_col.unique()), color_by)
        for group, part in data.groupby(group_col, sort=True):
            fig.add_bar(
                x=part["Ticker"],
                y=part[metric_label],
                name=group,
                marker_color=colors[group],
                customdata=_customdata(part),
                hovertemplate=hover,
            )
    fig.update_layout(barmode="relative")
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=order,
        tickangle=-90 if len(order) > 25 else 0,
        title=None,
    )
    fig.update_yaxes(
        title=metric_label,
        tickformat=metric.value_format,
        tickprefix=metric.prefix,
        ticksuffix=metric.suffix,
        zeroline=True,
    )
    return fig


def _scatter_figure(df: pd.DataFrame, x_label: str, y_label: str, color_by: str) -> go.Figure:
    x_metric, y_metric = METRICS[x_label], METRICS[y_label]
    data = df.dropna(subset=[x_label, y_label])
    group_col = data["Recommendation"].fillna(NO_DATA) if color_by == "Recommendation" else data["Sector"].fillna("n/a")
    colors = _group_colors(list(group_col.unique()), color_by)
    hover = (
        "<b>%{text}</b> %{customdata[1]}<br>%{customdata[2]}<br>"
        f"{x_label}: {_hover_value(x_metric, 'x')}<br>{y_label}: {_hover_value(y_metric, 'y')}<extra></extra>"
    )
    fig = go.Figure()
    for group, part in data.groupby(group_col, sort=True):
        fig.add_scatter(
            x=part[x_label],
            y=part[y_label],
            mode="markers+text",
            text=part["Ticker"],
            textposition="top center",
            textfont=dict(size=10),
            name=group,
            marker=dict(size=11, color=colors[group], line=dict(width=1, color="rgba(255,255,255,0.6)")),
            customdata=_customdata(part),
            hovertemplate=hover,
        )
    for metric, label, axis in ((x_metric, x_label, "x"), (y_metric, y_label, "y")):
        update = fig.update_xaxes if axis == "x" else fig.update_yaxes
        update(
            title=label,
            tickformat=metric.value_format,
            tickprefix=metric.prefix,
            ticksuffix=metric.suffix,
            zeroline=metric.diverging,
        )
    return fig


def symbols_chart(
    df: pd.DataFrame,
    kind: str = "Bar",
    *,
    metric: str = "% Change",
    x: str = "P/E",
    y: str = "% Change",
    color_by: str = "Sector",
) -> go.Figure:
    """One chart over every row of `df` (callers pass the already-filtered
    frame). Bar: `metric` per company, sorted high to low. Scatter: `x` vs `y`.
    Companies missing a charted value are left out (see missing_count)."""
    df = with_derived_metrics(df)
    fig = _scatter_figure(df, x, y, color_by) if kind == "Scatter" else _bar_figure(df, metric, color_by)
    fig.update_layout(
        height=560,
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
        clickmode="event+select",
    )
    return fig
