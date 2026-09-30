"""Reusable table/chart builders for the Streamlit dashboard -- direct
analog of soccer-predictor's dashboard/components.py.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.orm import Session

from stock_predictor.model.sentiment_adjustment import SENTIMENT_IMPACT_CAP
from stock_predictor.model.technical_score import BUY_VOTE_THRESHOLD, CompositeSignal, SELL_VOTE_THRESHOLD
from stock_predictor.prediction.service import predict_symbol
from stock_predictor.storage.models import Symbol
from stock_predictor.storage.repository import (
    all_symbols,
    latest_fundamentals,
    latest_sentiment,
    latest_social_sentiment,
    price_bars_for_symbol,
)

WATCHLIST_DISPLAY_COLUMNS = (
    "Ticker",
    "Name",
    "Sector",
    "Price",
    "Change",
    "% Change",
    "Volume",
    "52W Low",
    "52W High",
    "MA Signal",
    "P/E",
    "News Sentiment",
    "Social Sentiment",
    "Recommendation",
)

# Main/browse page adds a read-only Watchlist indicator column; Watchlist
# page shows the same table (already filtered to is_watchlisted=True) with
# the same column, always checked. See render_symbol_table's docstring for
# why the actual add/remove control is a button pair, not this column.
SYMBOL_TABLE_DISPLAY_COLUMNS = WATCHLIST_DISPLAY_COLUMNS + ("Watchlist",)

_MA_SIGNAL_LABELS = {1: "🟢 Golden Cross", -1: "🔴 Death Cross", 0: "⚪ Neutral"}

# 52-week range window -- calendar days, matching how "52-week high/low" is
# conventionally quoted (trailing year from the most recent bar), not a
# trading-day count.
FIFTY_TWO_WEEK_DAYS = 365

# Streamlit's dataframe grid applies a pandas Styler's per-cell CSS on top
# of column_config's own value formatting (format="dollar"/"percent" etc.
# still controls what's displayed; this only adds color) -- so both can be
# used together without conflict. Shared by the Watchlist and Competitors
# tables (see dashboard/views/watchlist.py, render_competitors_table below).
_UP_COLOR = "color: #16a34a"  # green
_DOWN_COLOR = "color: #dc2626"  # red


def color_by_sign(value: float | None) -> str:
    if pd.isna(value):
        return ""
    if value > 0:
        return _UP_COLOR
    if value < 0:
        return _DOWN_COLOR
    return ""


def market_snapshot(price_df: pd.DataFrame) -> dict:
    """Price/change/volume/52-week-range straight from already-ingested
    OHLCV history (see ingest/price_history.py) -- no new data source
    needed, same "derive it locally" spirit as
    model/technical_indicators.py's moving averages. All values None for
    an empty `price_df` (no history fetched yet for this symbol).
    """
    if price_df.empty:
        return {"Price": None, "Change": None, "% Change": None, "Volume": None, "52W Low": None, "52W High": None}

    last = price_df.iloc[-1]
    price = float(last["close"])
    volume = int(last["volume"])

    if len(price_df) >= 2:
        previous_close = float(price_df.iloc[-2]["close"])
        change = price - previous_close
        pct_change = change / previous_close if previous_close else None
    else:
        change = None
        pct_change = None

    cutoff = last["date"] - dt.timedelta(days=FIFTY_TWO_WEEK_DAYS)
    window = price_df[price_df["date"] >= cutoff]

    return {
        "Price": price,
        "Change": change,
        "% Change": pct_change,
        "Volume": volume,
        "52W Low": float(window["low"].min()),
        "52W High": float(window["high"].max()),
    }


def watchlist_dataframe(session: Session, symbols: list[Symbol] | None = None) -> pd.DataFrame:
    """One row per symbol in `symbols` (all tracked symbols if omitted) --
    the direct analog of soccer-predictor's leaderboard_dataframe. Includes
    a `symbol_id` column for row-click selection (pass
    `column_order=WATCHLIST_DISPLAY_COLUMNS` to hide it, or
    `SYMBOL_TABLE_DISPLAY_COLUMNS` to also show the "Watchlist" indicator
    column) and a "Watchlist" column mirroring `Symbol.is_watchlisted` --
    see render_symbol_table and dashboard/views/main.py's/watchlist.py's
    add/remove buttons.

    "Recommendation" (Buy/Hold/Sell) is prediction.composite.recommendation
    (see model/technical_score.py) -- an equal-weighted vote across the MA
    signal, P/E, news sentiment, and social sentiment columns shown right
    next to it, not a separate opaque signal. Not financial advice --
    worth repeating wherever this column is actually shown in the UI, not
    just in code.
    """
    rows = []
    for symbol in symbols if symbols is not None else all_symbols(session):
        prediction = predict_symbol(session, symbol.id)
        composite = prediction.composite if prediction else None
        fundamentals = latest_fundamentals(session, symbol.id)
        sentiment_row = latest_sentiment(session, symbol.id)
        social = latest_social_sentiment(session, symbol.id)
        snapshot = market_snapshot(price_bars_for_symbol(session, symbol.id))
        rows.append(
            {
                "symbol_id": symbol.id,
                "Ticker": symbol.ticker,
                "Name": symbol.name,
                "Sector": symbol.sector,
                **snapshot,
                "MA Signal": _MA_SIGNAL_LABELS.get(composite.ma_vote) if composite else None,
                "P/E": fundamentals.pe_ratio if fundamentals else None,
                "News Sentiment": sentiment_row.overall_sentiment_score if sentiment_row else None,
                "Social Sentiment": social.overall_sentiment_score if social else None,
                "Recommendation": composite.recommendation if composite else None,
                "Watchlist": symbol.is_watchlisted,
            }
        )
    return pd.DataFrame(rows)


_SYMBOL_TABLE_COLUMN_CONFIG = {
    "Price": st.column_config.NumberColumn(format="dollar"),
    "Change": st.column_config.NumberColumn(format="dollar"),
    "% Change": st.column_config.NumberColumn(format="percent"),
    "Volume": st.column_config.NumberColumn(format="compact"),
    "52W Low": st.column_config.NumberColumn(format="dollar"),
    "52W High": st.column_config.NumberColumn(format="dollar"),
    # Unformatted, this rendered raw floats to 6 decimal places
    # (e.g. "39.080000") -- same %.2fx style as the Competitors table's own
    # P/E column.
    "P/E": st.column_config.NumberColumn(format="%.2fx"),
    "News Sentiment": st.column_config.NumberColumn(format="%+.3f"),
    "Social Sentiment": st.column_config.NumberColumn(format="%+.3f"),
    # A plain (non-editable) checkmark -- st.dataframe never lets you edit
    # a cell, so this is purely a "already on your Watchlist?" indicator;
    # the button pair render_symbol_table's caller shows for the selected
    # row is the actual add/remove control (see that function's docstring
    # for why a data_editor + inline checkbox isn't used).
    "Watchlist": st.column_config.CheckboxColumn(help="Already on your personal Watchlist"),
}


def render_symbol_table(df: pd.DataFrame, key: str) -> dict | None:
    """The Main/Watchlist pages' shared table -- every WATCHLIST_DISPLAY_COLUMNS
    field plus a read-only "Watchlist" indicator. Click a row to select it;
    returns that row as a dict (None if nothing's selected, or `df` is
    empty and the caller should show its own empty-state message instead).

    Uses st.dataframe, not st.data_editor -- the installed Streamlit
    version's data_editor has no row-click/on_select navigation, which
    previously meant substituting a link column for navigation, but that
    renders as a real <a> tag that opens in a new browser tab (no way to
    force same-tab via st.column_config.LinkColumn's public API). This
    keeps the original click-to-select, same-tab-navigation experience;
    the caller renders explicit "Add/Remove Watchlist" and "View detail"
    buttons for whatever row is selected (see dashboard/views/main.py).
    """
    if df.empty:
        return None

    event = st.dataframe(
        df.style.map(color_by_sign, subset=["Change", "% Change"]),
        use_container_width=True,
        hide_index=True,
        height=35 * (len(df) + 1) + 3,
        column_order=SYMBOL_TABLE_DISPLAY_COLUMNS,
        column_config=_SYMBOL_TABLE_COLUMN_CONFIG,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    selected_rows = event.selection.rows if event and event.selection else []
    if not selected_rows:
        return None
    return df.iloc[selected_rows[0]].to_dict()


# CNN-style adjustable range presets, keyed by trading-day bar count (our
# price history is daily-only -- see ingest/price_history.py -- so there's
# no real intraday "1 Day" chart; "1D" here means the single most recent
# daily bar, shown as a point rather than a line).
PRICE_CHART_RANGES: dict[str, int | None] = {
    "1D": 1,
    "5D": 5,
    "14D": 14,
    "1M": 21,  # ~1 trading month
    "All": None,
}


def price_chart(price_df: pd.DataFrame, ticker: str, range_label: str = "All") -> go.Figure:
    """`price_df` needs `date`/`close` columns (see
    storage.repository.price_bars_for_symbol). Analog of soccer-predictor's
    scoreline_chart -- a plain plotly figure, no Streamlit runtime needed
    to build it (only to render it), so it's independently testable.

    `range_label` narrows to the trailing N bars per PRICE_CHART_RANGES
    (unknown labels fall back to the full history, same as "All").
    """
    bars = PRICE_CHART_RANGES.get(range_label)
    windowed = price_df if bars is None else price_df.tail(bars)

    fig = go.Figure(
        go.Scatter(
            x=windowed["date"],
            y=windowed["close"],
            mode="lines+markers" if len(windowed) <= 5 else "lines",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{ticker} -- adjusted close",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis_tickprefix="$",
        height=360,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def render_prediction_breakdown(prediction, ticker: str) -> None:
    """Shows every number behind one symbol's SECONDARY statistical
    forecast (the Student-t return-distribution model from Phase 1) -- the
    base fit, the sentiment nudge, and the resulting distribution -- so the
    actual math can be checked by hand, same transparency philosophy as
    soccer-predictor's own render_prediction_breakdown. No-op if
    `prediction` has no breakdown attached (no trained model yet).

    As of Phase 2 this is no longer the primary Buy/Hold/Sell surface (see
    render_composite_breakdown for that) and no longer wraps its own
    st.expander -- Streamlit doesn't support nested expanders, and the
    caller (dashboard/views/symbol_detail.py) already wraps this call in
    its own "Statistical model (secondary)" expander.

    Takes a plain `ticker` string (not a Symbol row) deliberately -- avoids
    any risk of touching a detached SQLAlchemy instance's attributes after
    its session_scope() block has already closed (see
    dashboard/views/symbol_detail.py's own comment on this).
    """
    b = prediction.breakdown if prediction else None
    if b is None:
        return

    st.markdown("**Where these numbers actually come from**")
    st.write(
        f"Daily mean return and volatility are fit **only from past prices** "
        f"({b.n_bars:,} daily returns), last fitted {b.fitted_at[:10]}. This base fit never "
        f"looks at news/sentiment -- only this symbol's own price history. Recent days count "
        f"for more than old ones (time-decay rate ξ = {b.xi:.4f})."
    )
    if b.sentiment_score is not None:
        st.write(
            f"Today's news sentiment (score **{b.sentiment_score:+.3f}**, roughly -1=bearish to "
            f"+1=bullish) nudges the daily mean return: {b.base_mean_return:+.5f} → "
            f"**{b.adjusted_mean_return:+.5f}** (capped at ±{SENTIMENT_IMPACT_CAP:.3f} absolute -- "
            "a deliberately small, speculative nudge, not a validated signal; see "
            "model/sentiment_adjustment.py)."
        )
    else:
        st.caption("No sentiment reading available for this symbol yet -- prediction uses the base fit only.")

    st.markdown(f"**{b.horizon_days}-trading-day forecast**")
    st.write(f"Expected return: **{prediction.expected_return:+.2%}**")
    lo, hi = prediction.return_range_80pct
    st.write(f"80% range: **{lo:+.2%}** to **{hi:+.2%}**")
    st.write(
        f"P(Up) **{prediction.p_up:.1%}** · P(Flat) **{prediction.p_flat:.1%}** · "
        f"P(Down) **{prediction.p_down:.1%}**"
    )
    st.caption(
        f"Modeled as a Student-t distribution (dof={b.dof:g}) around the {b.horizon_days}-day "
        "expected return, not Normal -- fatter tails, so a genuine large move isn't treated as "
        "impossibly rare (see model/markets.py)."
    )


def render_composite_breakdown(signal: CompositeSignal) -> None:
    """Shows each of the four votes plainly behind the composite
    Recommendation (see model/technical_score.py) -- the primary
    Buy/Hold/Sell surface as of Phase 2, mirroring
    render_prediction_breakdown's "show the actual math" transparency for
    a simple equal-weighted vote sum instead of a probability distribution.
    """
    st.markdown(f"**Recommendation: {signal.recommendation}**")
    st.write(
        f"MA signal: **{signal.ma_vote:+d}** · P/E: **{signal.pe_vote:+d}** · "
        f"News sentiment: **{signal.news_vote:+d}** · Social sentiment: **{signal.social_vote:+d}** "
        f"→ total **{signal.total:+d}**"
    )
    st.caption(
        "Each signal casts one vote (+1 bullish, -1 bearish, 0 neutral/no data yet). Buy needs a "
        f"total of +{BUY_VOTE_THRESHOLD} or higher, Sell needs {SELL_VOTE_THRESHOLD} or lower, "
        "otherwise Hold -- a simple, equal-weighted, non-backtested heuristic, not financial advice."
    )


def render_recent_articles(articles: list[dict]) -> None:
    """The raw recent articles behind a symbol's News Sentiment score (see
    ingest/sentiment.py::fetch_recent_articles) -- so a user can read the
    real reporting themselves instead of just trusting one aggregated
    number. A natural place for a future AI-summarization pass over these,
    not built here.

    Takes plain dicts (title/url/source/published_at/summary), not
    NewsArticleSnapshot ORM rows -- same detached-instance-avoidance
    reasoning as render_prediction_breakdown's `ticker: str` parameter
    (see dashboard/views/symbol_detail.py's own comment on this).
    """
    if not articles:
        return

    st.markdown("**Recent articles**")
    for article in articles:
        published = (
            article["published_at"].strftime("%Y-%m-%d %H:%M UTC") if article["published_at"] else "date unknown"
        )
        st.markdown(f"[{article['title']}]({article['url']})")
        st.caption(f"{article['source']} · {published}")


def render_company_background(fundamentals: dict) -> None:
    """The company's own descriptive profile (sector/industry/exchange/
    description/website). Description/address always come from Alpha
    Vantage's OVERVIEW call (Finnhub's profile has no equivalent field);
    the rest may come from either source -- see
    ingest/fundamentals.py::fetch_overview's merge. A no-op if no
    description is on record yet (no fundamentals snapshot, or a symbol
    Alpha Vantage doesn't cover).

    Deliberately doesn't show employee count or founding year, unlike some
    other stock sites' "About" panels -- Alpha Vantage's OVERVIEW has no
    such fields (confirmed against its real documented schema; see
    ingest/fundamentals.py's own module docstring) and this app never
    fabricates data to match a richer competitor's UI.

    Takes a plain dict, not a FundamentalsSnapshot ORM row -- same
    detached-instance-avoidance reasoning as render_recent_articles.
    """
    if not fundamentals.get("description"):
        return

    heading = f"About {fundamentals['name']}" if fundamentals.get("name") else "About"
    st.subheader(heading)
    st.write(fundamentals["description"])

    sector_display = (fundamentals.get("sector") or "n/a").title()
    industry_display = (fundamentals.get("industry") or "n/a").title()
    exchange_display = fundamentals.get("exchange") or "n/a"
    # Plain text, not st.metric -- st.metric truncates longer values (e.g.
    # Alpha Vantage's own "INFORMATION TECHNOLOGY SERVICES" industry
    # strings don't fit its fixed-width card), and every other field on
    # this page already uses plain st.write, not metric cards.
    st.write(f"**Sector:** {sector_display} · **Industry:** {industry_display} · **Exchange:** {exchange_display}")

    if fundamentals.get("official_site"):
        st.write(f"Website: [{fundamentals['official_site']}]({fundamentals['official_site']})")


def _format_money(value: float | None) -> str:
    """Abbreviated currency (e.g. "$96.22B", "$5.51T"), matching CNN's own
    Financials/Competitors panel style -- easier to scan than a full
    13-digit market-cap figure.
    """
    # pd.isna, not `value is None` -- render_competitors_table applies this
    # over a pandas column built from a mix of real numbers and None
    # (e.g. a symbol with no fundamentals data yet), and pandas silently
    # upcasts None to NaN in a numeric column (confirmed live: this
    # produced a literal "$nan" in the UI before this check was added).
    if pd.isna(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"{sign}${magnitude / 1e12:.2f}T"
    if magnitude >= 1e9:
        return f"{sign}${magnitude / 1e9:.2f}B"
    if magnitude >= 1e6:
        return f"{sign}${magnitude / 1e6:.2f}M"
    return f"{sign}${magnitude:,.0f}"


def _percent_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / previous


def financials_bar_chart(quarters: list[dict], ticker: str) -> go.Figure:
    """Grouped bar chart of quarterly Total Revenue vs. Net Income,
    oldest-left (see ingest/financials.py::fetch_quarterly_financials) --
    the "1-year income & revenue" analog of CNN's own Financials panel.
    Mirrors price_chart's construction: a plain plotly figure, no
    Streamlit runtime needed to build it, so it's independently testable.
    Bars are labeled by the actual fiscal_date_ending month/year rather
    than a guessed "Q1/Q2/..." label -- not every company's fiscal
    quarters align to the calendar, and this app has no per-company
    fiscal-year-end data to derive the real quarter number from.
    """
    labels = [q["fiscal_date_ending"].strftime("%b %Y") for q in quarters]
    fig = go.Figure()
    fig.add_bar(name="Total Revenue", x=labels, y=[q["total_revenue"] for q in quarters], marker_color="#3b82f6")
    fig.add_bar(name="Net Income", x=labels, y=[q["net_income"] for q in quarters], marker_color="#f59e0b")
    fig.update_layout(
        title=f"{ticker} -- quarterly revenue & net income",
        barmode="group",
        yaxis_title="USD",
        yaxis_tickprefix="$",
        height=360,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def render_financials_section(quarters: list[dict], ticker: str) -> None:
    """CNN-style "Financials" panel: the latest quarter's revenue/net
    income/EPS/net profit margin, a quarterly bar chart, and short
    computed (not AI-generated -- plain arithmetic on real fetched
    numbers) year-over-year/quarter-over-quarter change notes. A no-op if
    no quarters are on record yet.

    Deliberately doesn't show Free Cash Flow or Debt-to-Equity, unlike
    CNN's own panel -- those need Alpha Vantage's CASH_FLOW/BALANCE_SHEET
    endpoints, two MORE calls on the same shared 25/day budget; not built
    this round (see ingest/financials.py's own module docstring for the
    real trade-off, not a silent gap).

    Takes a list of plain dicts, not QuarterlyFinancialsSnapshot ORM rows
    -- same detached-instance-avoidance reasoning as render_recent_articles.
    """
    if not quarters:
        return

    chart_quarters = quarters[-4:]
    latest = chart_quarters[-1]
    prior_quarter = chart_quarters[-2] if len(chart_quarters) >= 2 else None
    prior_year = quarters[-5] if len(quarters) >= 5 else None

    st.subheader(f"{ticker} Financials")
    st.caption(f"Quarter ending {latest['fiscal_date_ending']}")

    margin = latest["net_profit_margin"]
    margin_display = f"{margin:.1%}" if margin is not None else "n/a"
    eps_display = f"${latest['eps']:.2f}" if latest["eps"] is not None else "n/a"
    # st.metric, not st.write/markdown -- its value isn't markdown-parsed,
    # which matters here: multiple "$..." values in one markdown string
    # get misread as $...$ LaTeX math delimiters (confirmed live -- the
    # first two dollar signs paired up and silently ate the text and bold
    # markup between them). st.metric sidesteps that entirely.
    cols = st.columns(4)
    cols[0].metric("Total revenue", _format_money(latest["total_revenue"]))
    cols[1].metric("Net income", _format_money(latest["net_income"]))
    cols[2].metric("Net profit margin", margin_display)
    cols[3].metric("EPS", eps_display)

    st.plotly_chart(financials_bar_chart(chart_quarters, ticker), use_container_width=True)

    for label, key in (("Total revenue", "total_revenue"), ("Net income", "net_income")):
        yoy = _percent_change(latest[key], prior_year[key]) if prior_year else None
        qoq = _percent_change(latest[key], prior_quarter[key]) if prior_quarter else None
        parts = []
        if yoy is not None:
            parts.append(f"{'increased' if yoy >= 0 else 'decreased'} {abs(yoy):.1%} since last year")
        if qoq is not None:
            parts.append(f"{'increased' if qoq >= 0 else 'decreased'} {abs(qoq):.1%} since last quarter")
        if parts:
            st.write(f"**{label}** {' and '.join(parts)}.")

    st.caption(
        "Revenue/Net Income/EPS via Alpha Vantage's INCOME_STATEMENT/EARNINGS endpoints, refreshed "
        "monthly. Free Cash Flow and Debt-to-Equity aren't shown -- they'd need two more Alpha Vantage "
        "endpoints on the same shared 25 requests/day budget."
    )


COMPETITORS_DISPLAY_COLUMNS = (
    "Ticker",
    "Name",
    "Market Cap",
    "P/E",
    "Price",
    "Change",
    "% Change",
    "52W Low",
    "52W High",
)


def render_competitors_table(rows: list[dict], ticker: str) -> None:
    """CNN-style "Competitors" panel: `ticker`'s own row (already-computed
    price/fundamentals data, no extra fetch) plus its real, hand-curated
    competitors (see config/competitors.yaml, ingest/competitors.py) side
    by side on Market Cap/P-E/Price/Change/52-week range. A no-op if
    `rows` is empty (e.g. no competitors configured for this ticker).

    `rows` is a list of plain dicts with keys Ticker/Name/Market Cap/P-E/
    Price/Change/% Change/52W Low/52W High -- built by the caller from
    already-open-session data (own detached-instance-avoidance reasoning
    as render_recent_articles).
    """
    if not rows:
        return

    df = pd.DataFrame(rows)
    # Market Cap needs a "$...T" abbreviation Streamlit's built-in
    # NumberColumn presets don't offer (its "compact" preset abbreviates
    # but has no currency prefix) -- pre-formatted as a plain string here,
    # same as model/dashboard/components.py's own Financials section.
    df["Market Cap"] = df["Market Cap"].apply(_format_money)
    styled = df.style.map(color_by_sign, subset=["Change", "% Change"])

    st.subheader(f"{ticker} Competitors")
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_order=COMPETITORS_DISPLAY_COLUMNS,
        column_config={
            "P/E": st.column_config.NumberColumn(format="%.2fx"),
            "Price": st.column_config.NumberColumn(format="dollar"),
            "Change": st.column_config.NumberColumn(format="dollar"),
            "% Change": st.column_config.NumberColumn(format="percent"),
            "52W Low": st.column_config.NumberColumn(format="dollar"),
            "52W High": st.column_config.NumberColumn(format="dollar"),
        },
    )
    st.caption(
        "Competitor relationships are hand-curated (see config/competitors.yaml), not derived live -- "
        "Alpha Vantage's free tier has no \"list companies by industry\" endpoint. Refreshed monthly, "
        "sharing Alpha Vantage's 25 requests/day budget for Market Cap/P-E (price/change/52-week range "
        "are keyless and unlimited)."
    )
