"""One symbol's page: price chart, fundamentals, news/social sentiment,
and the recommendation. Reached from the Watchlist page. Direct analog of
soccer-predictor's dashboard/views/team_detail.py.
"""

from __future__ import annotations

import streamlit as st

from stock_predictor.dashboard import navigation
from stock_predictor.config import competitors_for
from stock_predictor.dashboard.components import (
    PRICE_CHART_RANGES,
    market_snapshot,
    price_chart,
    render_company_background,
    render_competitors_table,
    render_composite_breakdown,
    render_financials_section,
    render_prediction_breakdown,
    render_recent_articles,
)
from stock_predictor.prediction.service import predict_symbol
from stock_predictor.storage.db import session_scope
from stock_predictor.storage.repository import (
    competitor_snapshots,
    latest_fundamentals,
    latest_sentiment,
    latest_social_sentiment,
    price_bars_for_symbol,
    quarterly_financials,
    recent_articles,
    symbol_by_id,
)


def _parse_symbol_id() -> int | None:
    raw = st.query_params.get("symbol")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def render() -> None:
    symbol_id = _parse_symbol_id()

    with session_scope() as session:
        symbol = symbol_by_id(session, symbol_id) if symbol_id is not None else None
        # Extract plain values while the session is still open -- `symbol`
        # becomes a detached instance once this `with` block exits, and
        # touching its lazy attributes afterward raises DetachedInstanceError
        # (same reasoning as soccer-predictor's team_detail.py).
        if symbol is not None:
            ticker, name, sector = symbol.ticker, symbol.name, symbol.sector
        else:
            ticker = name = sector = None

    if symbol is None:
        st.info("Pick a symbol from the Watchlist page to see its detail.")
        if st.button("← Back to Watchlist"):
            st.switch_page(navigation.watchlist_page())
        st.stop()

    if st.button("← Back to Watchlist"):
        st.switch_page(navigation.watchlist_page())

    st.title(ticker)
    st.caption(name + (f" · {sector}" if sector else ""))

    with session_scope() as session:
        fundamentals_row = latest_fundamentals(session, symbol_id)
        # Extract to a plain dict while the session is still open -- same
        # detached-instance reasoning as every other snapshot row on this
        # page. Reused below for the "Fundamentals" section too, so P/E
        # only gets queried once per page render.
        fundamentals = (
            {
                "name": fundamentals_row.name,
                "description": fundamentals_row.description,
                # Symbol.sector (config/watchlist.yaml) is this app's one
                # source of truth for sector, already shown on the
                # Watchlist table -- Alpha Vantage's OVERVIEW also returns
                # its own "Sector" field, but it's a different taxonomy
                # and isn't persisted separately here to avoid two
                # disagreeing "Sector" values for the same symbol.
                "sector": sector,
                "industry": fundamentals_row.industry,
                "exchange": fundamentals_row.exchange,
                "official_site": fundamentals_row.official_site,
                "pe_ratio": fundamentals_row.pe_ratio,
                "forward_pe": fundamentals_row.forward_pe,
                "market_cap": fundamentals_row.market_cap,
                "date": fundamentals_row.date,
            }
            if fundamentals_row is not None
            else None
        )
    render_company_background(fundamentals or {})

    st.subheader("Price history")
    with session_scope() as session:
        price_df = price_bars_for_symbol(session, symbol_id)
    if price_df.empty:
        st.info("No price history available for this symbol yet.")
    else:
        range_label = st.segmented_control(
            "Range", options=list(PRICE_CHART_RANGES.keys()), default="1M", key=f"price_range_{symbol_id}"
        )
        st.plotly_chart(price_chart(price_df, ticker, range_label or "All"), use_container_width=True)
        latest_bar = price_df.iloc[-1]
        st.write(f"Last close ({latest_bar['date']}): **${latest_bar['close']:.2f}**")

    st.subheader("Fundamentals")
    if fundamentals is None:
        st.caption("No P/E reading available yet -- this updates weekly, since P/E doesn't move day to day.")
    else:
        pe_display = f"{fundamentals['pe_ratio']:.2f}" if fundamentals["pe_ratio"] is not None else "n/a"
        forward_pe_display = f"{fundamentals['forward_pe']:.2f}" if fundamentals["forward_pe"] is not None else "n/a"
        st.write(
            f"P/E ratio: **{pe_display}** · Forward P/E: **{forward_pe_display}** "
            f"(as of {fundamentals['date']})"
        )

    with session_scope() as session:
        # Extract to plain dicts while the session is open -- same
        # detached-instance reasoning as every other snapshot row on this
        # page. net_profit_margin is computed here rather than stored
        # (see QuarterlyFinancialsSnapshot's own docstring).
        financial_quarters = [
            {
                "fiscal_date_ending": q.fiscal_date_ending,
                "total_revenue": q.total_revenue,
                "net_income": q.net_income,
                "eps": q.eps,
                "net_profit_margin": (
                    q.net_income / q.total_revenue if q.total_revenue and q.net_income is not None else None
                ),
            }
            for q in quarterly_financials(session, symbol_id)
        ]
    if not financial_quarters:
        st.caption("No quarterly financials available yet -- this updates monthly.")
    else:
        render_financials_section(financial_quarters, ticker)

    if competitors_for(ticker):
        own_snapshot = market_snapshot(price_df)
        competitor_rows = [
            {
                "Ticker": ticker,
                "Name": name,
                "Market Cap": fundamentals["market_cap"] if fundamentals else None,
                "P/E": fundamentals["pe_ratio"] if fundamentals else None,
                "Price": own_snapshot["Price"],
                "Change": own_snapshot["Change"],
                "% Change": own_snapshot["% Change"],
                "52W Low": own_snapshot["52W Low"],
                "52W High": own_snapshot["52W High"],
            }
        ]
        with session_scope() as session:
            # Extract to plain dicts while the session is open -- same
            # detached-instance reasoning as every other snapshot row on
            # this page.
            competitor_rows.extend(
                {
                    "Ticker": c.competitor_ticker,
                    "Name": c.competitor_name or c.competitor_ticker,
                    "Market Cap": c.market_cap,
                    "P/E": c.pe_ratio,
                    "Price": c.price,
                    "Change": c.change,
                    "% Change": c.pct_change,
                    "52W Low": c.fifty_two_week_low,
                    "52W High": c.fifty_two_week_high,
                }
                for c in competitor_snapshots(session, symbol_id)
            )
        if len(competitor_rows) == 1:
            st.caption(f"No competitor data available yet for {ticker} -- this updates monthly.")
        else:
            render_competitors_table(competitor_rows, ticker)

    st.subheader("News sentiment")
    with session_scope() as session:
        sentiment = latest_sentiment(session, symbol_id)
        # Extract plain values while the session is still open -- `sentiment`
        # becomes a detached instance once this `with` block exits, and
        # touching its attributes afterward raises DetachedInstanceError
        # (same reasoning as this file's own `symbol` extraction above).
        if sentiment is not None:
            sentiment_score, article_count, sentiment_date = (
                sentiment.overall_sentiment_score,
                sentiment.article_count,
                sentiment.date,
            )
    if sentiment is None:
        st.caption("No news sentiment reading available yet -- this updates daily.")
    else:
        st.write(
            f"**{sentiment_score:+.3f}** (roughly -1=bearish to +1=bullish), "
            f"from {article_count} article(s) on {sentiment_date}."
        )

    with session_scope() as session:
        # Extract to plain dicts while the session is open -- same
        # detached-instance reasoning as every other snapshot row on this
        # page (see render_recent_articles' own docstring for why it takes
        # plain dicts rather than the ORM rows directly).
        article_rows = [
            {
                "title": a.title,
                "url": a.url,
                "source": a.source,
                "published_at": a.published_at,
                "summary": a.summary,
            }
            for a in recent_articles(session, symbol_id)
        ]
    render_recent_articles(article_rows)

    st.subheader("Social sentiment")
    with session_scope() as session:
        social = latest_social_sentiment(session, symbol_id)
        if social is not None:
            social_score, social_count, social_date = (
                social.overall_sentiment_score,
                social.message_count,
                social.date,
            )
    if social is None:
        st.caption("No social sentiment reading available yet -- this updates daily.")
    else:
        st.write(
            f"**{social_score:+.3f}** (roughly -1=bearish to +1=bullish), "
            f"from {social_count} recent post(s) on {social_date}."
        )

    st.subheader("Recommendation")
    with session_scope() as session:
        prediction = predict_symbol(session, symbol_id)
    if prediction is None or prediction.composite is None:
        st.warning("No prediction available for this symbol yet.")
    else:
        render_composite_breakdown(prediction.composite)
        with st.expander("Statistical model (secondary)", expanded=False):
            render_prediction_breakdown(prediction, ticker)
