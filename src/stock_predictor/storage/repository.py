"""Query helpers shared by ingestion, the model layer, and the dashboard.
Mirrors soccer-predictor's storage/repository.py shape and naming
conventions (upsert_match -> upsert_price_bar, matches_for_league ->
price_bars_for_symbol, replace_injuries -> replace_sentiment_snapshot).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from stock_predictor.storage.models import (
    CompetitorSnapshot,
    FundamentalsSnapshot,
    NewsArticleSnapshot,
    PriceBar,
    QuarterlyFinancialsSnapshot,
    RecommendationLog,
    ReturnModelParams,
    SentimentSnapshot,
    SocialSentimentSnapshot,
    Symbol,
)


def get_or_create_symbol(session: Session, ticker: str, name: str, sector: str | None = None) -> Symbol:
    symbol = session.scalar(select(Symbol).where(Symbol.ticker == ticker))
    if symbol is None:
        symbol = Symbol(ticker=ticker, name=name, sector=sector)
        session.add(symbol)
        session.flush()
    return symbol


def symbol_by_ticker(session: Session, ticker: str) -> Symbol | None:
    return session.scalar(select(Symbol).where(Symbol.ticker == ticker))


def symbol_by_id(session: Session, symbol_id: int) -> Symbol | None:
    return session.get(Symbol, symbol_id)


def all_symbols(session: Session) -> list[Symbol]:
    return list(session.scalars(select(Symbol).order_by(Symbol.ticker)).all())


def upsert_price_bar(
    session: Session,
    symbol_id: int,
    date: dt.date,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    source: str = "yahoo_finance",
) -> None:
    existing = session.scalar(
        select(PriceBar).where(PriceBar.symbol_id == symbol_id, PriceBar.date == date)
    )
    if existing is not None:
        existing.open = open
        existing.high = high
        existing.low = low
        existing.close = close
        existing.volume = volume
        existing.source = source
        return
    session.add(
        PriceBar(
            symbol_id=symbol_id,
            date=date,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            source=source,
        )
    )


def price_bars_for_symbol(session: Session, symbol_id: int) -> pd.DataFrame:
    """Oldest-first -- feeds the return-model fit, which doesn't care about
    order, but oldest-first is the natural shape for a price chart too.
    """
    rows = session.execute(
        select(PriceBar.date, PriceBar.open, PriceBar.high, PriceBar.low, PriceBar.close, PriceBar.volume)
        .where(PriceBar.symbol_id == symbol_id)
        .order_by(PriceBar.date.asc())
    ).all()
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def latest_price_bar(session: Session, symbol_id: int) -> PriceBar | None:
    return session.scalar(
        select(PriceBar).where(PriceBar.symbol_id == symbol_id).order_by(PriceBar.date.desc()).limit(1)
    )


def replace_sentiment_snapshot(
    session: Session,
    symbol_id: int,
    date: dt.date,
    overall_sentiment_score: float,
    article_count: int,
) -> None:
    """Replaces the day's sentiment reading if one already exists -- same
    "delete/overwrite, never duplicate" idiom as replace_injuries, since a
    same-day re-ingest just means the refresh script ran twice, not that
    two independent readings both count.
    """
    existing = session.scalar(
        select(SentimentSnapshot).where(SentimentSnapshot.symbol_id == symbol_id, SentimentSnapshot.date == date)
    )
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    if existing is not None:
        existing.overall_sentiment_score = overall_sentiment_score
        existing.article_count = article_count
        existing.fetched_at = now
        return
    session.add(
        SentimentSnapshot(
            symbol_id=symbol_id,
            date=date,
            overall_sentiment_score=overall_sentiment_score,
            article_count=article_count,
            fetched_at=now,
        )
    )


def latest_sentiment(session: Session, symbol_id: int) -> SentimentSnapshot | None:
    return session.scalar(
        select(SentimentSnapshot)
        .where(SentimentSnapshot.symbol_id == symbol_id)
        .order_by(SentimentSnapshot.date.desc())
        .limit(1)
    )


def replace_recent_articles(
    session: Session,
    symbol_id: int,
    articles: list[tuple[str, str, str, dt.datetime | None, str]],
) -> None:
    """Fully replaces this symbol's stored "recent articles" list --
    `articles` is a list of (title, url, source, published_at, summary)
    tuples, newest-first (see ingest/sentiment.py::fetch_recent_articles).
    A full delete+insert, not an upsert-by-URL: these represent "the
    current top N," not a permanent archive, so a day where fewer/different
    articles qualify should fully replace the prior list, not merge with it.
    """
    session.execute(delete(NewsArticleSnapshot).where(NewsArticleSnapshot.symbol_id == symbol_id))
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    for title, url, source, published_at, summary in articles:
        session.add(
            NewsArticleSnapshot(
                symbol_id=symbol_id,
                title=title,
                url=url,
                source=source,
                published_at=published_at,
                summary=summary,
                fetched_at=now,
            )
        )


def recent_articles(session: Session, symbol_id: int) -> list[NewsArticleSnapshot]:
    """Newest-first (NULLs -- an unparseable published_at -- sort last)."""
    return list(
        session.scalars(
            select(NewsArticleSnapshot)
            .where(NewsArticleSnapshot.symbol_id == symbol_id)
            .order_by(NewsArticleSnapshot.published_at.desc().nulls_last())
        ).all()
    )


def replace_quarterly_financials(
    session: Session,
    symbol_id: int,
    quarters: list[tuple[dt.date, float | None, float | None, float | None]],
) -> None:
    """Fully replaces this symbol's stored quarterly financials --
    `quarters` is a list of (fiscal_date_ending, total_revenue, net_income,
    eps) tuples (see ingest/financials.py::fetch_quarterly_financials).
    Full delete+insert, not an upsert-by-date, same "current snapshot, not
    an accumulating archive" idiom as replace_recent_articles.
    """
    session.execute(delete(QuarterlyFinancialsSnapshot).where(QuarterlyFinancialsSnapshot.symbol_id == symbol_id))
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    for fiscal_date_ending, total_revenue, net_income, eps in quarters:
        session.add(
            QuarterlyFinancialsSnapshot(
                symbol_id=symbol_id,
                fiscal_date_ending=fiscal_date_ending,
                total_revenue=total_revenue,
                net_income=net_income,
                eps=eps,
                fetched_at=now,
            )
        )


def quarterly_financials(session: Session, symbol_id: int) -> list[QuarterlyFinancialsSnapshot]:
    """Oldest-first -- the natural order for a chart timeline (same
    convention as price_bars_for_symbol)."""
    return list(
        session.scalars(
            select(QuarterlyFinancialsSnapshot)
            .where(QuarterlyFinancialsSnapshot.symbol_id == symbol_id)
            .order_by(QuarterlyFinancialsSnapshot.fiscal_date_ending.asc())
        ).all()
    )


def latest_quarterly_financials_fetch_time(session: Session, symbol_id: int) -> dt.datetime | None:
    """When this symbol's quarterly financials were last refreshed (the
    MAX fetched_at across its stored quarters) -- used to enforce the
    monthly refresh cadence in scripts/refresh_live_data.py. None if
    nothing's been fetched yet.
    """
    return session.scalar(
        select(func.max(QuarterlyFinancialsSnapshot.fetched_at)).where(
            QuarterlyFinancialsSnapshot.symbol_id == symbol_id
        )
    )


def replace_competitor_snapshots(
    session: Session,
    symbol_id: int,
    competitors: list[tuple[str, str | None, float | None, float | None, float | None, float | None, float | None, float | None, float | None]],
) -> None:
    """Fully replaces this watchlist symbol's stored competitor-comparison
    rows -- `competitors` is a list of (competitor_ticker, competitor_name,
    price, change, pct_change, fifty_two_week_low, fifty_two_week_high,
    market_cap, pe_ratio) tuples (see
    ingest/competitors.py::fetch_competitor_snapshot). Full delete+insert,
    same "current comparison, not an accumulating archive" idiom as
    replace_recent_articles/replace_quarterly_financials.
    """
    session.execute(delete(CompetitorSnapshot).where(CompetitorSnapshot.symbol_id == symbol_id))
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    for (
        competitor_ticker,
        competitor_name,
        price,
        change,
        pct_change,
        fifty_two_week_low,
        fifty_two_week_high,
        market_cap,
        pe_ratio,
    ) in competitors:
        session.add(
            CompetitorSnapshot(
                symbol_id=symbol_id,
                competitor_ticker=competitor_ticker,
                competitor_name=competitor_name,
                price=price,
                change=change,
                pct_change=pct_change,
                fifty_two_week_low=fifty_two_week_low,
                fifty_two_week_high=fifty_two_week_high,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                fetched_at=now,
            )
        )


def competitor_snapshots(session: Session, symbol_id: int) -> list[CompetitorSnapshot]:
    return list(
        session.scalars(select(CompetitorSnapshot).where(CompetitorSnapshot.symbol_id == symbol_id)).all()
    )


def latest_competitor_snapshot_fetch_time(session: Session, symbol_id: int) -> dt.datetime | None:
    """When this symbol's competitor comparison was last refreshed (the
    MAX fetched_at across its stored competitor rows) -- used to enforce
    the monthly refresh cadence in scripts/refresh_live_data.py. None if
    nothing's been fetched yet.
    """
    return session.scalar(
        select(func.max(CompetitorSnapshot.fetched_at)).where(CompetitorSnapshot.symbol_id == symbol_id)
    )


def replace_fundamentals_snapshot(
    session: Session,
    symbol_id: int,
    date: dt.date,
    pe_ratio: float | None,
    forward_pe: float | None,
    name: str | None = None,
    description: str | None = None,
    industry: str | None = None,
    exchange: str | None = None,
    country: str | None = None,
    address: str | None = None,
    official_site: str | None = None,
    market_cap: float | None = None,
) -> None:
    """Replaces the day's P/E + company-profile reading if one already
    exists -- same "delete/overwrite, never duplicate" idiom as
    replace_sentiment_snapshot. The profile fields (name/description/...)
    default to None so existing callers that only pass pe_ratio/forward_pe
    keep working unchanged.
    """
    existing = session.scalar(
        select(FundamentalsSnapshot).where(FundamentalsSnapshot.symbol_id == symbol_id, FundamentalsSnapshot.date == date)
    )
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    if existing is not None:
        existing.pe_ratio = pe_ratio
        existing.forward_pe = forward_pe
        existing.name = name
        existing.description = description
        existing.industry = industry
        existing.exchange = exchange
        existing.country = country
        existing.address = address
        existing.official_site = official_site
        existing.market_cap = market_cap
        existing.fetched_at = now
        return
    session.add(
        FundamentalsSnapshot(
            symbol_id=symbol_id,
            date=date,
            pe_ratio=pe_ratio,
            forward_pe=forward_pe,
            name=name,
            description=description,
            industry=industry,
            exchange=exchange,
            country=country,
            address=address,
            official_site=official_site,
            market_cap=market_cap,
            fetched_at=now,
        )
    )


def latest_fundamentals(session: Session, symbol_id: int) -> FundamentalsSnapshot | None:
    return session.scalar(
        select(FundamentalsSnapshot)
        .where(FundamentalsSnapshot.symbol_id == symbol_id)
        .order_by(FundamentalsSnapshot.date.desc())
        .limit(1)
    )


def replace_social_sentiment_snapshot(
    session: Session,
    symbol_id: int,
    date: dt.date,
    overall_sentiment_score: float,
    message_count: int,
) -> None:
    """Same "delete/overwrite, never duplicate" idiom as
    replace_sentiment_snapshot, for the separate social-sentiment table.
    """
    existing = session.scalar(
        select(SocialSentimentSnapshot).where(
            SocialSentimentSnapshot.symbol_id == symbol_id, SocialSentimentSnapshot.date == date
        )
    )
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    if existing is not None:
        existing.overall_sentiment_score = overall_sentiment_score
        existing.message_count = message_count
        existing.fetched_at = now
        return
    session.add(
        SocialSentimentSnapshot(
            symbol_id=symbol_id,
            date=date,
            overall_sentiment_score=overall_sentiment_score,
            message_count=message_count,
            fetched_at=now,
        )
    )


def latest_social_sentiment(session: Session, symbol_id: int) -> SocialSentimentSnapshot | None:
    return session.scalar(
        select(SocialSentimentSnapshot)
        .where(SocialSentimentSnapshot.symbol_id == symbol_id)
        .order_by(SocialSentimentSnapshot.date.desc())
        .limit(1)
    )


def log_recommendation(session: Session, symbol_id: int, recommendation: str) -> None:
    session.add(
        RecommendationLog(
            symbol_id=symbol_id,
            recommendation=recommendation,
            recorded_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
        )
    )


def latest_recommendation_log(session: Session, symbol_id: int) -> RecommendationLog | None:
    return session.scalar(
        select(RecommendationLog)
        .where(RecommendationLog.symbol_id == symbol_id)
        .order_by(RecommendationLog.recorded_at.desc())
        .limit(1)
    )


def save_return_model_params(
    session: Session, symbol_id: int, mu: float, sigma: float, dof: float, xi: float, n_bars: int
) -> ReturnModelParams:
    params = ReturnModelParams(
        symbol_id=symbol_id,
        fitted_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
        mu=mu,
        sigma=sigma,
        dof=dof,
        xi=xi,
        n_bars=n_bars,
    )
    session.add(params)
    session.flush()
    return params


def latest_return_model_params(session: Session, symbol_id: int) -> ReturnModelParams | None:
    return session.scalar(
        select(ReturnModelParams)
        .where(ReturnModelParams.symbol_id == symbol_id)
        .order_by(ReturnModelParams.fitted_at.desc())
        .limit(1)
    )
