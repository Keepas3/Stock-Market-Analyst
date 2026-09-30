"""SQLAlchemy ORM models -- the canonical schema every ingest module writes
into. Direct analog of soccer-predictor's storage/models.py; see that
project's own PredictionRecord docstring for the forward-snapshot pattern
this schema is designed to support later (see PredictionRecord below).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Symbol(Base):
    """One tracked ticker -- the direct analog of soccer-predictor's Team.
    Unlike Team, no alias table is needed: a ticker is already a clean,
    stable, universal key with no name-matching problem to solve.

    `is_watchlisted` is a personal, UI-managed flag (Main page's "Add to
    Watchlist" checkbox, see dashboard/views/main.py) -- separate from
    being tracked at all (a row existing here, refreshed by
    scripts/refresh_live_data.py off config/watchlist.yaml). Every tracked
    symbol shows up on the Main/browse page regardless of this flag; the
    Watchlist page shows only the ones flagged True.
    """

    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    sector: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    is_watchlisted: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")


class PriceBar(Base):
    """One daily OHLCV bar -- the direct analog of soccer-predictor's Match
    (a completed historical result). Source is always the Yahoo Finance
    chart endpoint for now (see ingest/price_history.py); kept as a column
    anyway, same "record provenance" habit as Match.source, in case a
    second price source is ever added.
    """

    __tablename__ = "price_bars"
    __table_args__ = (UniqueConstraint("symbol_id", "date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    close: Mapped[float]
    volume: Mapped[int]
    source: Mapped[str] = mapped_column(String, default="yahoo_finance")


class SentimentSnapshot(Base):
    """One day's aggregated news-sentiment reading for one symbol -- the
    direct analog of soccer-predictor's current-form stats (Understat/ASA),
    not its Injury table (there's no "player" concept here). One row per
    (symbol_id, date); re-ingesting the same day replaces it (see
    storage/repository.py::replace_sentiment_snapshot), same idiom as
    replace_injuries.
    """

    __tablename__ = "sentiment_snapshots"
    __table_args__ = (UniqueConstraint("symbol_id", "date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    # Average VADER compound score across that day's articles for this
    # symbol -- see ingest/sentiment.py. -1 (very bearish) to +1 (very
    # bullish), same convention this app reuses in model/sentiment_adjustment.py.
    overall_sentiment_score: Mapped[float]
    article_count: Mapped[int]
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)


class NewsArticleSnapshot(Base):
    """One of the current top-N most recent news articles for a symbol
    (see ingest/sentiment.py::fetch_recent_articles) -- lets a user read
    the real reporting behind SentimentSnapshot's aggregated score, rather
    than just trusting a single number. Always fully replaced on refresh
    (see storage/repository.py::replace_recent_articles), not accumulated
    -- these represent "recent," not a permanent archive.
    """

    __tablename__ = "news_article_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    title: Mapped[str]
    url: Mapped[str]
    source: Mapped[str]
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    summary: Mapped[str]
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)


class QuarterlyFinancialsSnapshot(Base):
    """One reported quarter's revenue/net income/EPS for a symbol (see
    ingest/financials.py::fetch_quarterly_financials) -- powers the
    Financials bar chart + change notes on Symbol Detail. Net profit
    margin is derived (net_income / total_revenue), not stored -- same
    "derive, don't duplicate" pattern as dashboard/components.py's own
    market_snapshot. Always fully replaced on refresh (see
    storage/repository.py::replace_quarterly_financials), not accumulated
    -- represents "the current known quarters," not a permanent archive.
    """

    __tablename__ = "quarterly_financials_snapshots"
    __table_args__ = (UniqueConstraint("symbol_id", "fiscal_date_ending"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    fiscal_date_ending: Mapped[dt.date] = mapped_column(Date, index=True)
    total_revenue: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    net_income: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    eps: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)


class FundamentalsSnapshot(Base):
    """Latest P/E reading AND company-profile fields for one symbol --
    primarily from Finnhub, with Alpha Vantage's OVERVIEW call as a
    fallback/description source (see ingest/fundamentals.py), refreshed
    weekly rather than daily (P/E doesn't move day to day anyway; the
    profile fields move even less).
    """

    __tablename__ = "fundamentals_snapshots"
    __table_args__ = (UniqueConstraint("symbol_id", "date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    pe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    forward_pe: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    # Company-profile fields, from the SAME OVERVIEW call as pe_ratio above
    # -- no extra API cost. Alpha Vantage's OVERVIEW has no employee-count
    # or founding-year field (confirmed against its real documented
    # schema) -- not stored here rather than fabricated; see
    # ingest/fundamentals.py's own module docstring.
    name: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    description: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    industry: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    exchange: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    country: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    address: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    official_site: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)


class CompetitorSnapshot(Base):
    """One real competitor's peer-comparison data for a watchlist symbol
    (see ingest/competitors.py, config/competitors.yaml) -- CNN's own
    "Competitors" panel inspiration. `symbol_id` is the WATCHLIST symbol
    this row is a competitor OF, not the competitor's own Symbol row (a
    competitor ticker generally isn't itself a tracked watchlist symbol).

    Alpha Vantage's free tier has no "list companies by industry"
    endpoint (confirmed against its documented API surface) -- the
    competitor RELATIONSHIP is a small hand-curated config file, same
    "manual, human-curated fact list" pattern soccer-predictor uses for
    captains/star players, not fabricated or algorithmically inferred.
    price/change/52-week-range come from the keyless, unlimited Yahoo
    Finance endpoint; market_cap/pe_ratio come from Alpha Vantage's
    OVERVIEW (shares the account-wide 25/day budget -- see
    ingest/competitors.py's own module docstring). Always fully replaced
    on refresh (see storage/repository.py::replace_competitor_snapshots),
    not accumulated -- represents "the current comparison," not a
    permanent archive.
    """

    __tablename__ = "competitor_snapshots"
    __table_args__ = (UniqueConstraint("symbol_id", "competitor_ticker"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    competitor_ticker: Mapped[str] = mapped_column(String, index=True)
    competitor_name: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    price: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    change: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    pct_change: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fifty_two_week_low: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fifty_two_week_high: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    pe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)


class SocialSentimentSnapshot(Base):
    """One day's aggregated social-media sentiment reading (StockTwits +
    VADER, see ingest/social_sentiment.py) -- same shape as
    SentimentSnapshot again, kept as a separate table since news and
    social are two distinct pipelines worth showing distinctly, not one
    blended number.
    """

    __tablename__ = "social_sentiment_snapshots"
    __table_args__ = (UniqueConstraint("symbol_id", "date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    overall_sentiment_score: Mapped[float]
    message_count: Mapped[int]
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime)


class RecommendationLog(Base):
    """One row per refresh per symbol recording the composite Buy/Hold/Sell
    call at that moment (see model/technical_score.py) -- lets
    scripts/refresh_live_data.py detect a CHANGE since the last refresh and
    fire a webhook alert (see alerts/notifier.py). Every row is kept, not
    just the latest, so a future "alert history" view has real data.
    """

    __tablename__ = "recommendation_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    recommendation: Mapped[str] = mapped_column(String)
    recorded_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)


class ReturnModelParams(Base):
    """Persisted per-symbol return-distribution fit -- the direct analog of
    soccer-predictor's FittedParams, just one row per SYMBOL instead of one
    per LEAGUE (each symbol is fit independently; there's no cross-symbol
    interaction to fit jointly, unlike Dixon-Coles' per-league attack/
    defense fit -- see model/return_model.py).
    """

    __tablename__ = "return_model_params"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    fitted_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    mu: Mapped[float]  # weighted mean daily log return
    sigma: Mapped[float]  # weighted std dev of daily log return
    dof: Mapped[float]  # Student-t degrees of freedom (fixed constant for Phase 1)
    xi: Mapped[float]  # time-decay rate used for this fit
    n_bars: Mapped[int]  # price bars used in the fit
