from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from stock_predictor.storage.models import Base
from stock_predictor.storage.repository import (
    all_symbols,
    competitor_snapshots,
    get_alert_threshold,
    get_or_create_symbol,
    latest_competitor_snapshot_fetch_time,
    latest_fundamentals,
    latest_price_bar,
    latest_quarterly_financials_fetch_time,
    latest_recommendation_log,
    latest_sentiment,
    latest_social_sentiment,
    log_recommendation,
    price_bars_for_symbol,
    quarterly_financials,
    recent_articles,
    replace_competitor_snapshots,
    replace_fundamentals_snapshot,
    replace_quarterly_financials,
    replace_recent_articles,
    replace_sentiment_snapshot,
    replace_social_sentiment_snapshot,
    set_alert_threshold,
    set_alert_triggered_state,
    set_watchlisted,
    symbol_by_ticker,
    symbols_with_alert_thresholds,
    upsert_price_bar,
    watchlisted_symbols,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_get_or_create_symbol_is_idempotent(session):
    first = get_or_create_symbol(session, "AAPL", "Apple Inc.", "Technology")
    second = get_or_create_symbol(session, "AAPL", "Apple Inc.", "Technology")
    assert first.id == second.id
    assert len(all_symbols(session)) == 1


def test_new_symbol_is_not_watchlisted_by_default(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    assert symbol.is_watchlisted is False
    assert watchlisted_symbols(session) == []


def test_set_watchlisted_adds_and_removes(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    get_or_create_symbol(session, "MSFT", "Microsoft Corporation")

    set_watchlisted(session, symbol.id, True)
    assert [s.ticker for s in watchlisted_symbols(session)] == ["AAPL"]

    set_watchlisted(session, symbol.id, False)
    assert watchlisted_symbols(session) == []


def test_set_watchlisted_on_unknown_symbol_id_is_a_no_op(session):
    set_watchlisted(session, 999, True)  # doesn't raise
    assert watchlisted_symbols(session) == []


def test_set_alert_threshold_creates_and_updates(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")

    set_alert_threshold(session, symbol.id, upper_price=200.0, lower_price=150.0)
    alert = get_alert_threshold(session, symbol.id)
    assert alert.upper_price == 200.0
    assert alert.lower_price == 150.0
    assert alert.above_triggered is False
    assert alert.below_triggered is False

    set_alert_threshold(session, symbol.id, upper_price=210.0, lower_price=None)
    alert = get_alert_threshold(session, symbol.id)
    assert alert.upper_price == 210.0
    assert alert.lower_price is None


def test_set_alert_threshold_resets_triggered_flags_on_change(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    set_alert_threshold(session, symbol.id, upper_price=200.0, lower_price=None)
    set_alert_triggered_state(session, symbol.id, above=True)
    assert get_alert_threshold(session, symbol.id).above_triggered is True

    set_alert_threshold(session, symbol.id, upper_price=210.0, lower_price=None)
    assert get_alert_threshold(session, symbol.id).above_triggered is False


def test_get_alert_threshold_returns_none_when_unset(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    assert get_alert_threshold(session, symbol.id) is None


def test_symbols_with_alert_thresholds_requires_watchlisted_and_a_threshold(session):
    watchlisted_with_threshold = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    set_watchlisted(session, watchlisted_with_threshold.id, True)
    set_alert_threshold(session, watchlisted_with_threshold.id, upper_price=200.0, lower_price=None)

    watchlisted_no_threshold = get_or_create_symbol(session, "MSFT", "Microsoft Corporation")
    set_watchlisted(session, watchlisted_no_threshold.id, True)

    not_watchlisted_with_threshold = get_or_create_symbol(session, "TSLA", "Tesla Inc.")
    set_alert_threshold(session, not_watchlisted_with_threshold.id, upper_price=300.0, lower_price=None)

    results = symbols_with_alert_thresholds(session)
    assert [s.ticker for s, _ in results] == ["AAPL"]


def test_symbols_with_alert_thresholds_excludes_a_symbol_unwatchlisted_after_being_set(session):
    """A removed-from-Watchlist symbol keeps its PriceAlert row (so
    re-adding it restores the threshold) but is excluded from the poll
    list until is_watchlisted is True again.
    """
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    set_watchlisted(session, symbol.id, True)
    set_alert_threshold(session, symbol.id, upper_price=200.0, lower_price=None)

    set_watchlisted(session, symbol.id, False)

    assert symbols_with_alert_thresholds(session) == []
    assert get_alert_threshold(session, symbol.id) is not None  # threshold itself preserved


def test_set_alert_triggered_state_only_updates_the_side_passed(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    set_alert_threshold(session, symbol.id, upper_price=200.0, lower_price=150.0)

    set_alert_triggered_state(session, symbol.id, above=True)
    alert = get_alert_threshold(session, symbol.id)
    assert alert.above_triggered is True
    assert alert.below_triggered is False

    set_alert_triggered_state(session, symbol.id, below=True)
    alert = get_alert_threshold(session, symbol.id)
    assert alert.above_triggered is True  # untouched
    assert alert.below_triggered is True


def test_set_alert_triggered_state_on_symbol_without_alert_row_is_a_no_op(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    set_alert_triggered_state(session, symbol.id, above=True)  # doesn't raise
    assert get_alert_threshold(session, symbol.id) is None


def test_symbol_by_ticker_returns_none_when_missing(session):
    assert symbol_by_ticker(session, "NOPE") is None


def test_upsert_price_bar_updates_not_duplicates(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    d = dt.date(2026, 1, 2)
    upsert_price_bar(session, symbol.id, d, 100, 102, 99, 101, 1000)
    upsert_price_bar(session, symbol.id, d, 100, 102, 99, 105, 2000)  # same date, revised close

    bars = price_bars_for_symbol(session, symbol.id)
    assert len(bars) == 1
    assert bars.iloc[0]["close"] == 105
    assert bars.iloc[0]["volume"] == 2000


def test_price_bars_for_symbol_sorted_oldest_first(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    upsert_price_bar(session, symbol.id, dt.date(2026, 1, 3), 1, 1, 1, 1, 1)
    upsert_price_bar(session, symbol.id, dt.date(2026, 1, 1), 1, 1, 1, 1, 1)
    upsert_price_bar(session, symbol.id, dt.date(2026, 1, 2), 1, 1, 1, 1, 1)

    bars = price_bars_for_symbol(session, symbol.id)
    assert list(bars["date"]) == [dt.date(2026, 1, 1), dt.date(2026, 1, 2), dt.date(2026, 1, 3)]


def test_latest_price_bar_returns_most_recent(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    upsert_price_bar(session, symbol.id, dt.date(2026, 1, 1), 1, 1, 1, 100, 1)
    upsert_price_bar(session, symbol.id, dt.date(2026, 1, 5), 1, 1, 1, 200, 1)

    latest = latest_price_bar(session, symbol.id)
    assert latest.date == dt.date(2026, 1, 5)
    assert latest.close == 200


def test_replace_sentiment_snapshot_overwrites_same_day(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    d = dt.date(2026, 1, 1)
    replace_sentiment_snapshot(session, symbol.id, d, 0.5, 3)
    replace_sentiment_snapshot(session, symbol.id, d, -0.2, 5)

    latest = latest_sentiment(session, symbol.id)
    assert latest.overall_sentiment_score == -0.2
    assert latest.article_count == 5


def test_replace_fundamentals_snapshot_overwrites_same_day(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    d = dt.date(2026, 1, 1)
    replace_fundamentals_snapshot(session, symbol.id, d, pe_ratio=30.0, forward_pe=28.0)
    replace_fundamentals_snapshot(session, symbol.id, d, pe_ratio=32.0, forward_pe=29.0)

    latest = latest_fundamentals(session, symbol.id)
    assert latest.pe_ratio == 32.0
    assert latest.forward_pe == 29.0


def test_latest_fundamentals_returns_none_when_missing(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    assert latest_fundamentals(session, symbol.id) is None


def test_replace_fundamentals_snapshot_stores_company_profile_fields(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    replace_fundamentals_snapshot(
        session,
        symbol.id,
        dt.date(2026, 1, 1),
        pe_ratio=30.0,
        forward_pe=28.0,
        name="Apple Inc",
        description="Apple designs and sells consumer electronics.",
        industry="COMPUTER HARDWARE",
        exchange="NASDAQ",
        country="USA",
        address="One Apple Park Way",
        official_site="https://www.apple.com",
        market_cap=3_900_000_000_000.0,
    )

    latest = latest_fundamentals(session, symbol.id)
    assert latest.name == "Apple Inc"
    assert latest.description == "Apple designs and sells consumer electronics."
    assert latest.industry == "COMPUTER HARDWARE"
    assert latest.exchange == "NASDAQ"
    assert latest.country == "USA"
    assert latest.address == "One Apple Park Way"
    assert latest.official_site == "https://www.apple.com"
    assert latest.market_cap == 3_900_000_000_000.0


def test_replace_fundamentals_snapshot_profile_fields_default_to_none(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    replace_fundamentals_snapshot(session, symbol.id, dt.date(2026, 1, 1), pe_ratio=30.0, forward_pe=28.0)

    latest = latest_fundamentals(session, symbol.id)
    assert latest.name is None
    assert latest.description is None


def test_replace_social_sentiment_snapshot_overwrites_same_day(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    d = dt.date(2026, 1, 1)
    replace_social_sentiment_snapshot(session, symbol.id, d, overall_sentiment_score=0.4, message_count=10)
    replace_social_sentiment_snapshot(session, symbol.id, d, overall_sentiment_score=-0.1, message_count=20)

    latest = latest_social_sentiment(session, symbol.id)
    assert latest.overall_sentiment_score == -0.1
    assert latest.message_count == 20


def test_latest_social_sentiment_returns_none_when_missing(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    assert latest_social_sentiment(session, symbol.id) is None


def test_log_recommendation_keeps_every_row(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    log_recommendation(session, symbol.id, "Hold")
    log_recommendation(session, symbol.id, "Buy")

    latest = latest_recommendation_log(session, symbol.id)
    assert latest.recommendation == "Buy"


def test_latest_recommendation_log_returns_none_when_missing(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    assert latest_recommendation_log(session, symbol.id) is None


def test_replace_recent_articles_stores_newest_first_order(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    older = dt.datetime(2026, 1, 1)
    newer = dt.datetime(2026, 1, 2)
    replace_recent_articles(
        session,
        symbol.id,
        [
            ("Older", "https://example.com/older", "Reuters", older, "summary 1"),
            ("Newer", "https://example.com/newer", "AP", newer, "summary 2"),
        ],
    )

    articles = recent_articles(session, symbol.id)
    assert [a.title for a in articles] == ["Newer", "Older"]


def test_replace_recent_articles_fully_replaces_prior_list(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    replace_recent_articles(
        session, symbol.id, [("First run", "https://example.com/1", "Reuters", dt.datetime(2026, 1, 1), "s")]
    )
    replace_recent_articles(
        session, symbol.id, [("Second run", "https://example.com/2", "AP", dt.datetime(2026, 1, 2), "s")]
    )

    articles = recent_articles(session, symbol.id)
    assert len(articles) == 1
    assert articles[0].title == "Second run"


def test_replace_recent_articles_handles_none_published_at(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    replace_recent_articles(session, symbol.id, [("No date", "https://example.com/1", "Reuters", None, "s")])

    articles = recent_articles(session, symbol.id)
    assert articles[0].published_at is None


def test_recent_articles_returns_empty_list_when_none_stored(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    assert recent_articles(session, symbol.id) == []


def test_replace_quarterly_financials_stores_oldest_first(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    replace_quarterly_financials(
        session,
        symbol.id,
        [
            (dt.date(2026, 6, 30), 96.0, 59.0, 2.46),
            (dt.date(2026, 3, 31), 81.0, 44.0, 1.81),
        ],
    )

    quarters = quarterly_financials(session, symbol.id)
    assert [q.fiscal_date_ending for q in quarters] == [dt.date(2026, 3, 31), dt.date(2026, 6, 30)]
    assert quarters[-1].total_revenue == 96.0
    assert quarters[-1].eps == 2.46


def test_replace_quarterly_financials_fully_replaces_prior_list(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    replace_quarterly_financials(session, symbol.id, [(dt.date(2026, 3, 31), 81.0, 44.0, 1.81)])
    replace_quarterly_financials(session, symbol.id, [(dt.date(2026, 6, 30), 96.0, 59.0, 2.46)])

    quarters = quarterly_financials(session, symbol.id)
    assert len(quarters) == 1
    assert quarters[0].fiscal_date_ending == dt.date(2026, 6, 30)


def test_quarterly_financials_returns_empty_list_when_none_stored(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    assert quarterly_financials(session, symbol.id) == []


def test_latest_quarterly_financials_fetch_time_returns_none_when_missing(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    assert latest_quarterly_financials_fetch_time(session, symbol.id) is None


def test_latest_quarterly_financials_fetch_time_returns_a_value_after_replace(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    replace_quarterly_financials(session, symbol.id, [(dt.date(2026, 6, 30), 96.0, 59.0, 2.46)])

    fetch_time = latest_quarterly_financials_fetch_time(session, symbol.id)
    assert fetch_time is not None


def test_replace_competitor_snapshots_stores_rows(session):
    symbol = get_or_create_symbol(session, "NVDA", "NVIDIA Corporation")
    replace_competitor_snapshots(
        session,
        symbol.id,
        [
            ("AMD", "Advanced Micro Devices, Inc.", 607.57, -0.30, -0.0005, 159.33, 639.0, 992_330_000_000.0, 156.0),
            ("INTC", "Intel Corporation", 45.31, 1.52, 0.0347, 10.93, 45.74, 3_070_000_000_000.0, 49.01),
        ],
    )

    rows = competitor_snapshots(session, symbol.id)
    assert {r.competitor_ticker for r in rows} == {"AMD", "INTC"}
    amd = next(r for r in rows if r.competitor_ticker == "AMD")
    assert amd.competitor_name == "Advanced Micro Devices, Inc."
    assert amd.price == 607.57
    assert amd.market_cap == 992_330_000_000.0
    assert amd.pe_ratio == 156.0


def test_replace_competitor_snapshots_fully_replaces_prior_list(session):
    symbol = get_or_create_symbol(session, "NVDA", "NVIDIA Corporation")
    replace_competitor_snapshots(
        session, symbol.id, [("AMD", "AMD", 600.0, 1.0, 0.001, 100.0, 700.0, 1e12, 150.0)]
    )
    replace_competitor_snapshots(
        session, symbol.id, [("INTC", "Intel", 45.0, 1.5, 0.03, 10.0, 45.0, 3e12, 49.0)]
    )

    rows = competitor_snapshots(session, symbol.id)
    assert len(rows) == 1
    assert rows[0].competitor_ticker == "INTC"


def test_competitor_snapshots_returns_empty_list_when_none_stored(session):
    symbol = get_or_create_symbol(session, "NVDA", "NVIDIA Corporation")
    assert competitor_snapshots(session, symbol.id) == []


def test_latest_competitor_snapshot_fetch_time_returns_none_when_missing(session):
    symbol = get_or_create_symbol(session, "NVDA", "NVIDIA Corporation")
    assert latest_competitor_snapshot_fetch_time(session, symbol.id) is None


def test_latest_competitor_snapshot_fetch_time_returns_a_value_after_replace(session):
    symbol = get_or_create_symbol(session, "NVDA", "NVIDIA Corporation")
    replace_competitor_snapshots(
        session, symbol.id, [("AMD", "AMD", 600.0, 1.0, 0.001, 100.0, 700.0, 1e12, 150.0)]
    )

    assert latest_competitor_snapshot_fetch_time(session, symbol.id) is not None
