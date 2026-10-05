"""Refreshes today's price bar + sentiment/fundamentals/financials/social
readings for every watchlisted symbol, refits each symbol's return model,
and fires a webhook alert whenever a symbol's composite Recommendation
changes.

Usage:
    uv run python scripts/refresh_live_data.py [TICKER ...]

Fundamentals (P/E + company profile) and news sentiment now come primarily
from FINNHUB_API_KEY (free tier: 60 requests/minute, no daily cap -- see
ingest/finnhub_client.py) -- this is what let the watchlist grow from
~16 to ~50 symbols (see config/watchlist.yaml's own header comment).
Quarterly financials (revenue/net income/EPS) still require
ALPHA_VANTAGE_API_KEY, and Alpha Vantage's OVERVIEW is still called weekly
as a fundamentals fallback and as the sole source of description/address
(see ingest/fundamentals.py) -- both light, well under Alpha Vantage's
shared 25 requests/day budget (see ingest/alpha_vantage_client.py) now
that sentiment no longer draws from it at all. Social sentiment
(StockTwits + VADER) is keyless and unrelated to any of these budgets.
Every one of these degrades to "skipped" per-symbol (never raises) on
failure; price refresh and training still proceed either way. Alerts
require DISCORD_WEBHOOK_URL in .env -- skipped (not an error) if unset --
and only fire for symbols with Symbol.is_watchlisted set (i.e. starred on
the Watchlist page), even though data for the full config/watchlist.yaml
universe is refreshed and retrained regardless.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_predictor.alerts.notifier import format_price_move, send_webhook_alert  # noqa: E402
from stock_predictor.config import competitors_for, load_watchlist  # noqa: E402
from stock_predictor.ingest.competitors import fetch_competitor_snapshot  # noqa: E402
from stock_predictor.ingest.financials import fetch_quarterly_financials  # noqa: E402
from stock_predictor.ingest.fundamentals import fetch_overview  # noqa: E402
from stock_predictor.ingest.price_history import fetch_recent_bars  # noqa: E402
from stock_predictor.ingest.sentiment import fetch_recent_articles, fetch_sentiment  # noqa: E402
from stock_predictor.ingest.social_sentiment import fetch_recent_posts, fetch_social_sentiment  # noqa: E402
from stock_predictor.prediction.service import predict_symbol  # noqa: E402
from stock_predictor.prediction.training import train_symbol  # noqa: E402
from stock_predictor.storage.db import init_db, session_scope  # noqa: E402
from stock_predictor.storage.repository import (  # noqa: E402
    competitor_snapshots,
    get_or_create_symbol,
    latest_competitor_snapshot_fetch_time,
    latest_fundamentals,
    latest_quarterly_financials_fetch_time,
    latest_recommendation_log,
    log_recommendation,
    price_bars_for_symbol,
    replace_competitor_snapshots,
    replace_fundamentals_snapshot,
    replace_quarterly_financials,
    replace_recent_articles,
    replace_recent_social_posts,
    replace_sentiment_snapshot,
    replace_social_sentiment_snapshot,
    upsert_price_bar,
)

FUNDAMENTALS_REFRESH_DAYS = 7  # see ingest/fundamentals.py -- weekly, not daily
FINANCIALS_REFRESH_DAYS = 30  # see ingest/financials.py -- monthly, real quarters only change ~every 90 days
COMPETITORS_REFRESH_DAYS = 30  # see ingest/competitors.py -- monthly; secondary/supplementary comparison data


def main() -> None:
    init_db()
    watchlist = load_watchlist()
    requested = sys.argv[1:] or list(watchlist.keys())

    for ticker in requested:
        entry = watchlist.get(ticker)
        if entry is None:
            print(f"{ticker}: not in config/watchlist.yaml, skipping")
            continue
        print(f"{ticker} ({entry.name})")

        with session_scope() as session:
            symbol = get_or_create_symbol(session, entry.ticker, entry.name, entry.sector)
            symbol_id = symbol.id
            is_watchlisted = symbol.is_watchlisted

        bars = fetch_recent_bars(ticker)
        with session_scope() as session:
            for bar in bars:
                upsert_price_bar(
                    session, symbol_id, bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume
                )
        print(f"  price: {len(bars)} recent bars refreshed")

        sentiment = fetch_sentiment(ticker)
        if sentiment is None:
            print("  sentiment: skipped -- no reading available (missing key, rate limit, or no articles)")
        else:
            with session_scope() as session:
                replace_sentiment_snapshot(
                    session, symbol_id, dt.date.today(), sentiment.overall_sentiment_score, sentiment.article_count
                )
            print(
                f"  sentiment: {sentiment.overall_sentiment_score:+.3f} "
                f"from {sentiment.article_count} article(s)"
            )

        # Reuses ingest/sentiment.py's own disk-cached feed -- costs no
        # extra Alpha Vantage request beyond the fetch_sentiment call above.
        articles = fetch_recent_articles(ticker)
        with session_scope() as session:
            replace_recent_articles(
                session,
                symbol_id,
                [(a.title, a.url, a.source, a.published_at, a.summary) for a in articles],
            )
        print(f"  recent articles: {len(articles)} stored")

        with session_scope() as session:
            existing_fundamentals = latest_fundamentals(session, symbol_id)
            # Extract while the session is still open -- existing_fundamentals
            # is a detached ORM instance once this `with` block exits, and
            # touching its attributes afterward raises DetachedInstanceError
            # (same reasoning as the dashboard's own detached-instance comments).
            existing_fundamentals_date = existing_fundamentals.date if existing_fundamentals else None
        needs_pe_refresh = existing_fundamentals_date is None or (
            dt.date.today() - existing_fundamentals_date
        ).days >= FUNDAMENTALS_REFRESH_DAYS
        if not needs_pe_refresh:
            age_days = (dt.date.today() - existing_fundamentals_date).days
            print(f"  fundamentals: skipped -- refreshed {age_days}d ago (weekly cadence)")
        else:
            overview = fetch_overview(ticker)
            if overview is None:
                print("  fundamentals: skipped -- no reading available (missing key, rate limit, or no data)")
            else:
                with session_scope() as session:
                    replace_fundamentals_snapshot(
                        session,
                        symbol_id,
                        dt.date.today(),
                        overview.pe_ratio,
                        overview.forward_pe,
                        name=overview.name,
                        description=overview.description,
                        industry=overview.industry,
                        exchange=overview.exchange,
                        country=overview.country,
                        address=overview.address,
                        official_site=overview.official_site,
                        market_cap=overview.market_cap,
                    )
                pe_display = f"{overview.pe_ratio:.2f}" if overview.pe_ratio is not None else "n/a"
                print(f"  fundamentals: P/E {pe_display}, profile: {'yes' if overview.description else 'no'}")

        with session_scope() as session:
            last_financials_fetch = latest_quarterly_financials_fetch_time(session, symbol_id)
        now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        needs_financials_refresh = (
            last_financials_fetch is None or (now - last_financials_fetch).days >= FINANCIALS_REFRESH_DAYS
        )
        if not needs_financials_refresh:
            age_days = (now - last_financials_fetch).days
            print(f"  financials: skipped -- refreshed {age_days}d ago (monthly cadence)")
        else:
            quarters = fetch_quarterly_financials(ticker)
            if not quarters:
                print("  financials: skipped -- no reading available (missing key, rate limit, or no data)")
            else:
                with session_scope() as session:
                    replace_quarterly_financials(
                        session,
                        symbol_id,
                        [(q.fiscal_date_ending, q.total_revenue, q.net_income, q.eps) for q in quarters],
                    )
                print(f"  financials: {len(quarters)} quarter(s) stored")

        competitor_tickers = competitors_for(ticker)
        with session_scope() as session:
            last_competitors_fetch = latest_competitor_snapshot_fetch_time(session, symbol_id)
            stored_competitor_tickers = {c.competitor_ticker for c in competitor_snapshots(session, symbol_id)}
        now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        # Also refresh early when config/competitors.yaml gained a ticker
        # that isn't stored yet, so an edit shows up on the next run instead
        # of waiting out the monthly cadence. (A competitor whose fetch keeps
        # failing is simply retried each run.)
        needs_competitors_refresh = (
            last_competitors_fetch is None
            or (now - last_competitors_fetch).days >= COMPETITORS_REFRESH_DAYS
            or not set(competitor_tickers) <= stored_competitor_tickers
        )
        if not competitor_tickers:
            print("  competitors: none configured in config/competitors.yaml")
        elif not needs_competitors_refresh:
            age_days = (now - last_competitors_fetch).days
            print(f"  competitors: skipped -- refreshed {age_days}d ago (monthly cadence)")
        else:
            rows = []
            for competitor_ticker in competitor_tickers:
                snapshot = fetch_competitor_snapshot(competitor_ticker)
                if snapshot is None:
                    continue
                rows.append(
                    (
                        snapshot.ticker,
                        snapshot.name,
                        snapshot.price,
                        snapshot.change,
                        snapshot.pct_change,
                        snapshot.fifty_two_week_low,
                        snapshot.fifty_two_week_high,
                        snapshot.market_cap,
                        snapshot.pe_ratio,
                    )
                )
            with session_scope() as session:
                replace_competitor_snapshots(session, symbol_id, rows)
            print(f"  competitors: {len(rows)}/{len(competitor_tickers)} fetched and stored")

        social = fetch_social_sentiment(ticker)
        if social is None:
            print("  social sentiment: skipped -- no reading available")
        else:
            with session_scope() as session:
                replace_social_sentiment_snapshot(
                    session, symbol_id, dt.date.today(), social.overall_sentiment_score, social.message_count
                )
            print(
                f"  social sentiment: {social.overall_sentiment_score:+.3f} "
                f"from {social.message_count} post(s)"
            )

        posts = fetch_recent_posts(ticker)
        with session_scope() as session:
            replace_recent_social_posts(
                session,
                symbol_id,
                [(p.body, p.username, p.external_id, p.tagged_sentiment, p.created_at) for p in posts],
            )
        print(f"  recent social posts: {len(posts)} stored")

        with session_scope() as session:
            fit = train_symbol(session, symbol_id)
        if fit is None:
            print("  training: skipped -- not enough price history yet")
        else:
            print(f"  training: refit on {fit.n_bars} daily returns (mu={fit.mu:.5f}, sigma={fit.sigma:.5f})")

        with session_scope() as session:
            prediction = predict_symbol(session, symbol_id)
            if prediction is None or prediction.composite is None:
                continue
            previous = latest_recommendation_log(session, symbol_id)
            new_recommendation = prediction.composite.recommendation
            if previous is not None and previous.recommendation == new_recommendation:
                continue
            log_recommendation(session, symbol_id, new_recommendation)
            previous_label = previous.recommendation if previous is not None else None
        print(f"  recommendation: {new_recommendation}")
        if previous_label is not None:
            if not is_watchlisted:
                print("  alert: skipped (symbol not on Watchlist)")
            else:
                with session_scope() as session:
                    closes = price_bars_for_symbol(session, symbol_id)["close"]
                latest_close = float(closes.iloc[-1]) if len(closes) else None
                previous_close = float(closes.iloc[-2]) if len(closes) >= 2 else None
                price_note = f" | {format_price_move(latest_close, previous_close)}" if latest_close else ""
                alerted = send_webhook_alert(
                    f"{ticker}: {previous_label} -> {new_recommendation}{price_note}"
                )
                print(f"  alert: {'sent' if alerted else 'skipped (no webhook configured or send failed)'}")


if __name__ == "__main__":
    main()
