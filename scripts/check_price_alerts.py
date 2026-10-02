"""Polls current price for every watchlisted symbol with a price-alert
threshold set, and fires a Discord/Slack webhook the moment price crosses
that threshold -- the "closer to real-time" counterpart to
scripts/refresh_live_data.py's once/day recommendation-change alert.
Deliberately its own separate script (not folded into refresh_live_data.py):
that script is heavy (price history + sentiment + fundamentals + financials
+ competitors + training per symbol) and runs once/day; this one only
calls Finnhub's cheap /quote endpoint (see ingest/quote.py) for a small
subset of symbols, so it's safe to run every ~15 minutes during market
hours (see .github/workflows/check-price-alerts.yml) without coming close
to Finnhub's free-tier 60-requests/minute cap.

Usage:
    uv run python scripts/check_price_alerts.py

Requires FINNHUB_API_KEY (same as the rest of this app's Finnhub calls)
and DISCORD_WEBHOOK_URL (see alerts/notifier.py) -- skipped per-symbol
(not an error) if a quote can't be fetched; the whole run never aborts
over one bad symbol.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_predictor.alerts.notifier import format_price_move, send_webhook_alert  # noqa: E402
from stock_predictor.ingest.quote import fetch_quote  # noqa: E402
from stock_predictor.storage.db import init_db, session_scope  # noqa: E402
from stock_predictor.storage.repository import (  # noqa: E402
    set_alert_triggered_state,
    symbols_with_alert_thresholds,
)


def main() -> None:
    init_db()

    with session_scope() as session:
        targets = [
            (
                symbol.id,
                symbol.ticker,
                alert.upper_price,
                alert.lower_price,
                alert.above_triggered,
                alert.below_triggered,
            )
            for symbol, alert in symbols_with_alert_thresholds(session)
        ]

    if not targets:
        print("No watchlisted symbols have a price alert threshold set -- nothing to check.")
        return

    for symbol_id, ticker, upper_price, lower_price, above_triggered, below_triggered in targets:
        quote = fetch_quote(ticker)
        if quote is None:
            print(f"{ticker}: skipped -- no quote available (missing key, rate limit, or unknown symbol)")
            continue

        price = quote.current_price
        print(f"{ticker}: ${price:.2f}")

        if upper_price is not None:
            if price >= upper_price and not above_triggered:
                alerted = send_webhook_alert(
                    f"{ticker} crossed above your ${upper_price:.2f} alert threshold: "
                    f"now {format_price_move(price, quote.previous_close)}"
                )
                status = "sent" if alerted else "skipped (webhook not configured or failed)"
                print(f"  above ${upper_price:.2f}: alert {status}")
                with session_scope() as session:
                    set_alert_triggered_state(session, symbol_id, above=True)
            elif price < upper_price and above_triggered:
                # Back within range -- re-arm so a future re-crossing alerts again.
                with session_scope() as session:
                    set_alert_triggered_state(session, symbol_id, above=False)

        if lower_price is not None:
            if price <= lower_price and not below_triggered:
                alerted = send_webhook_alert(
                    f"{ticker} crossed below your ${lower_price:.2f} alert threshold: "
                    f"now {format_price_move(price, quote.previous_close)}"
                )
                status = "sent" if alerted else "skipped (webhook not configured or failed)"
                print(f"  below ${lower_price:.2f}: alert {status}")
                with session_scope() as session:
                    set_alert_triggered_state(session, symbol_id, below=True)
            elif price > lower_price and below_triggered:
                with session_scope() as session:
                    set_alert_triggered_state(session, symbol_id, below=False)


if __name__ == "__main__":
    main()
