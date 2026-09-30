"""Downloads full daily price history for every watchlisted symbol.

Usage:
    uv run python scripts/fetch_historical_data.py [TICKER ...]

With no arguments, fetches every symbol in config/watchlist.yaml. Safe to
re-run any time -- upsert_price_bar overwrites, never duplicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_predictor.config import load_watchlist  # noqa: E402
from stock_predictor.ingest.price_history import fetch_full_history  # noqa: E402
from stock_predictor.storage.db import init_db, session_scope  # noqa: E402
from stock_predictor.storage.repository import get_or_create_symbol, upsert_price_bar  # noqa: E402


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
            symbol_id = get_or_create_symbol(session, entry.ticker, entry.name, entry.sector).id

        bars = fetch_full_history(ticker)
        if not bars:
            print("  no price history returned -- skipping")
            continue

        with session_scope() as session:
            for bar in bars:
                upsert_price_bar(
                    session, symbol_id, bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume
                )
        print(f"  {len(bars)} daily bars ingested ({bars[0].date} to {bars[-1].date})")


if __name__ == "__main__":
    main()
