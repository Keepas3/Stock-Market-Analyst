"""Refits the return model for every watchlisted symbol from its currently
stored price history -- no new fetching (use refresh_live_data.py for that).

Usage:
    uv run python scripts/run_training.py [TICKER ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_predictor.config import load_watchlist  # noqa: E402
from stock_predictor.prediction.training import train_symbol  # noqa: E402
from stock_predictor.storage.db import init_db, session_scope  # noqa: E402
from stock_predictor.storage.repository import symbol_by_ticker  # noqa: E402


def main() -> None:
    init_db()
    watchlist = load_watchlist()
    requested = sys.argv[1:] or list(watchlist.keys())

    for ticker in requested:
        with session_scope() as session:
            symbol = symbol_by_ticker(session, ticker)
            if symbol is None:
                print(f"{ticker}: not seeded yet -- run fetch_historical_data.py first")
                continue
            fit = train_symbol(session, symbol.id)
        if fit is None:
            print(f"{ticker}: skipped -- not enough price history yet")
        else:
            print(f"{ticker}: refit on {fit.n_bars} daily returns (mu={fit.mu:.5f}, sigma={fit.sigma:.5f})")


if __name__ == "__main__":
    main()
