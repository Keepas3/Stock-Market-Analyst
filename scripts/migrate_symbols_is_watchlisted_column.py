"""One-time migration: adds the `is_watchlisted` column to an existing
symbols table (see storage/models.py::Symbol, dashboard/views/main.py) --
needed for the Main page's per-row "Add to Watchlist" checkbox. Same
reasoning as scripts/migrate_fundamentals_market_cap_column.py:
`create_all()` never alters an existing table's columns. Idempotent --
safe to re-run.

Usage:
    uv run python scripts/migrate_symbols_is_watchlisted_column.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import inspect, text  # noqa: E402

from stock_predictor.storage.db import get_engine, init_db  # noqa: E402


def main() -> None:
    init_db()
    engine = get_engine()
    inspector = inspect(engine)
    if "symbols" not in inspector.get_table_names():
        print("symbols table doesn't exist yet -- nothing to migrate (init_db() will create it fresh).")
        return

    existing_columns = {col["name"] for col in inspector.get_columns("symbols")}
    if "is_watchlisted" in existing_columns:
        print("is_watchlisted: already present, skipping")
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE symbols ADD COLUMN is_watchlisted BOOLEAN DEFAULT 0"))
    print("is_watchlisted: added (defaulted to False for existing rows)")


if __name__ == "__main__":
    main()
