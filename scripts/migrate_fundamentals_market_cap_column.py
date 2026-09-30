"""One-time migration: adds the `market_cap` column to an existing
fundamentals_snapshots table (see storage/models.py::FundamentalsSnapshot,
ingest/fundamentals.py) -- needed so a watchlist symbol's own market cap
can appear alongside its competitors' in the Competitors table (see
dashboard/components.py::render_competitors_table). Same reasoning as
scripts/migrate_fundamentals_profile_columns.py: `create_all()` never
alters an existing table's columns. Idempotent -- safe to re-run.

Usage:
    uv run python scripts/migrate_fundamentals_market_cap_column.py
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
    if "fundamentals_snapshots" not in inspector.get_table_names():
        print("fundamentals_snapshots table doesn't exist yet -- nothing to migrate (init_db() will create it fresh).")
        return

    existing_columns = {col["name"] for col in inspector.get_columns("fundamentals_snapshots")}
    if "market_cap" in existing_columns:
        print("market_cap: already present, skipping")
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE fundamentals_snapshots ADD COLUMN market_cap FLOAT"))
    print("market_cap: added")


if __name__ == "__main__":
    main()
