"""One-time migration: adds the company-profile columns (name,
description, industry, exchange, country, address, official_site) to an
existing fundamentals_snapshots table -- that table originally shipped
with just pe_ratio/forward_pe (see storage/models.py::FundamentalsSnapshot,
ingest/fundamentals.py). `Base.metadata.create_all()` (storage/db.py's
init_db()) only creates tables that don't exist yet; it never alters an
existing table's columns, so a database created before this change needs
this migration once. Idempotent -- safe to re-run, skips any column that's
already present.

Usage:
    uv run python scripts/migrate_fundamentals_profile_columns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import inspect, text  # noqa: E402

from stock_predictor.storage.db import get_engine, init_db  # noqa: E402

NEW_COLUMNS = ("name", "description", "industry", "exchange", "country", "address", "official_site")


def main() -> None:
    init_db()
    engine = get_engine()
    inspector = inspect(engine)
    if "fundamentals_snapshots" not in inspector.get_table_names():
        print("fundamentals_snapshots table doesn't exist yet -- nothing to migrate (init_db() will create it fresh).")
        return

    existing_columns = {col["name"] for col in inspector.get_columns("fundamentals_snapshots")}
    with engine.begin() as conn:
        for column in NEW_COLUMNS:
            if column in existing_columns:
                print(f"{column}: already present, skipping")
                continue
            conn.execute(text(f"ALTER TABLE fundamentals_snapshots ADD COLUMN {column} VARCHAR"))
            print(f"{column}: added")


if __name__ == "__main__":
    main()
