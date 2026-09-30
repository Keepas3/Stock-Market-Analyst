"""Engine/session setup -- the local stocks.db SQLite file. Direct copy of
soccer-predictor's storage/db.py, minus the Turso branch (no deployment
story yet for this project -- see the plan's "Out of scope").
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from stock_predictor.config import DATA_DIR, DB_PATH
from stock_predictor.storage.models import Base

_engine = None
_SessionLocal: sessionmaker | None = None

# Columns added to a model after its table first shipped.
# Base.metadata.create_all() only creates tables that don't exist yet --
# it never alters an existing table's columns -- so a database created
# before one of these was added would otherwise raise "no such column"
# the first time it's queried (confirmed live: symbols.is_watchlisted,
# fundamentals_snapshots.market_cap and its profile columns have each hit
# this in the wild). init_db() now applies every entry here automatically
# on startup (idempotent -- skips a column that's already present, and a
# brand-new database already has all of them via create_all() alone), so
# this list is the one place to add a line when a future column joins an
# existing table, instead of a one-off scripts/migrate_*.py that's easy
# to forget to run.
_COLUMN_MIGRATIONS = (
    ("fundamentals_snapshots", "market_cap", "FLOAT"),
    ("fundamentals_snapshots", "name", "VARCHAR"),
    ("fundamentals_snapshots", "description", "VARCHAR"),
    ("fundamentals_snapshots", "industry", "VARCHAR"),
    ("fundamentals_snapshots", "exchange", "VARCHAR"),
    ("fundamentals_snapshots", "country", "VARCHAR"),
    ("fundamentals_snapshots", "address", "VARCHAR"),
    ("fundamentals_snapshots", "official_site", "VARCHAR"),
    ("symbols", "is_watchlisted", "BOOLEAN DEFAULT 0"),
)


def get_engine():
    global _engine
    if _engine is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    return _engine


def _apply_column_migrations(engine) -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, column, ddl_type in _COLUMN_MIGRATIONS:
            if table not in existing_tables:
                continue  # a brand-new table -- create_all() already made it with every current column
            existing_columns = {col["name"] for col in inspector.get_columns(table)}
            if column in existing_columns:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    _apply_column_migrations(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), future=True)
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
