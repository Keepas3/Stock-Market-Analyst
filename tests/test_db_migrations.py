from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from stock_predictor.storage import db


def _fresh_engine(tmp_path):
    """A real sqlite file (not :memory:, which loses state on close) --
    db._apply_column_migrations opens its own connection via engine.begin(),
    so an in-memory engine created here wouldn't share the same database.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    return engine


def test_init_db_adds_missing_column_to_a_pre_existing_table(tmp_path, monkeypatch):
    """Reproduces the real "no such column: symbols.is_watchlisted" crash
    users hit on a database created before that column existed -- a bare
    `symbols` table (old schema) must come out queryable on that column
    after init_db(), with no manual migration script required.
    """
    engine = _fresh_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE symbols (id INTEGER PRIMARY KEY, ticker VARCHAR, name VARCHAR)"))

    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_SessionLocal", None)

    db.init_db()

    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("symbols")}
    assert "is_watchlisted" in columns

    # Genuinely queryable, not just present in the schema -- the real
    # failure mode was an OperationalError on SELECT, not a missing column
    # in isolation.
    with engine.connect() as conn:
        conn.execute(text("SELECT is_watchlisted FROM symbols"))


def test_init_db_is_idempotent_on_an_already_migrated_table(tmp_path, monkeypatch):
    engine = _fresh_engine(tmp_path)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_SessionLocal", None)

    db.init_db()
    db.init_db()  # must not raise "duplicate column" on the second pass

    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("symbols")]
    assert columns.count("is_watchlisted") == 1


def test_init_db_skips_migrations_for_tables_that_do_not_exist_yet(tmp_path, monkeypatch):
    """A brand-new database: create_all() makes every table with every
    current column already, so _apply_column_migrations has nothing to do
    -- this just confirms it doesn't error out trying to ALTER a table
    that create_all() hasn't created (e.g. a symbol/table pair removed in
    a future refactor).
    """
    engine = _fresh_engine(tmp_path)
    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_SessionLocal", None)
    monkeypatch.setattr(db, "_COLUMN_MIGRATIONS", (("a_table_that_does_not_exist", "some_column", "VARCHAR"),))

    db.init_db()  # must not raise
