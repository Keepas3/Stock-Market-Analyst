"""Per-company math overrides, stored in a git-tracked YAML file
(config/model_overrides.yaml) rather than the SQLite DB: stocks.db is
rewritten by cron and Streamlit Cloud redeploys from git, so DB writes made
by the live app would be wiped, and the Discord-alert cron would never see
them. A file in the repo is seen by both. See model/overrides_sync.py for
how a change made on the deployed app gets committed back.

File shape (hand-editable):

    overrides:
      AAPL: {pe_value_threshold: 20, news_weight: 0.5}
    history:            # newest last, capped at HISTORY_LIMIT
      - {id: ..., ts: ..., ticker: AAPL, param: ..., old: ..., new: ..., reason: ..., source: ai}

Setting a param back to its default removes it from `overrides`. Everything
read from the file is sanitized (model/tuning.py::sanitize_overrides), so a
bad hand edit degrades to defaults instead of crashing a page or the cron.
"""

from __future__ import annotations

import copy
import datetime as dt
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from stock_predictor import config
from stock_predictor.model.tuning import (
    PARAM_SPECS,
    TunableParams,
    coerce_value,
    combined_errors,
    resolve_params,
    sanitize_overrides,
)

OVERRIDES_FILENAME = "model_overrides.yaml"
HISTORY_LIMIT = 200
_HEADER = (
    "# Per-company overrides of the recommendation math (see src/stock_predictor/model/tuning.py\n"
    "# for every parameter, default and bound). Written by the Assistant page and safe to edit by hand.\n"
    "# Removing a line restores that default.\n"
)

_lock = threading.Lock()
_cache: dict[tuple[str, int, int], dict] = {}


@dataclass(frozen=True)
class Change:
    id: str
    ts: str
    ticker: str
    param: str
    old: float
    new: float
    reason: str
    source: str  # "ai" | "owner" | "undo" | "reset"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts,
            "ticker": self.ticker,
            "param": self.param,
            "old": self.old,
            "new": self.new,
            "reason": self.reason,
            "source": self.source,
        }


def overrides_path() -> Path:
    # Read at call time (not import time) so tests can monkeypatch CONFIG_DIR.
    return config.CONFIG_DIR / OVERRIDES_FILENAME


def _empty() -> dict:
    return {"overrides": {}, "history": []}


def _normalize(raw: object) -> dict:
    """Sanitize a parsed YAML document into the canonical in-memory shape."""
    store = _empty()
    if not isinstance(raw, dict):
        return store
    overrides = raw.get("overrides")
    if isinstance(overrides, dict):
        for ticker, params in overrides.items():
            if not isinstance(ticker, str) or not isinstance(params, dict):
                continue
            clean = sanitize_overrides(params)
            if clean and not combined_errors(TunableParams(**clean)):
                store["overrides"][ticker] = clean
    history = raw.get("history")
    if isinstance(history, list):
        for entry in history:
            if isinstance(entry, dict) and {"id", "ticker", "param", "old", "new"} <= entry.keys():
                store["history"].append(entry)
    store["history"] = store["history"][-HISTORY_LIMIT:]
    return store


def parse_store(text: str) -> dict:
    try:
        return _normalize(yaml.safe_load(text))
    except yaml.YAMLError:
        return _empty()


def load_store() -> dict:
    """The current store (a fresh copy -- safe to mutate). Cached by file
    mtime so the Companies table, which calls this per render, doesn't
    re-parse YAML 51 times.
    """
    path = overrides_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return _empty()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _cache.get(key)
    if cached is None:
        cached = parse_store(path.read_text(encoding="utf-8"))
        _cache.clear()
        _cache[key] = cached
    return copy.deepcopy(cached)


def dump_store(store: dict) -> str:
    return _HEADER + yaml.safe_dump(store, sort_keys=False, default_flow_style=False, allow_unicode=True)


def _write_store(store: dict) -> None:
    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".model_overrides.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(dump_store(store))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def overrides_for(ticker: str) -> dict[str, float]:
    return dict(load_store()["overrides"].get(ticker, {}))


def params_for(ticker: str) -> TunableParams:
    return resolve_params(overrides_for(ticker))


def all_overrides() -> dict[str, dict[str, float]]:
    return load_store()["overrides"]


def history(ticker: str | None = None) -> list[dict]:
    entries = load_store()["history"]
    return [e for e in entries if ticker is None or e["ticker"] == ticker]


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def apply_entries(store: dict, entries: list[dict]) -> None:
    """Replay already-built change entries onto `store` (mutates it). Used
    both for the local write and to re-apply a turn's changes onto the
    freshly fetched remote file (model/overrides_sync.py), so a hand edit
    made on GitHub in the meantime isn't clobbered. Entries whose id is
    already in the history are skipped (idempotent).
    """
    seen = {e["id"] for e in store["history"]}
    for entry in entries:
        if entry["id"] in seen:
            continue
        ticker, param, new = entry["ticker"], entry["param"], entry["new"]
        per_ticker = store["overrides"].setdefault(ticker, {})
        if new == PARAM_SPECS[param].default:
            per_ticker.pop(param, None)
        else:
            per_ticker[param] = new
        if not per_ticker:
            store["overrides"].pop(ticker, None)
        store["history"].append(entry)
        seen.add(entry["id"])
    store["history"] = store["history"][-HISTORY_LIMIT:]


def apply_changes(
    ticker: str,
    changes: dict[str, object],
    reason: str,
    source: str,
) -> list[Change]:
    """Validate and apply `changes` ({param: value}) to one company. All or
    nothing: ValueError (safe to show the model/user) if any value is out of
    bounds or the resulting set is inconsistent. Returns the entries that
    actually changed something (no-ops are dropped).
    """
    clean = {name: coerce_value(name, value) for name, value in changes.items()}
    reason = " ".join(str(reason).split())[:300]
    with _lock:
        store = load_store()
        current = store["overrides"].get(ticker, {})
        errors = combined_errors(TunableParams(**{**current, **clean}))
        if errors:
            raise ValueError(" ".join(errors))
        entries = []
        for name, new in clean.items():
            old = current.get(name, PARAM_SPECS[name].default)
            if new == old:
                continue
            entries.append(
                Change(
                    id=uuid.uuid4().hex[:12],
                    ts=_now(),
                    ticker=ticker,
                    param=name,
                    old=old,
                    new=new,
                    reason=reason,
                    source=source,
                )
            )
        if entries:
            apply_entries(store, [e.as_dict() for e in entries])
            _write_store(store)
        return entries


def reset(ticker: str, params: list[str] | None, reason: str, source: str = "reset") -> list[Change]:
    """Restore defaults for `params` (all of this company's overrides if
    None)."""
    current = overrides_for(ticker)
    names = list(current) if params is None else [p for p in params if p in current]
    if not names:
        return []
    return apply_changes(ticker, {n: PARAM_SPECS[n].default for n in names}, reason, source)


def undo(entry_id: str) -> list[Change]:
    """Revert one history entry by setting its param back to the value it
    had before (a new, logged change -- history is append-only)."""
    for entry in reversed(history()):
        if entry["id"] == entry_id:
            return apply_changes(
                entry["ticker"],
                {entry["param"]: entry["old"]},
                f"Undo of change {entry_id}",
                "undo",
            )
    raise ValueError(f"No change with id {entry_id}.")
