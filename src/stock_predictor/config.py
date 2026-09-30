"""Loads .env and config/watchlist.yaml -- mirrors soccer-predictor's own
config.py shape (League/load_leagues()) almost exactly, just simpler:
tickers need no fuzzy name-matching/alias table the way club names did.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "stocks.db"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Symbol:
    ticker: str
    name: str
    sector: str | None = None


@lru_cache
def load_watchlist() -> dict[str, Symbol]:
    raw = yaml.safe_load((CONFIG_DIR / "watchlist.yaml").read_text(encoding="utf-8"))
    return {
        entry["ticker"]: Symbol(
            ticker=entry["ticker"],
            name=entry["name"],
            sector=entry.get("sector"),
        )
        for entry in raw["watchlist"]
    }


@lru_cache
def load_competitor_map() -> dict[str, list[str]]:
    """Ticker -> hand-curated list of real competitor tickers (see
    config/competitors.yaml's own header comment). Alpha Vantage's free
    tier has no "list companies by industry" endpoint (confirmed against
    its documented API surface) to derive this live, so it's a small
    manually-maintained fact list -- same pattern soccer-predictor uses
    for config/captains.yaml/config/star_players.yaml, not fabricated or
    algorithmically inferred. Empty dict (not an error) if the file is
    missing/empty.
    """
    path = CONFIG_DIR / "competitors.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("competitors", {}) or {}


def competitors_for(ticker: str) -> list[str]:
    return load_competitor_map().get(ticker, [])


def alpha_vantage_api_key() -> str | None:
    return os.environ.get("ALPHA_VANTAGE_API_KEY") or None


def discord_webhook_url() -> str | None:
    return os.environ.get("DISCORD_WEBHOOK_URL") or None
