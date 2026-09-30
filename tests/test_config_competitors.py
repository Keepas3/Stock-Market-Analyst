from __future__ import annotations

import pytest

from stock_predictor import config


@pytest.fixture(autouse=True)
def _clear_competitor_cache():
    config.load_competitor_map.cache_clear()
    yield
    config.load_competitor_map.cache_clear()


def test_load_competitor_map_parses_real_shaped_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    (tmp_path / "competitors.yaml").write_text(
        "competitors:\n  NVDA: [AMD, INTC, AVGO]\n  AAPL: [MSFT, GOOGL]\n", encoding="utf-8"
    )

    result = config.load_competitor_map()

    assert result == {"NVDA": ["AMD", "INTC", "AVGO"], "AAPL": ["MSFT", "GOOGL"]}


def test_load_competitor_map_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    assert config.load_competitor_map() == {}


def test_load_competitor_map_empty_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    (tmp_path / "competitors.yaml").write_text("", encoding="utf-8")
    assert config.load_competitor_map() == {}


def test_competitors_for_known_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    (tmp_path / "competitors.yaml").write_text("competitors:\n  NVDA: [AMD, INTC]\n", encoding="utf-8")

    assert config.competitors_for("NVDA") == ["AMD", "INTC"]


def test_competitors_for_unknown_ticker_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    (tmp_path / "competitors.yaml").write_text("competitors:\n  NVDA: [AMD, INTC]\n", encoding="utf-8")

    assert config.competitors_for("ZZZZ") == []


def test_real_competitors_yaml_loads_and_covers_the_real_watchlist():
    """Not mocked -- confirms the real config/competitors.yaml shipped
    with this repo parses correctly and covers every real watchlist
    ticker (config/watchlist.yaml), so no symbol silently has zero
    competitors configured.
    """
    config.load_competitor_map.cache_clear()
    watchlist = config.load_watchlist()
    competitor_map = config.load_competitor_map()

    assert competitor_map  # not empty
    for ticker in watchlist:
        assert ticker in competitor_map, f"{ticker} has no entry in config/competitors.yaml"
        assert len(competitor_map[ticker]) >= 2
