from __future__ import annotations

import base64

import pytest
import yaml

from stock_predictor import config
from stock_predictor.model import overrides_store, overrides_sync
from stock_predictor.model.tuning import PARAM_SPECS, TunableParams


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    return tmp_path


def test_missing_file_means_no_overrides():
    assert overrides_store.all_overrides() == {}
    assert overrides_store.params_for("AAPL") == TunableParams()


def test_apply_changes_round_trips_and_logs_history():
    entries = overrides_store.apply_changes("AAPL", {"news_weight": 0.5}, "news is noisy", "ai")

    assert len(entries) == 1
    assert overrides_store.overrides_for("AAPL") == {"news_weight": 0.5}
    assert overrides_store.params_for("AAPL").news_weight == 0.5
    history = overrides_store.history("AAPL")
    assert history[-1]["old"] == 1.0 and history[-1]["new"] == 0.5
    assert history[-1]["source"] == "ai" and history[-1]["reason"] == "news is noisy"
    assert overrides_store.overrides_for("MSFT") == {}


def test_values_are_snapped_and_validated_all_or_nothing():
    with pytest.raises(ValueError):
        overrides_store.apply_changes("AAPL", {"news_weight": 0.5, "ma_weight": 9}, "x", "ai")
    assert overrides_store.overrides_for("AAPL") == {}

    overrides_store.apply_changes("AAPL", {"buy_threshold": 2.3}, "x", "ai")
    assert overrides_store.overrides_for("AAPL") == {"buy_threshold": 2.5}


def test_inconsistent_combination_is_rejected():
    with pytest.raises(ValueError, match="pe_expensive_threshold"):
        overrides_store.apply_changes("AAPL", {"pe_expensive_threshold": 16, "pe_value_threshold": 20}, "x", "ai")
    assert overrides_store.overrides_for("AAPL") == {}


def test_setting_a_param_back_to_default_clears_it():
    overrides_store.apply_changes("AAPL", {"news_weight": 0.5}, "x", "ai")
    overrides_store.apply_changes("AAPL", {"news_weight": PARAM_SPECS["news_weight"].default}, "x", "ai")
    assert overrides_store.all_overrides() == {}


def test_noop_change_writes_nothing():
    assert overrides_store.apply_changes("AAPL", {"news_weight": 1.0}, "x", "ai") == []
    assert not overrides_store.overrides_path().exists()


def test_reset_and_undo():
    overrides_store.apply_changes("AAPL", {"news_weight": 0.5, "ma_weight": 1.5}, "x", "ai")
    overrides_store.reset("AAPL", ["news_weight"], "reset one")
    assert overrides_store.overrides_for("AAPL") == {"ma_weight": 1.5}

    ma_entry = next(e for e in overrides_store.history("AAPL") if e["param"] == "ma_weight")
    overrides_store.undo(ma_entry["id"])
    assert overrides_store.overrides_for("AAPL") == {}
    assert overrides_store.history("AAPL")[-1]["source"] == "undo"

    with pytest.raises(ValueError, match="No change"):
        overrides_store.undo("missing")


def test_hand_edited_garbage_is_sanitized_on_load(isolated_config_dir):
    (isolated_config_dir / "model_overrides.yaml").write_text(
        "overrides:\n  AAPL: {ma_weight: 99, bogus: 1, pe_weight: abc}\n  MSFT: [1, 2]\nhistory: nope\n",
        encoding="utf-8",
    )
    assert overrides_store.all_overrides() == {"AAPL": {"ma_weight": 2}}
    assert overrides_store.history() == []


def test_unparseable_yaml_degrades_to_empty(isolated_config_dir):
    (isolated_config_dir / "model_overrides.yaml").write_text("overrides: [unclosed", encoding="utf-8")
    assert overrides_store.all_overrides() == {}


def test_history_is_capped(monkeypatch):
    monkeypatch.setattr(overrides_store, "HISTORY_LIMIT", 8)
    for i in range(14):
        overrides_store.apply_changes("AAPL", {"ma_weight": 0.5 if i % 2 == 0 else 1.5}, f"n{i}", "ai")
    assert len(overrides_store.history()) == 8


def test_cache_picks_up_external_edits(isolated_config_dir):
    overrides_store.apply_changes("AAPL", {"news_weight": 0.5}, "x", "ai")
    assert overrides_store.overrides_for("AAPL") == {"news_weight": 0.5}
    path = overrides_store.overrides_path()
    path.write_text("overrides:\n  AAPL: {news_weight: 1.5}\nhistory: []\n", encoding="utf-8")
    assert overrides_store.overrides_for("AAPL") == {"news_weight": 1.5}


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")


def test_sync_not_configured_is_a_clear_local_only_result(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    entries = [e.as_dict() for e in overrides_store.apply_changes("AAPL", {"news_weight": 0.5}, "x", "ai")]
    result = overrides_sync.commit_entries(entries)
    assert not result.ok
    assert "locally only" in result.message


def test_sync_replays_entries_onto_the_remote_file(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t0k")
    monkeypatch.setenv("GITHUB_REPO", "o/r")
    entries = [e.as_dict() for e in overrides_store.apply_changes("AAPL", {"news_weight": 0.5}, "x", "ai")]

    remote_text = "overrides:\n  MSFT: {ma_weight: 1.5}\nhistory: []\n"
    calls = {}

    def fake_get(url, headers, params, timeout):
        assert headers["Authorization"] == "Bearer t0k"
        return _Resp(200, {"sha": "abc", "content": base64.b64encode(remote_text.encode()).decode()})

    def fake_put(url, headers, json, timeout):
        calls["payload"] = json
        return _Resp(200, {})

    monkeypatch.setattr(overrides_sync.requests, "get", fake_get)
    monkeypatch.setattr(overrides_sync.requests, "put", fake_put)

    result = overrides_sync.commit_entries(entries)

    assert result.ok
    pushed = yaml.safe_load(base64.b64decode(calls["payload"]["content"]))
    assert pushed["overrides"] == {"MSFT": {"ma_weight": 1.5}, "AAPL": {"news_weight": 0.5}}
    assert calls["payload"]["sha"] == "abc"
    assert calls["payload"]["message"] == "AI tuning: 1 change(s) to AAPL"


def test_sync_failure_is_reported_not_raised(monkeypatch):
    import requests

    monkeypatch.setenv("GITHUB_TOKEN", "t0k")
    monkeypatch.setenv("GITHUB_REPO", "o/r")

    def boom(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(overrides_sync.requests, "get", boom)
    entries = [e.as_dict() for e in overrides_store.apply_changes("AAPL", {"news_weight": 0.5}, "x", "ai")]
    result = overrides_sync.commit_entries(entries)
    assert not result.ok
    assert "not saved to git" in result.message
