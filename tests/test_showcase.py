from __future__ import annotations

import json

import pytest

from stock_predictor import config
from stock_predictor.assistant import showcase
from stock_predictor.model import overrides_sync


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    return tmp_path


def make_chat(chat_id="c1", title="NVDA question", published="2026-02-01T00:00:00Z", n=2):
    transcript = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"message {i}"} for i in range(n)]
    return showcase._clean_chat({"id": chat_id, "title": title, "published": published, "transcript": transcript})


def test_no_file_means_no_published_chats():
    assert showcase.load_chats() == []


def test_chat_from_conversation_keeps_only_visible_text():
    chat = showcase.chat_from_conversation(
        "abc",
        "  My   title ",
        [{"role": "user", "text": "hi", "api_junk": "dropped"}, {"role": "assistant", "text": "hello"}],
    )
    assert chat["title"] == "My title"
    assert chat["transcript"] == [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]
    assert set(chat) == {"id", "title", "published", "transcript"}


def test_empty_or_invalid_conversations_cannot_be_published():
    assert showcase.chat_from_conversation("a", "t", []) is None
    assert showcase.chat_from_conversation("a", "t", [{"role": "system", "text": "x"}]) is None


def test_text_and_length_are_clamped():
    chat = showcase.chat_from_conversation(
        "a", "t" * 500, [{"role": "user", "text": "x" * (showcase.MAX_TEXT + 50)}] * (showcase.MAX_MESSAGES + 20)
    )
    assert len(chat["title"]) == showcase.MAX_TITLE
    assert len(chat["transcript"]) == showcase.MAX_MESSAGES
    assert len(chat["transcript"][0]["text"]) == showcase.MAX_TEXT


@pytest.mark.parametrize(
    "text",
    ["", "not json", "[]", '{"version": 2, "chats": []}', '{"version": 1, "chats": "x"}', '{"version": 1, "chats": [1, null, {"id": 5}]}'],
)
def test_parse_degrades_to_empty_on_garbage(text):
    assert showcase.parse(text)["chats"] == []


def test_parse_dedupes_and_caps():
    chats = [make_chat(f"c{i}") for i in range(showcase.MAX_CHATS + 5)] + [make_chat("c0")]
    parsed = showcase.parse(json.dumps({"version": 1, "chats": chats}))
    assert len(parsed["chats"]) == showcase.MAX_CHATS
    assert len({c["id"] for c in parsed["chats"]}) == showcase.MAX_CHATS


def test_publish_then_visitors_can_load_it_newest_first():
    result = showcase.publish(make_chat("old", published="2026-01-01T00:00:00Z"))
    showcase.publish(make_chat("new", "Newer", published="2026-03-01T00:00:00Z"))
    assert not result.ok and "locally only" in result.message  # no GitHub config in this test
    assert [c["id"] for c in showcase.load_chats()] == ["new", "old"]


def test_republishing_replaces_instead_of_duplicating():
    showcase.publish(make_chat("a", title="First", n=2))
    showcase.publish(make_chat("a", title="Updated", n=4))
    chats = showcase.load_chats()
    assert len(chats) == 1 and chats[0]["title"] == "Updated" and len(chats[0]["transcript"]) == 4


def test_unpublish_removes_only_that_chat():
    showcase.publish(make_chat("a"))
    showcase.publish(make_chat("b"))
    showcase.unpublish("a")
    assert [c["id"] for c in showcase.load_chats()] == ["b"]


def test_publishing_beyond_the_cap_drops_the_oldest():
    for i in range(showcase.MAX_CHATS + 2):
        showcase.publish(make_chat(f"c{i}", published=f"2026-01-{i + 1:02d}T00:00:00Z"))
    ids = [c["id"] for c in showcase.load_chats()]
    assert len(ids) == showcase.MAX_CHATS and "c0" not in ids and f"c{showcase.MAX_CHATS + 1}" in ids


def test_cache_follows_external_edits(isolated):
    showcase.publish(make_chat("a", title="Before"))
    assert showcase.load_chats()[0]["title"] == "Before"
    path = showcase.showcase_path()
    path.write_text(json.dumps({"version": 1, "chats": [make_chat("a", title="After")]}), encoding="utf-8")
    assert showcase.load_chats()[0]["title"] == "After"


def test_with_github_configured_the_change_is_replayed_onto_the_remote_file(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPO", "o/r")
    seen = {}

    def fake_commit_file(path, transform, message):
        seen["path"], seen["message"] = path, message
        remote = json.dumps({"version": 1, "chats": [make_chat("remote-only")]})
        seen["pushed"] = showcase.parse(transform(remote))
        return overrides_sync.SyncResult(True, "Saved to git.")

    monkeypatch.setattr(overrides_sync, "commit_file", fake_commit_file)
    result = showcase.publish(make_chat("mine"))

    assert result.ok and seen["path"] == "config/showcase_chats.json"
    assert {c["id"] for c in seen["pushed"]["chats"]} == {"remote-only", "mine"}  # remote chats survive
    assert "Assistant chat" in seen["message"]
    assert "message 0" not in seen["message"]  # commit message never carries chat text
