from __future__ import annotations

import json

import pytest

from stock_predictor.assistant import conversations as cv
from stock_predictor.assistant.chat import Usage
from stock_predictor.assistant.conversations import (
    Conversation,
    ConversationBook,
    block_to_dict,
    conversation_from_dict,
    conversation_to_dict,
    dump_all,
    load_all,
    title_from,
)
from stock_predictor.assistant.tools import Proposal


@pytest.fixture(autouse=True)
def watchlist(monkeypatch):
    monkeypatch.setattr(cv, "load_watchlist", lambda: {"AAPL": None, "MSFT": None})


def make(conv_id="c1", title="Chat", updated="2026-01-02T00:00:00.000000Z", messages=True) -> Conversation:
    conv = Conversation(id=conv_id, title=title, created="2026-01-01T00:00:00.000000Z", updated=updated)
    if messages:
        conv.api_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        conv.transcript = [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"}]
    return conv


def test_sdk_blocks_become_api_ready_dicts_without_none_fields():
    from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock

    assert block_to_dict(TextBlock(type="text", text="hi", citations=None)) == {"type": "text", "text": "hi"}
    thinking = block_to_dict(ThinkingBlock(type="thinking", thinking="", signature="sig=="))
    assert thinking == {"type": "thinking", "thinking": "", "signature": "sig=="}
    tool = block_to_dict(ToolUseBlock(type="tool_use", id="t1", name="get_signals", input={"ticker": "AAPL"}))
    assert tool["input"] == {"ticker": "AAPL"} and tool["id"] == "t1" and "caller" not in tool or tool["caller"] is not None


def test_plain_dicts_and_simple_objects_are_handled():
    from types import SimpleNamespace

    assert block_to_dict({"type": "tool_result", "tool_use_id": "x", "content": "{}"}) == {
        "type": "tool_result",
        "tool_use_id": "x",
        "content": "{}",
    }
    assert block_to_dict(SimpleNamespace(type="text", text="yo", extra=None)) == {"type": "text", "text": "yo"}


def test_round_trip_preserves_everything_that_matters():
    conv = make(title="NVDA question")
    conv.usage = Usage(input_tokens=10, output_tokens=5, cache_write_tokens=3, cache_read_tokens=2)
    conv.ctx.tainted = True
    conv.ctx.proposals = {
        "p1": Proposal(id="p1", ticker="AAPL", changes={"news_weight": 0.5}, reason="noisy", created="t", downgraded=True)
    }
    restored = conversation_from_dict(json.loads(json.dumps(conversation_to_dict(conv))))

    assert restored.id == "c1" and restored.title == "NVDA question"
    assert restored.api_messages == conv.api_messages
    assert restored.transcript == conv.transcript
    assert restored.usage == conv.usage
    assert restored.ctx.tainted is True
    assert restored.ctx.proposals["p1"].changes == {"news_weight": 0.5} and restored.ctx.proposals["p1"].downgraded


def test_restored_proposals_are_revalidated():
    raw = conversation_to_dict(make())
    raw["proposals"] = [
        {"id": "ok", "ticker": "AAPL", "changes": {"news_weight": 0.5}, "reason": "r", "created": "t"},
        {"id": "bad-ticker", "ticker": "ZZZZ", "changes": {"news_weight": 0.5}},
        {"id": "bad-value", "ticker": "AAPL", "changes": {"news_weight": 99}},
        {"id": "bad-name", "ticker": "AAPL", "changes": {"bogus": 1}},
        "garbage",
    ]
    assert set(conversation_from_dict(raw).ctx.proposals) == {"ok"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("api_messages"),
        lambda r: r.update(api_messages="nope"),
        lambda r: r.update(api_messages=[{"role": "system", "content": "x"}]),
        lambda r: r.update(api_messages=[{"role": "user", "content": [{"nottype": 1}]}]),
        lambda r: r.update(transcript=[{"role": "user"}]),
        lambda r: r.update(api_messages=[{"role": "user", "content": "x"}] * (cv.MAX_MESSAGES + 1)),
    ],
)
def test_malformed_conversations_are_rejected(mutate):
    raw = conversation_to_dict(make())
    mutate(raw)
    assert conversation_from_dict(raw) is None


def test_odd_but_recoverable_fields_are_clamped():
    raw = conversation_to_dict(make())
    raw.update(title="  " + "x" * 500, usage={"input_tokens": -5, "output_tokens": "7"})
    conv = conversation_from_dict(raw)
    assert len(conv.title) == cv.MAX_TITLE
    assert conv.usage.input_tokens == 0 and conv.usage.output_tokens == 7


@pytest.mark.parametrize("raw", [None, "", "not json", "[]", '{"version": 99, "conversations": []}', '{"version": 1}', '{"version": 1, "conversations": "x"}'])
def test_load_all_never_raises_on_garbage(raw):
    assert load_all(raw) == []


def test_load_all_skips_bad_entries_and_duplicate_ids():
    good = conversation_to_dict(make("a"))
    payload = json.dumps({"version": 1, "conversations": [good, {"id": "broken"}, good, conversation_to_dict(make("b"))]})
    assert [c.id for c in load_all(payload)] == ["a", "b"]


def test_dump_all_saves_only_non_empty_newest_first():
    payload, dropped = dump_all([make("old", updated="2026-01-01T00:00:00.000000Z"), make("empty", messages=False), make("new", updated="2026-03-01T00:00:00.000000Z")])
    assert dropped == 0
    assert [c.id for c in load_all(payload)] == ["new", "old"]


def test_dump_all_caps_the_number_of_conversations(monkeypatch):
    monkeypatch.setattr(cv, "MAX_CONVERSATIONS", 3)
    convs = [make(f"c{i}", updated=f"2026-01-0{i + 1}T00:00:00.000000Z") for i in range(5)]
    payload, dropped = dump_all(convs)
    assert dropped == 2
    assert [c.id for c in load_all(payload)] == ["c4", "c3", "c2"]


def test_dump_all_shrinks_from_the_oldest_to_fit_the_quota(monkeypatch):
    convs = [make(f"c{i}", updated=f"2026-01-0{i + 1}T00:00:00.000000Z") for i in range(4)]
    one, _ = dump_all(convs[:1])
    monkeypatch.setattr(cv, "MAX_PAYLOAD_CHARS", int(len(one) * 2.5))
    payload, dropped = dump_all(convs)
    assert dropped == 2 and [c.id for c in load_all(payload)] == ["c3", "c2"]
    monkeypatch.setattr(cv, "MAX_PAYLOAD_CHARS", 10)
    payload, _ = dump_all(convs)
    assert len(load_all(payload)) == 1  # never drops the newest


def test_title_from_collapses_whitespace_and_clips():
    assert title_from("  what   is\nNVDA?  ") == "what is NVDA?"
    assert title_from("") == cv.DEFAULT_TITLE
    clipped = title_from("x" * 200)
    assert len(clipped) == cv.MAX_TITLE and clipped.endswith("…")


# ---- the book ----------------------------------------------------------------------------


def test_new_book_starts_with_one_empty_chat():
    book = ConversationBook()
    assert len(book.items) == 1 and book.current.is_empty and book.persistable() == []


def test_new_chat_reuses_the_existing_empty_one():
    book = ConversationBook()
    first = book.current_id
    book.new()
    assert book.current_id == first and len(book.items) == 1


def test_record_turn_names_the_chat_once_and_bumps_revision():
    book = ConversationBook()
    conv = book.current
    conv.transcript.append({"role": "user", "text": "Is NVDA a Buy?"})
    before = book.revision
    book.record_turn(conv, "Is NVDA a Buy?")
    assert conv.title == "Is NVDA a Buy?" and book.revision == before + 1
    book.record_turn(conv, "a different follow-up")
    assert conv.title == "Is NVDA a Buy?"


def test_new_chat_after_a_used_one_makes_a_second_conversation():
    book = ConversationBook()
    used = book.current
    used.transcript.append({"role": "user", "text": "hi"})
    book.new()
    assert len(book.items) == 2 and book.current_id != used.id
    book.switch(used.id)
    assert book.current_id == used.id
    book.switch("nonexistent")
    assert book.current_id == used.id


def test_rename_ignores_blank_titles():
    book = ConversationBook()
    conv = book.current
    book.rename(conv.id, "   ")
    assert conv.title == cv.DEFAULT_TITLE
    book.rename(conv.id, "  My   title ")
    assert conv.title == "My title"


def test_delete_current_falls_back_to_the_most_recent_other_chat():
    book = ConversationBook()
    a = book.current
    a.transcript.append({"role": "user", "text": "a"})
    book.record_turn(a, "a")
    b = book.new()
    b.transcript.append({"role": "user", "text": "b"})
    book.record_turn(b, "b")
    assert book.current_id == b.id
    rev = book.revision
    book.delete(b.id)
    assert book.current_id == a.id and b.id not in book.items and book.revision == rev + 1


def test_deleting_the_only_chat_leaves_a_fresh_empty_one():
    book = ConversationBook()
    only = book.current
    only.transcript.append({"role": "user", "text": "x"})
    book.delete(only.id)
    assert len(book.items) == 1 and book.current.is_empty and book.current_id != only.id


def test_merge_keeps_the_newer_copy_and_opens_saved_history():
    book = ConversationBook()
    empty_id = book.current_id
    book.merge([make("a", title="stored a", updated="2026-02-01T00:00:00.000000Z"), make("b", updated="2026-01-01T00:00:00.000000Z")])
    assert book.current_id == "a" and empty_id in book.items  # lands on the most recent saved chat
    assert book.revision == 0  # nothing new to write back

    book.items["b"].transcript.append({"role": "user", "text": "edited locally"})
    book.items["b"].updated = "2026-09-01T00:00:00.000000Z"
    book.merge([make("b", title="older stored b", updated="2026-01-01T00:00:00.000000Z")])
    assert book.items["b"].transcript[-1]["text"] == "edited locally"


def test_merge_flags_a_write_when_the_session_already_had_unsaved_chats():
    book = ConversationBook()
    book.current.transcript.append({"role": "user", "text": "typed before the load arrived"})
    book.merge([make("a")])
    assert book.revision == 1
    assert len(book.persistable()) == 2


def test_session_store_persists_nothing():
    store = cv.SessionStore()
    assert store.status == "ready" and store.sync([make()], 1) is None


def test_delete_prefers_a_saved_chat_over_a_blank_one():
    book = ConversationBook()
    saved = book.current
    saved.transcript.append({"role": "user", "text": "keep me"})
    book.record_turn(saved, "keep me")
    doomed = book.new()
    doomed.transcript.append({"role": "user", "text": "bye"})
    book.record_turn(doomed, "bye")
    blank = book.new()  # the newest item, but empty
    book.switch(doomed.id)
    book.delete(doomed.id)
    assert book.current_id == saved.id and blank.id in book.items
