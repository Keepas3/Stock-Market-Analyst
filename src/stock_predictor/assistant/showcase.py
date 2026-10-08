"""Published "example" Assistant conversations: chats the owner chose to make
public so portfolio visitors can read what the assistant does without being able
to use it (sending a message needs the owner password).

Stored in a git-tracked file, config/showcase_chats.json, committed through the
GitHub API the same way config/model_overrides.yaml is -- so they survive
redeploys, and are PUBLIC (the repo is public). Only each chat's visible text is
kept: the user's messages and the assistant's replies as displayed. Raw tool
results, thinking blocks, proposals and the API history never leave the owner's
browser. Streamlit-free so it's testable.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import tempfile
import threading
from dataclasses import dataclass

from stock_predictor import config
from stock_predictor.model import overrides_sync

SHOWCASE_FILENAME = "showcase_chats.json"
FILE_PATH = f"config/{SHOWCASE_FILENAME}"
VERSION = 1
MAX_CHATS = 10
MAX_MESSAGES = 80
MAX_TEXT = 20_000
MAX_TITLE = 60

_lock = threading.Lock()
_cache: dict[tuple[str, int, int], dict] = {}


def showcase_path():
    # Read at call time (not import time) so tests can monkeypatch CONFIG_DIR.
    return config.CONFIG_DIR / SHOWCASE_FILENAME


def _empty() -> dict:
    return {"version": VERSION, "chats": []}


def _clean_chat(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    transcript = raw.get("transcript")
    if not isinstance(raw.get("id"), str) or not isinstance(transcript, list):
        return None
    messages = [
        {"role": m["role"], "text": m["text"][:MAX_TEXT]}
        for m in transcript
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("text"), str)
    ][:MAX_MESSAGES]
    if not messages:
        return None
    title = " ".join(str(raw.get("title") or "").split())[:MAX_TITLE] or "Example chat"
    return {"id": raw["id"][:64], "title": title, "published": str(raw.get("published", "")), "transcript": messages}


def parse(text: str) -> dict:
    """Tolerant: a missing, corrupted or hand-edited file degrades to "no
    published chats" instead of breaking the page."""
    try:
        raw = json.loads(text) if text.strip() else None
    except ValueError:
        raw = None
    store = _empty()
    if not isinstance(raw, dict) or raw.get("version") != VERSION or not isinstance(raw.get("chats"), list):
        return store
    seen: set[str] = set()
    for item in raw["chats"]:
        chat = _clean_chat(item)
        if chat is not None and chat["id"] not in seen:
            seen.add(chat["id"])
            store["chats"].append(chat)
    store["chats"] = store["chats"][:MAX_CHATS]
    return store


def dump(store: dict) -> str:
    return json.dumps(store, indent=2, ensure_ascii=False) + "\n"


def load_chats() -> list[dict]:
    """Published chats, newest first. Cached by file mtime."""
    path = showcase_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return []
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _cache.get(key)
    if cached is None:
        cached = parse(path.read_text(encoding="utf-8"))
        _cache.clear()
        _cache[key] = cached
    return sorted(copy.deepcopy(cached["chats"]), key=lambda c: c["published"], reverse=True)


def apply_publish(store: dict, chat: dict) -> dict:
    """Add `chat` (replacing an earlier publish of the same id). If that goes
    over MAX_CHATS the oldest published chats are dropped."""
    chats = [c for c in store["chats"] if c["id"] != chat["id"]]
    chats.append(chat)
    chats.sort(key=lambda c: c["published"], reverse=True)
    store["chats"] = chats[:MAX_CHATS]
    return store


def apply_unpublish(store: dict, chat_id: str) -> dict:
    store["chats"] = [c for c in store["chats"] if c["id"] != chat_id]
    return store


def chat_from_conversation(conv_id: str, title: str, transcript: list[dict]) -> dict | None:
    """The public form of a conversation, or None if it has nothing to show."""
    return _clean_chat(
        {
            "id": conv_id,
            "title": title,
            "published": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "transcript": transcript,
        }
    )


def _write_local(store: dict) -> None:
    path = showcase_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".showcase.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(dump(store))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@dataclass
class PublishResult:
    ok: bool  # saved to git (durable)
    message: str


def _change(operate, commit_message: str) -> PublishResult:
    """Apply `operate(store)` to the local file (instant effect for visitors on
    this server) and to the file on GitHub (so it survives a redeploy)."""
    with _lock:
        path = showcase_path()
        current = parse(path.read_text(encoding="utf-8")) if path.exists() else _empty()
        _write_local(operate(current))
    if not overrides_sync.is_configured():
        return PublishResult(False, "Changed locally only: GITHUB_TOKEN and GITHUB_REPO aren't configured, so it will reset on the next redeploy.")
    result = overrides_sync.commit_file(FILE_PATH, lambda remote: dump(operate(parse(remote))), commit_message)
    return PublishResult(result.ok, result.message)


def publish(chat: dict) -> PublishResult:
    return _change(lambda store: apply_publish(store, chat), "Publish an example Assistant chat")


def unpublish(chat_id: str) -> PublishResult:
    return _change(lambda store: apply_unpublish(store, chat_id), "Unpublish an example Assistant chat")
