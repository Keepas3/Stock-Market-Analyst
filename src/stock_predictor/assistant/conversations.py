"""Multiple saved Assistant conversations: the data model, JSON (de)serialization,
and the `ConversationStore` interface persistence plugs into. Streamlit-free so
it's testable; the page holds one `ConversationBook` in st.session_state.

Where conversations are kept is behind `ConversationStore.sync`:
  * SessionStore  -- nothing persisted (a conversation lives as long as the session).
  * BrowserStore  -- the owner's own browser localStorage (dashboard/browser_store.py).
  * A future hosted database (Supabase, Turso, ...) is a third implementation of
    the same single method: load synchronously on first call, upsert whenever
    `revision` changes, return the loaded conversations once. Nothing else in the
    page needs to change. Until then the repo is public and the server disk is
    wiped on each redeploy, so neither git nor local files are used for chats.

Everything read back from a store is untrusted input (it round-trips through the
browser), so `load_all` validates structure and clamps values instead of trusting it.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Protocol

from stock_predictor.assistant.chat import Usage
from stock_predictor.assistant.tools import Proposal, ToolContext
from stock_predictor.config import load_watchlist
from stock_predictor.model.tuning import coerce_value

STORAGE_VERSION = 1
STORAGE_KEY = f"stock_assistant_conversations_v{STORAGE_VERSION}"  # the browser localStorage entry
DEFAULT_TITLE = "New chat"
MAX_CONVERSATIONS = 30
MAX_MESSAGES = 400
MAX_TITLE = 60
# localStorage allows roughly 5M characters per origin; stay well under it.
MAX_PAYLOAD_CHARS = 3_000_000


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = DEFAULT_TITLE
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    api_messages: list = field(default_factory=list)  # exactly what is sent to the API
    transcript: list[dict] = field(default_factory=list)  # {"role", "text"} for display
    usage: Usage = field(default_factory=Usage)
    ctx: ToolContext = field(default_factory=ToolContext)

    @property
    def is_empty(self) -> bool:
        return not self.transcript and not self.api_messages


def title_from(prompt: str) -> str:
    text = " ".join(prompt.split())
    if not text:
        return DEFAULT_TITLE
    return text if len(text) <= MAX_TITLE else text[: MAX_TITLE - 1].rstrip() + "…"


# ---- serialization -----------------------------------------------------------------------


def block_to_dict(block: object) -> dict:
    """An SDK response block (or an already-plain dict) as the dict the API
    accepts back as input. Thinking blocks must round-trip unchanged, signature
    included; None-valued fields are dropped."""
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    return {k: v for k, v in vars(block).items() if v is not None}


def _message_to_dict(message: dict) -> dict:
    content = message["content"]
    if isinstance(content, list):
        content = [block_to_dict(b) for b in content]
    return {"role": message["role"], "content": content}


def conversation_to_dict(conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "title": conv.title,
        "created": conv.created,
        "updated": conv.updated,
        "api_messages": [_message_to_dict(m) for m in conv.api_messages],
        "transcript": conv.transcript,
        "usage": asdict(conv.usage),
        "tainted": conv.ctx.tainted,
        "proposals": [asdict(p) for p in conv.ctx.proposals.values()],
    }


def _valid_message(message: object) -> bool:
    if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    return isinstance(content, list) and all(isinstance(b, dict) and isinstance(b.get("type"), str) for b in content)


def _restore_proposals(raw: object) -> dict[str, Proposal]:
    proposals: dict[str, Proposal] = {}
    watchlist = load_watchlist()
    for item in raw if isinstance(raw, list) else []:
        try:
            ticker = item["ticker"]
            if ticker not in watchlist:
                continue
            changes = {name: coerce_value(name, value) for name, value in item["changes"].items()}
            proposal = Proposal(
                id=str(item["id"])[:16],
                ticker=ticker,
                changes=changes,
                reason=str(item.get("reason", ""))[:300],
                created=str(item.get("created", "")),
                downgraded=bool(item.get("downgraded", False)),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if changes:
            proposals[proposal.id] = proposal
    return proposals


def conversation_from_dict(raw: object) -> Conversation | None:
    """None if `raw` isn't a well-formed conversation."""
    if not isinstance(raw, dict):
        return None
    try:
        messages = raw["api_messages"]
        transcript = raw["transcript"]
        if not isinstance(messages, list) or not isinstance(transcript, list) or len(messages) > MAX_MESSAGES:
            return None
        if not all(_valid_message(m) for m in messages):
            return None
        if not all(
            isinstance(t, dict) and t.get("role") in ("user", "assistant") and isinstance(t.get("text"), str)
            for t in transcript
        ):
            return None
        usage = raw.get("usage") or {}
        conv = Conversation(
            id=str(raw["id"])[:64],
            title=" ".join(str(raw.get("title") or DEFAULT_TITLE).split())[:MAX_TITLE] or DEFAULT_TITLE,
            created=str(raw.get("created", _now())),
            updated=str(raw.get("updated", _now())),
            api_messages=messages,
            transcript=[{"role": t["role"], "text": t["text"]} for t in transcript],
            usage=Usage(**{k: max(0, int(usage.get(k, 0))) for k in ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens")}),
        )
    except (KeyError, TypeError, ValueError):
        return None
    conv.ctx.tainted = bool(raw.get("tainted", False))
    conv.ctx.proposals = _restore_proposals(raw.get("proposals"))
    return conv


def dump_all(conversations: list[Conversation]) -> tuple[str, int]:
    """(JSON payload, number of conversations dropped to fit). Only non-empty
    conversations are saved, newest first, at most MAX_CONVERSATIONS, shrinking
    from the oldest end until the payload fits the browser's storage quota."""
    kept = sorted((c for c in conversations if not c.is_empty), key=lambda c: c.updated, reverse=True)
    dropped = max(0, len(kept) - MAX_CONVERSATIONS)
    kept = kept[:MAX_CONVERSATIONS]
    while True:
        payload = json.dumps(
            {"version": STORAGE_VERSION, "conversations": [conversation_to_dict(c) for c in kept]},
            separators=(",", ":"),
        )
        if len(payload) <= MAX_PAYLOAD_CHARS or len(kept) <= 1:
            return payload, dropped
        kept.pop()
        dropped += 1


def load_all(raw: str | None) -> list[Conversation]:
    """Tolerant: anything unparseable or malformed is skipped, never raised."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, dict) or data.get("version") != STORAGE_VERSION:
        return []
    items = data.get("conversations")
    if not isinstance(items, list):
        return []
    loaded = [c for c in (conversation_from_dict(item) for item in items[:MAX_CONVERSATIONS]) if c is not None]
    seen: set[str] = set()
    return [c for c in loaded if not (c.id in seen or seen.add(c.id))]


# ---- the set of conversations the page works with ----------------------------------------------


class ConversationBook:
    """All conversations in a session plus which one is open. `revision` goes up
    on every change worth persisting, so a store knows when to write."""

    def __init__(self) -> None:
        self.items: dict[str, Conversation] = {}
        self.current_id: str = ""
        self.revision = 0
        self.new()

    @property
    def current(self) -> Conversation:
        return self.items[self.current_id]

    def ordered(self) -> list[Conversation]:
        return sorted(self.items.values(), key=lambda c: c.updated, reverse=True)

    def persistable(self) -> list[Conversation]:
        return [c for c in self.items.values() if not c.is_empty]

    def new(self) -> Conversation:
        """Open a fresh chat -- or the existing empty one, so empties don't pile up."""
        for conv in self.items.values():
            if conv.is_empty:
                self.current_id = conv.id
                return conv
        conv = Conversation()
        self.items[conv.id] = conv
        self.current_id = conv.id
        return conv

    def switch(self, conv_id: str) -> None:
        if conv_id in self.items:
            self.current_id = conv_id

    def rename(self, conv_id: str, title: str) -> None:
        conv = self.items.get(conv_id)
        clean = " ".join(title.split())[:MAX_TITLE]
        if conv is not None and clean:
            conv.title = clean
            self.revision += 1

    def delete(self, conv_id: str) -> None:
        if conv_id not in self.items:
            return
        was_saved = not self.items[conv_id].is_empty
        del self.items[conv_id]
        if was_saved:
            self.revision += 1
        if self.current_id == conv_id or self.current_id not in self.items:
            saved = self.persistable()
            if saved:
                self.current_id = max(saved, key=lambda c: c.updated).id
            else:
                self.new()

    def mark_dirty(self) -> None:
        """Something persisted changed outside the methods above (e.g. a pending
        proposal was applied or dismissed)."""
        self.revision += 1

    def record_turn(self, conv: Conversation, prompt: str) -> None:
        """Call after a turn finishes: names a new chat from its first message,
        bumps `updated`, and marks the book dirty."""
        if conv.title == DEFAULT_TITLE:
            conv.title = title_from(prompt)
        conv.updated = _now()
        self.revision += 1

    def merge(self, loaded: list[Conversation]) -> None:
        """Fold stored conversations in, keeping whichever copy is newer. If this
        session already had saved-worthy chats the store doesn't, mark dirty so
        the next sync writes them."""
        had_unsaved = bool(self.persistable())
        for incoming in loaded:
            existing = self.items.get(incoming.id)
            if existing is None or incoming.updated > existing.updated:
                self.items[incoming.id] = incoming
        current = self.items.get(self.current_id)
        saved = self.persistable()
        if current is not None and current.is_empty and saved:
            # The auto-created empty chat shouldn't be what the owner lands on
            # if they have history: open the most recent saved one instead.
            self.current_id = max(saved, key=lambda c: c.updated).id
        if had_unsaved:
            self.revision += 1


class ConversationStore(Protocol):
    """Where conversations persist. One method, so a store that loads
    asynchronously (the browser) and one that loads synchronously (a database)
    look the same to the page."""

    def sync(self, conversations: list[Conversation], revision: int) -> list[Conversation] | None:
        """Call on every page run. Persists `conversations` whenever `revision`
        has changed since the last call. Returns the previously stored
        conversations exactly once -- on the run they become available -- and
        None on every other run."""
        ...

    @property
    def status(self) -> str:
        """"loading", "ready", or an error message for the page to show."""
        ...


class SessionStore:
    """No persistence: nothing is stored, so there's nothing to load."""

    status = "ready"

    def sync(self, conversations: list[Conversation], revision: int) -> list[Conversation] | None:
        return None
