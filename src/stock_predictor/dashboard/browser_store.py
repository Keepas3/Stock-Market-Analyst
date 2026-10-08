"""Saves Assistant conversations in the owner's own browser (localStorage) via a
tiny inline Streamlit v2 component -- nothing is stored on any server. A
ConversationStore implementation (see assistant/conversations.py); swapping in a
hosted database later means writing another class with the same `sync` method.

How it works, and why it's built this way:
  * On mount the JS sends whatever is in localStorage to Python (state value
    "stored"). That arrives one rerun after the page first renders.
  * Nothing is written until that load has arrived, so an empty fresh session
    can never overwrite saved chats.
  * To save, Python passes the JSON `payload` plus an increasing `seq`; the JS
    writes it and acknowledges with state value "saved_seq". Python keeps
    passing the payload until the ack arrives, so a rerun racing the write
    can't lose it.
  * The component is mounted by the Assistant page only after the owner
    password gate, so stored chats are never sent to the server before unlock.
  * Two tabs open at once share one localStorage entry: the last write wins.
"""

from __future__ import annotations

import streamlit as st

from stock_predictor.assistant.conversations import STORAGE_KEY, Conversation, dump_all, load_all

_STATE_KEY = "_assistant_browser_store"

_JS = """
export default function (component) {
    const { data, setStateValue, parentElement } = component;
    try {
        if (!parentElement.__loaded) {
            parentElement.__loaded = true;
            setStateValue("stored", { raw: window.localStorage.getItem(data.storage_key) });
        }
        if (typeof data.payload === "string" && data.seq > (parentElement.__savedSeq || 0)) {
            window.localStorage.setItem(data.storage_key, data.payload);
            parentElement.__savedSeq = data.seq;
            setStateValue("saved_seq", data.seq);
        }
    } catch (err) {
        setStateValue("error", String((err && err.message) || err));
    }
}
"""

_component = st.components.v2.component("assistant_browser_store", js=_JS)


class BrowserStore:
    def __init__(self) -> None:
        self._state = st.session_state.setdefault(
            _STATE_KEY, {"hydrated": False, "last_revision": None, "payload": None, "seq": 0, "dropped": 0}
        )
        self._error: str | None = None

    @property
    def status(self) -> str:
        if self._error:
            return f"Your browser won't let this page save chats ({self._error}), so they'll be lost on refresh."
        return "ready" if self._state["hydrated"] else "loading"

    @property
    def dropped(self) -> int:
        """Conversations left out of the last save to stay under the browser's storage quota."""
        return self._state["dropped"]

    def sync(self, conversations: list[Conversation], revision: int) -> list[Conversation] | None:
        state = self._state
        if state["hydrated"] and revision != state["last_revision"]:
            payload, dropped = dump_all(conversations)
            state["payload"] = payload
            state["seq"] += 1
            state["dropped"] = dropped
            state["last_revision"] = revision

        result = _component(
            key="assistant_browser_store",
            data={"storage_key": STORAGE_KEY, "payload": state["payload"], "seq": state["seq"]},
            default={"stored": None, "saved_seq": 0, "error": None},
            on_stored_change=lambda: None,
            on_saved_seq_change=lambda: None,
            on_error_change=lambda: None,
        )
        self._error = result.error
        if state["payload"] is not None and (result.saved_seq or 0) >= state["seq"]:
            state["payload"] = None  # acknowledged: stop re-sending it

        if not state["hydrated"] and result.stored is not None:
            state["hydrated"] = True
            state["last_revision"] = revision  # the store already holds what was just loaded
            return load_all(result.stored.get("raw"))
        return None


def get_store():
    """The configured store (config.assistant_chat_store): "browser" (default),
    or "session" for no persistence."""
    from stock_predictor import config
    from stock_predictor.assistant.conversations import SessionStore

    return SessionStore() if config.assistant_chat_store() == "session" else BrowserStore()
