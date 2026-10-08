"""Assistant page: a Claude-powered chat about the tracked companies that can
also tune each company's recommendation math (see assistant/tools.py for the
tools and the safety rules enforced in code, assistant/chat.py for the loop).

Anyone can open the page and read the owner's PUBLISHED example conversations
(assistant/showcase.py) -- read-only. Sending a message, the private
conversation list, the AI adjustments panel and publishing are owner-only
(dashboard/auth.py::is_owner, which fails closed): every message spends the
owner's Anthropic API credits and the AI can change shared state. A visitor who
tries to send is asked for the owner password in a dialog.

Several conversations can be kept and switched between (sidebar). They live in
st.session_state and are saved to a ConversationStore -- by default the owner's
own browser localStorage, never the server (see dashboard/browser_store.py and
assistant/conversations.py, which also describes where a hosted database would
plug in).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import streamlit as st

from stock_predictor import config
from stock_predictor.assistant import chat, tools
from stock_predictor.assistant import showcase
from stock_predictor.assistant.conversations import ConversationBook
from stock_predictor.dashboard.auth import is_owner, unlock_dialog
from stock_predictor.dashboard.browser_store import get_store
from stock_predictor.dashboard.symbol_browser import BROWSE_CONTEXT_KEY
from stock_predictor.model import overrides_store, overrides_sync
from stock_predictor.model.tuning import PARAM_SPECS

_BOOK = "assistant_book"
_SYNC = "assistant_sync_status"
_UNSYNCED = "assistant_unsynced_entries"  # override changes not yet committed to git (session-wide)
_SHOWCASE_VIEW = "assistant_showcase_view"  # id of the published example being read, if any
_PUBLISH_STATUS = "assistant_publish_status"
_PENDING = "assistant_pending_prompt"  # what a visitor typed before being asked for the password
_RUN_PENDING = "assistant_run_pending"  # set by a successful unlock: send that prompt now

_IMAGE_RE = re.compile(r"!\[")
_EXAMPLES = (
    "Is NVDA a Buy right now, and why?",
    "Compare AAPL's valuation to its competitors.",
    "Social sentiment on TSLA looks noisy -- should I weight it less?",
)


def safe_markdown(text: str) -> str:
    """Neutralize what could misrender or leak in model output. "![" would
    make the browser fetch an arbitrary image URL (a way to smuggle data out
    in a query string) so it becomes a plain link; "$" is escaped because
    st.markdown treats it as a LaTeX delimiter and prices are everywhere.
    """
    return _IMAGE_RE.sub("[", text).replace("$", "\\$")


def safe_stream(chunks: Iterator[str]) -> Iterator[str]:
    """safe_markdown over a stream of text deltas. A trailing "!" is held
    back until the next chunk so "!" + "[" split across two chunks is still
    caught."""
    carry = ""
    for chunk in chunks:
        text = carry + chunk
        carry = "!" if text.endswith("!") else ""
        if carry:
            text = text[:-1]
        if text:
            yield safe_markdown(text)
    if carry:
        yield carry


def _book() -> ConversationBook:
    if _BOOK not in st.session_state:
        st.session_state[_BOOK] = ConversationBook()
    return st.session_state[_BOOK]


def _sync(ctx: tools.ToolContext) -> None:
    """Commit override changes to git (one commit for the whole batch). Entries
    are kept session-wide on failure, so a retry works even after switching to
    another conversation."""
    pending = st.session_state.setdefault(_UNSYNCED, [])
    pending.extend(ctx.applied)
    ctx.applied.clear()
    if not pending:
        return
    if not overrides_sync.is_configured():
        st.session_state[_SYNC] = ("local", "Applied locally only. Set GITHUB_TOKEN and GITHUB_REPO to also save changes to git.")
        pending.clear()
        return
    result = overrides_sync.commit_entries(pending)
    if result.ok:
        pending.clear()
        st.session_state[_SYNC] = ("ok", result.message)
    else:
        st.session_state[_SYNC] = ("error", result.message)


def _render_sync_status() -> None:
    status = st.session_state.get(_SYNC)
    if status is None:
        return
    kind, message = status
    if kind == "ok":
        st.caption(f"{message} It can take about a minute for the deployed app to restart with the change.")
    elif kind == "local":
        st.caption(message)
    else:
        st.warning(message)
        if st.session_state.get(_UNSYNCED) and st.button("Retry saving to git", key="assistant_retry_sync"):
            _sync(tools.ToolContext())
            st.rerun()


def _render_transcript(transcript: list[dict]) -> None:
    for message in transcript:
        with st.chat_message(message["role"]):
            st.markdown(safe_markdown(message["text"]))


def _render_showcase_sidebar(chats: list[dict], viewing: dict | None) -> None:
    """The published example conversations -- visible to everyone."""
    st.markdown("### Example conversations")
    if not chats:
        st.caption("None published yet.")
    for chat_item in chats:
        is_open = viewing is not None and chat_item["id"] == viewing["id"]
        if st.button(
            chat_item["title"],
            key=f"showcase_{chat_item['id']}",
            type="primary" if is_open else "secondary",
            use_container_width=True,
        ):
            st.session_state[_SHOWCASE_VIEW] = chat_item["id"]
            st.rerun()


def _render_publish_controls(conv, published_ids: set[str]) -> None:
    """Inside the owner's rename/delete popover: make this chat a public example."""
    st.divider()
    if conv.id in published_ids:
        st.caption("This chat is published as a public example (publish again after more messages to update it).")
        if st.button("Unpublish", key=f"unpublish_{conv.id}"):
            st.session_state[_PUBLISH_STATUS] = showcase.unpublish(conv.id)
            st.rerun()
        if st.button("Publish again", key=f"republish_{conv.id}", disabled=conv.is_empty):
            chat_item = showcase.chat_from_conversation(conv.id, conv.title, conv.transcript)
            if chat_item is not None:
                st.session_state[_PUBLISH_STATUS] = showcase.publish(chat_item)
                st.rerun()
        return
    confirmed = st.checkbox(
        "I understand anyone who visits the site can read this chat, including my questions.",
        key=f"publish_ok_{conv.id}",
        disabled=conv.is_empty,
    )
    if st.button("Publish as public example", key=f"publish_{conv.id}", disabled=conv.is_empty or not confirmed):
        chat_item = showcase.chat_from_conversation(conv.id, conv.title, conv.transcript)
        if chat_item is not None:
            st.session_state[_PUBLISH_STATUS] = showcase.publish(chat_item)
            st.rerun()


def _render_owner_sidebar(book: ConversationBook, store, chats: list[dict], viewing: dict | None) -> None:
    with st.sidebar:
        st.markdown("### Conversations")
        if st.button("New chat", key="conv_new", icon=":material/add:", use_container_width=True):
            st.session_state.pop(_SHOWCASE_VIEW, None)
            book.new()
            st.rerun()
        for conv in book.ordered():
            is_current = conv.id == book.current_id and viewing is None
            if conv.is_empty and conv.id != book.current_id:
                continue  # an unused blank chat is just clutter once you've moved on
            if st.button(
                conv.title,
                key=f"conv_{conv.id}",
                type="primary" if is_current else "secondary",
                use_container_width=True,
            ):
                st.session_state.pop(_SHOWCASE_VIEW, None)
                book.switch(conv.id)
                st.rerun()

        conv = book.current
        with st.popover("Rename, delete or publish this chat", use_container_width=True):
            title = st.text_input("Title", value=conv.title, max_chars=60, key=f"rename_{conv.id}")
            if st.button("Save title", key=f"rename_save_{conv.id}"):
                book.rename(conv.id, title)
                st.rerun()
            if st.button("Delete this chat", key=f"delete_{conv.id}", type="primary"):
                book.delete(conv.id)
                st.rerun()
            _render_publish_controls(conv, {c["id"] for c in chats})

        status = st.session_state.get(_PUBLISH_STATUS)
        if status is not None:
            (st.caption if status.ok else st.warning)(status.message)

        if store.status == "loading":
            st.caption("Loading your saved chats...")
        elif store.status != "ready":
            st.warning(store.status)
        elif getattr(store, "dropped", 0):
            st.caption(f"{store.dropped} oldest chat(s) weren't saved to keep browser storage under its limit.")
        else:
            st.caption("Your chats are saved in this browser only.")

        st.divider()
        _render_showcase_sidebar(chats, viewing)


def _fmt(value: float) -> str:
    return f"{value:g}"


def _render_proposals(ctx: tools.ToolContext, book: ConversationBook) -> None:
    for proposal in list(ctx.proposals.values()):
        with st.container(border=True):
            st.markdown(f"**Proposed change for {proposal.ticker}**")
            for row in tools.proposal_diff(proposal):
                st.markdown(
                    f"- `{row['param']}`: {_fmt(row['current'])} → **{_fmt(row['proposed'])}** "
                    f"(default {_fmt(row['default'])})"
                )
            if proposal.downgraded:
                st.caption(
                    "The assistant tried to apply this directly, but the conversation includes third-party text "
                    "(news, posts or profiles), so it needs your approval."
                )
            if proposal.reason:
                st.caption(f"Assistant's reasoning (unverified): {safe_markdown(proposal.reason)}")
            apply_col, dismiss_col, _ = st.columns([1, 1, 4])
            if apply_col.button("Apply", key=f"apply_{proposal.id}", type="primary"):
                try:
                    tools.apply_proposal(ctx, proposal.id)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    book.mark_dirty()
                    _sync(ctx)
                    st.rerun()
            if dismiss_col.button("Dismiss", key=f"dismiss_{proposal.id}"):
                tools.dismiss_proposal(ctx, proposal.id)
                book.mark_dirty()
                st.rerun()


def _render_adjustments_panel(ctx: tools.ToolContext) -> None:
    active = overrides_store.all_overrides()
    recent = overrides_store.history()[-10:][::-1]
    label = f"AI adjustments ({len(active)} compan{'y' if len(active) == 1 else 'ies'} tuned)"
    with st.expander(label, expanded=False):
        if not active:
            st.caption("No company is using custom math. Every company uses the defaults.")
        for ticker, params in active.items():
            text_col, button_col = st.columns([5, 1])
            text_col.markdown(
                f"**{ticker}**: "
                + ", ".join(f"`{name}` = {_fmt(value)} (default {_fmt(PARAM_SPECS[name].default)})" for name, value in params.items())
            )
            if button_col.button("Reset", key=f"reset_{ticker}"):
                entries = overrides_store.reset(ticker, None, "Reset by owner", "owner")
                ctx.applied.extend(e.as_dict() for e in entries)
                _sync(ctx)
                st.rerun()
        if recent:
            st.markdown("**Recent changes**")
            for entry in recent:
                text_col, button_col = st.columns([5, 1])
                text_col.caption(
                    f"{entry['ts']} · {entry['ticker']} · `{entry['param']}` {_fmt(entry['old'])} → {_fmt(entry['new'])} "
                    f"· {entry.get('source', '?')}"
                    + (f" · {safe_markdown(str(entry['reason']))}" if entry.get("reason") else "")
                )
                if entry.get("source") != "undo" and button_col.button("Undo", key=f"undo_{entry['id']}"):
                    try:
                        entries = overrides_store.undo(entry["id"])
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        ctx.applied.extend(e.as_dict() for e in entries)
                        _sync(ctx)
                        st.rerun()


def _arm_pending_prompt() -> None:
    st.session_state[_RUN_PENDING] = True


def _render_visitor(chats: list[dict], viewing: dict | None) -> None:
    with st.sidebar:
        _render_showcase_sidebar(chats, viewing)

    st.caption(
        "A Claude-powered assistant for the companies this app tracks. Below are example conversations; "
        "sending a new message needs the owner's password. Not financial advice."
    )
    if viewing is not None:
        st.markdown(f"**Example: {safe_markdown(viewing['title'])}**")
        _render_transcript(viewing["transcript"])
    elif not chats:
        st.info("No example conversations have been published yet.")

    prompt = st.chat_input("Enter the owner password to send a message...")
    if prompt:
        st.session_state[_PENDING] = prompt
        unlock_dialog(_arm_pending_prompt, "Sending a message uses the owner's API credits, so it needs the owner password.")


def _render_owner(chats: list[dict], viewing: dict | None) -> None:
    if not config.anthropic_api_key():
        st.warning(
            "No ANTHROPIC_API_KEY is configured. Add one (in .env locally, or in the app's secrets on "
            "Streamlit Cloud) to enable the chat."
        )
        st.stop()

    book = _book()
    store = get_store()
    # Mounted once per run. Returns the previously saved chats exactly once, on
    # the run they arrive from the browser (never before the owner has unlocked).
    loaded = store.sync(book.persistable(), book.revision)
    if loaded is not None:
        book.merge(loaded)
        st.rerun()

    _render_owner_sidebar(book, store, chats, viewing)

    pending = None
    if st.session_state.pop(_RUN_PENDING, False):
        pending = st.session_state.pop(_PENDING, None)
    else:
        st.session_state.pop(_PENDING, None)

    conv = book.current
    ctx, usage = conv.ctx, conv.usage
    # What was last on screen on the Companies/Watchlist pages (None if never opened).
    ctx.screen_context = st.session_state.get(BROWSE_CONTEXT_KEY)

    st.caption(
        "Ask about any tracked company, or tell me how to tune its recommendation math. I use the app's own data "
        f"(refreshed daily), not live quotes. Model: {config.assistant_model()}. Not financial advice."
    )
    _render_adjustments_panel(ctx)

    if viewing is not None:
        st.markdown(f"**Public example: {safe_markdown(viewing['title'])}** (read-only; sending starts a new chat)")
        _render_transcript(viewing["transcript"])
    else:
        st.caption(f"This chat: {usage.budget_used:,} of {chat.SESSION_TOKEN_BUDGET:,} budgeted tokens.")
        if not conv.transcript:
            st.markdown("**Try asking:**")
            for example in _EXAMPLES:
                st.markdown(f"- {example}")
        _render_transcript(conv.transcript)
        _render_proposals(ctx, book)
        _render_sync_status()

    exhausted = viewing is None and chat.over_budget(usage)
    if exhausted:
        st.info("This chat reached its token budget. Start a new chat from the sidebar.")
    prompt = st.chat_input("Ask a question or describe a change...", disabled=exhausted) or pending
    if not prompt:
        return

    if viewing is not None:  # sending from a read-only example opens a fresh chat
        st.session_state.pop(_SHOWCASE_VIEW, None)
        conv = book.new()
        ctx, usage = conv.ctx, conv.usage
        ctx.screen_context = st.session_state.get(BROWSE_CONTEXT_KEY)

    conv.transcript.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(safe_markdown(prompt))

    try:
        client = chat.make_client()
    except Exception as exc:
        conv.transcript.append({"role": "assistant", "text": f"Couldn't start the assistant: {type(exc).__name__}."})
        book.record_turn(conv, prompt)
        st.rerun()

    runner = chat.TurnRunner(client, conv.api_messages, prompt, ctx, usage)
    raw_parts: list[str] = []

    def recorded() -> Iterator[str]:
        for chunk in runner.stream():
            raw_parts.append(chunk)
            yield chunk

    with st.chat_message("assistant"):
        st.write_stream(safe_stream(recorded()))
    # The RAW text is stored; safe_markdown is applied each time it's displayed.
    conv.transcript.append({"role": "assistant", "text": "".join(raw_parts)})
    if runner.ok:
        conv.api_messages = runner.messages
    book.record_turn(conv, prompt)
    _sync(ctx)
    st.rerun()


def render() -> None:
    st.title("Assistant")
    owner = is_owner()
    chats = showcase.load_chats()

    viewing_id = st.session_state.get(_SHOWCASE_VIEW)
    viewing = next((c for c in chats if c["id"] == viewing_id), None)
    if viewing is None and not owner and chats:
        viewing = chats[0]  # visitors land on the newest published example

    if owner:
        _render_owner(chats, viewing)
    else:
        _render_visitor(chats, viewing)
