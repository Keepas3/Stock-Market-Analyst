"""Assistant page: a Claude-powered chat about the tracked companies that can
also tune each company's recommendation math (see assistant/tools.py for the
tools and the safety rules enforced in code, assistant/chat.py for the loop).

The whole page is owner-only (dashboard/auth.py::require_owner_page) because
every message spends the owner's Anthropic API credits and the AI can change
shared state. Conversation state lives in st.session_state only -- it is
per-browser-session and is lost on a redeploy, by design (no chat history is
stored server-side).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import streamlit as st

from stock_predictor import config
from stock_predictor.assistant import chat, tools
from stock_predictor.dashboard.auth import require_owner_page
from stock_predictor.dashboard.symbol_browser import BROWSE_CONTEXT_KEY
from stock_predictor.model import overrides_store, overrides_sync
from stock_predictor.model.tuning import PARAM_SPECS

_API = "assistant_api_messages"
_TRANSCRIPT = "assistant_transcript"
_CTX = "assistant_ctx"
_USAGE = "assistant_usage"
_SYNC = "assistant_sync_status"

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


def _state() -> tuple[list, list, tools.ToolContext, chat.Usage]:
    if _API not in st.session_state:
        st.session_state[_API] = []
        st.session_state[_TRANSCRIPT] = []
        st.session_state[_CTX] = tools.ToolContext()
        st.session_state[_USAGE] = chat.Usage()
    return st.session_state[_API], st.session_state[_TRANSCRIPT], st.session_state[_CTX], st.session_state[_USAGE]


def _clear() -> None:
    for key in (_API, _TRANSCRIPT, _CTX, _USAGE, _SYNC):
        st.session_state.pop(key, None)


def _sync(ctx: tools.ToolContext) -> None:
    """Commit this conversation's applied changes to git (one commit for the
    whole batch). Entries are kept on failure so the owner can retry."""
    if not ctx.applied:
        return
    if not overrides_sync.is_configured():
        st.session_state[_SYNC] = ("local", "Applied locally only. Set GITHUB_TOKEN and GITHUB_REPO to also save changes to git.")
        ctx.applied.clear()
        return
    result = overrides_sync.commit_entries(ctx.applied)
    if result.ok:
        ctx.applied.clear()
        st.session_state[_SYNC] = ("ok", result.message)
    else:
        st.session_state[_SYNC] = ("error", result.message)


def _render_sync_status(ctx: tools.ToolContext) -> None:
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
        if ctx.applied and st.button("Retry saving to git", key="assistant_retry_sync"):
            _sync(ctx)
            st.rerun()


def _fmt(value: float) -> str:
    return f"{value:g}"


def _render_proposals(ctx: tools.ToolContext) -> None:
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
                    _sync(ctx)
                    st.rerun()
            if dismiss_col.button("Dismiss", key=f"dismiss_{proposal.id}"):
                tools.dismiss_proposal(ctx, proposal.id)
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


def render() -> None:
    st.title("Assistant")
    require_owner_page()

    if not config.anthropic_api_key():
        st.warning(
            "No ANTHROPIC_API_KEY is configured. Add one (in .env locally, or in the app's secrets on "
            "Streamlit Cloud) to enable the chat."
        )
        st.stop()

    api_messages, transcript, ctx, usage = _state()
    # What was last on screen on the Companies/Watchlist pages (None if never opened).
    ctx.screen_context = st.session_state.get(BROWSE_CONTEXT_KEY)

    st.caption(
        "Ask about any tracked company, or tell me how to tune its recommendation math. I use the app's own data "
        f"(refreshed daily), not live quotes. Model: {config.assistant_model()}. Not financial advice."
    )
    top_left, top_right = st.columns([4, 1])
    top_left.caption(f"Session usage: {usage.budget_used:,} of {chat.SESSION_TOKEN_BUDGET:,} budgeted tokens.")
    if top_right.button("Clear conversation", key="assistant_clear"):
        _clear()
        st.rerun()

    _render_adjustments_panel(ctx)

    if not transcript:
        st.markdown("**Try asking:**")
        for example in _EXAMPLES:
            st.markdown(f"- {example}")

    for message in transcript:
        with st.chat_message(message["role"]):
            st.markdown(safe_markdown(message["text"]))

    _render_proposals(ctx)
    _render_sync_status(ctx)

    exhausted = chat.over_budget(usage)
    if exhausted:
        st.info("This conversation reached its token budget. Clear it to start a new one.")
    prompt = st.chat_input("Ask a question or describe a change...", disabled=exhausted)
    if not prompt:
        return

    transcript.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(safe_markdown(prompt))

    try:
        client = chat.make_client()
    except Exception as exc:
        transcript.append({"role": "assistant", "text": f"Couldn't start the assistant: {type(exc).__name__}."})
        st.rerun()

    runner = chat.TurnRunner(client, api_messages, prompt, ctx, usage)
    raw_parts: list[str] = []

    def recorded() -> Iterator[str]:
        for chunk in runner.stream():
            raw_parts.append(chunk)
            yield chunk

    with st.chat_message("assistant"):
        st.write_stream(safe_stream(recorded()))
    # The RAW text is stored; safe_markdown is applied each time it's displayed.
    transcript.append({"role": "assistant", "text": "".join(raw_parts)})
    if runner.ok:
        st.session_state[_API] = runner.messages
    _sync(ctx)
    st.rerun()
