"""Lightweight "owner mode" gate for write actions (Add/Remove Watchlist,
Save price alert on dashboard/views/main.py and dashboard/views/watchlist.py).

This app has no real user accounts -- deployed publicly, anyone who
reaches the URL could otherwise call any write action with zero check
(see the security review that prompted this). Read-only browsing
(Main/Watchlist tables, Symbol Detail) stays open to everyone, so a
portfolio visitor can still see the whole app working -- only the two
mutating actions are gated behind a password only the owner knows.

If OWNER_PASSWORD isn't set (config.py::owner_password, e.g. local dev),
`is_unlocked` always returns True and `require_owner` runs the action
immediately -- this gate is opt-in, never a requirement just to run the
app at all.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from stock_predictor.config import owner_password

_SESSION_KEY = "is_owner"


def _password_matches(entered: str, configured: str | None) -> bool:
    """Pulled out as a pure function -- the actual comparison logic, easy
    to unit test without a live Streamlit session. A blank/empty entry
    never matches, even if OWNER_PASSWORD were somehow also blank
    (config.owner_password() already treats a blank env var as unset/None,
    so `configured` being falsy here means the gate should never have been
    shown in the first place -- fail closed regardless).
    """
    return bool(entered) and bool(configured) and entered == configured


def is_unlocked() -> bool:
    """True if no password is configured (gate disabled) or this browser
    session already entered the correct one via require_owner below.
    """
    if not owner_password():
        return True
    return bool(st.session_state.get(_SESSION_KEY, False))


@st.dialog("Owner password")
def _password_dialog(on_success: Callable[[], None]) -> None:
    st.caption("Only needed to change the Watchlist or price alerts. Anyone can still browse.")
    entered = st.text_input("Password", type="password", key="owner_password_input")
    if st.button("Unlock"):
        if _password_matches(entered, owner_password()):
            st.session_state[_SESSION_KEY] = True
            on_success()
            st.rerun()
        else:
            st.error("Incorrect password.")


def require_owner(on_success: Callable[[], None]) -> bool:
    """Call this instead of running a write action directly. Runs
    `on_success` immediately if already unlocked (or no password is
    configured) and returns True -- the caller should follow up with
    whatever it needs afterward (e.g. st.rerun()), same as if this gate
    didn't exist. Otherwise opens a password dialog (see module docstring
    for why this stays open across reruns until dismissed or unlocked) and
    returns False -- the dialog calls `on_success` (and its own
    st.rerun()) itself once the correct password is entered, so the
    caller should do nothing further this run.
    """
    if is_unlocked():
        on_success()
        return True
    _password_dialog(on_success)
    return False
