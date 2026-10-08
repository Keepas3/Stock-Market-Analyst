from __future__ import annotations

import pytest

from stock_predictor.dashboard.auth import _password_matches


def test_password_matches_correct_password():
    assert _password_matches("hunter2", "hunter2") is True


def test_password_matches_rejects_wrong_password():
    assert _password_matches("wrong", "hunter2") is False


def test_password_matches_rejects_empty_entry():
    assert _password_matches("", "hunter2") is False


def test_password_matches_fails_closed_when_nothing_configured():
    """If owner_password() somehow returned a falsy value here (shouldn't
    happen -- is_unlocked() already treats "no password configured" as
    "gate disabled" and never shows the dialog), an empty entry must still
    never match an empty/None configured value.
    """
    assert _password_matches("", None) is False
    assert _password_matches("", "") is False


def test_lockout_after_too_many_recent_failures():
    from stock_predictor.dashboard.auth import _MAX_FAILURES, _lockout_seconds

    now = 1000.0
    assert _lockout_seconds([], now) == 0
    assert _lockout_seconds([now - 1] * (_MAX_FAILURES - 1), now) == 0
    assert _lockout_seconds([now - 1] * _MAX_FAILURES, now) > 0


def test_lockout_expires_once_failures_age_out():
    from stock_predictor.dashboard.auth import _FAILURE_WINDOW_SECONDS, _MAX_FAILURES, _lockout_seconds

    now = 10_000.0
    old = [now - _FAILURE_WINDOW_SECONDS - 5] * _MAX_FAILURES
    assert _lockout_seconds(old, now) == 0


# ---- strict owner check / shared unlock path -----------------------------------------------------------------


@pytest.fixture
def auth_env(monkeypatch):
    from stock_predictor.dashboard import auth

    session = {}
    monkeypatch.setattr(auth.st, "session_state", session)
    monkeypatch.setattr(auth, "_FAILED_ATTEMPTS", [])
    monkeypatch.setattr(auth, "owner_password", lambda: "hunter2")
    return auth, session


def test_is_owner_requires_a_configured_password_and_an_unlocked_session(auth_env, monkeypatch):
    auth, session = auth_env
    assert auth.is_owner() is False
    session["is_owner"] = True
    assert auth.is_owner() is True
    monkeypatch.setattr(auth, "owner_password", lambda: None)
    assert auth.is_owner() is False  # fails closed: no password configured, so nobody is "the owner"
    assert auth.is_unlocked() is True  # the older opt-in gate stays open, by design


def test_correct_password_unlocks_and_wrong_one_does_not(auth_env):
    auth, session = auth_env
    assert auth._try_unlock("wrong") == "Incorrect password."
    assert "is_owner" not in session
    assert auth._try_unlock("hunter2") is None
    assert session["is_owner"] is True


def test_unlock_fails_when_no_password_is_configured(auth_env, monkeypatch):
    auth, session = auth_env
    monkeypatch.setattr(auth, "owner_password", lambda: None)
    assert "No OWNER_PASSWORD" in auth._try_unlock("anything")
    assert "is_owner" not in session


def test_lockout_applies_even_to_the_correct_password(auth_env):
    auth, session = auth_env
    for _ in range(auth._MAX_FAILURES):
        auth._try_unlock("nope")
    message = auth._try_unlock("hunter2")
    assert "Too many incorrect attempts" in message
    assert "is_owner" not in session
