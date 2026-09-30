from __future__ import annotations

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
