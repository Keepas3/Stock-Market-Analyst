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
